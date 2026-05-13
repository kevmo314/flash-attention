# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SM120 (Blackwell GeForce / RTX Blackwell) forward pass.
#
# SM120 uses the same SM80-era MMA instructions (mma.sync.aligned.m16n8k16) but has
# a smaller shared memory capacity (99 KB vs 163 KB on SM80). This module subclasses
# FlashAttentionForwardSm80, overrides the SMEM capacity check, and specializes the
# forward mainloop for the single-stage SM120 configurations selected by interface.py.

from types import SimpleNamespace
from typing import Callable, Optional

import cutlass.cute as cute
import cutlass
import cutlass.utils as utils_basic
from cutlass import Float32, Int32, const_expr
from cutlass.base_dsl.arch import Arch

from quack import layout_utils

from flash_attn.cute import ampere_helpers as sm80_utils
from flash_attn.cute.flash_fwd import FlashAttentionForwardSm80
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.softmax import Softmax


class FlashAttentionForwardSm120(FlashAttentionForwardSm80):
    # Keep arch = 80 to use CpAsync code paths (no TMA for output).
    # The compilation target is determined by the GPU at compile time, not this field.
    arch = 80

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Base __init__ records the real runtime arch.  The SM120 compatibility
        # kernel still uses SM80-style output stores, so keep the inherited
        # epilogue off the SM90+ TMA-O branch.
        self.arch = Arch.sm_80
        # PackGQA is an optimization; disabling it keeps GQA/MQA on the regular
        # head layout instead of the SM90/SM100 packed layout path.
        self.pack_gqa = False

    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        num_stages,
        num_threads,
        is_causal,
        Q_in_regs=False,
    ) -> bool:
        """Check if the kernel can be implemented on SM120.

        Same logic as SM80 but uses SM120's shared memory capacity (99 KB).
        """
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if head_dim_v % 8 != 0:
            return False
        if tile_n % 16 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        # Shared memory usage: Q tile + (K tile + V tile)
        smem_usage_Q = tile_m * head_dim * 2
        smem_usage_K = tile_n * head_dim * num_stages * 2
        smem_usage_V = tile_n * head_dim_v * num_stages * 2
        smem_usage_QV = (
            (smem_usage_Q + smem_usage_V) if not Q_in_regs else max(smem_usage_Q, smem_usage_V)
        )
        smem_usage = smem_usage_QV + smem_usage_K
        # SM120 has 99 KB shared memory (vs 163 KB on SM80)
        smem_capacity = utils_basic.get_smem_capacity_in_bytes("sm_120")
        if smem_usage > smem_capacity:
            return False
        if (tile_m * 2) % num_threads != 0:
            return False
        return True

    @cute.jit
    def compute_one_n_block(
        self,
        n_block: Int32,
        smem_pipe_read: Int32,
        smem_pipe_write: Int32,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        load_K: Callable,
        load_V: Callable,
        score_mod: Callable | None,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        seqlen: SeqlenInfoQK,
        aux_tensors=None,
        fastdiv_mods=None,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
    ):
        """SM120 single-stage forward mainloop.

        The SM120 forward path always launches with one K/V stage to stay within
        the 99 KB per-CTA shared-memory limit. The SM80 base mainloop carries
        multistage branches and circular-buffer bookkeeping that are dead for
        SM120; spelling out the single-stage ordering removes that overhead from
        the hot n-block loop.
        """
        assert self.num_stages == 1, "SM120 native mainloop currently expects a single K/V stage"

        acc_shape_S = mma_params.thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        acc_S = cute.make_fragment(acc_shape_S, Float32)
        acc_S.fill(0.0)

        # Wait for the prologue/current K load before the QK matmul.  V for this
        # block is issued before QK so it can overlap with score computation.
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        load_V(n_block, smem_pipe_write, need_predicates=is_first_n_block)
        cute.arch.cp_async_commit_group()

        sm80_utils.gemm(
            mma_params.thr_mma_qk,
            acc_S,
            mma_params.tSrQ,
            mma_params.tSrK,
            smem_copy_params.tSsQ,
            smem_copy_params.tSsK[None, None, None, 0],
            smem_copy_params.smem_thr_copy_Q,
            smem_copy_params.smem_thr_copy_K,
            A_in_regs=self.Q_in_regs,
        )
        if const_expr(score_mod is not None):
            self.apply_score_mod(
                mma_params.thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                acc_S,
                n_block,
                softmax_scale=softmax.softmax_scale,
                seqlen=seqlen,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
            )

        # V must be resident before PV.  Once QK is done, K's smem tile can be
        # reused for the next n-block load while mask/softmax/PV runs.
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        if n_block >= 1:
            load_K(n_block - 1, smem_pipe_write, need_predicates=False)
        cute.arch.cp_async_commit_group()

        if const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=check_inf)
        softmax.rescale_O(mma_params.acc_O, row_scale)

        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        tOrP = layout_utils.reshape_acc_to_frgA(rP)
        sm80_utils.gemm_rs(
            mma_params.thr_mma_pv,
            mma_params.acc_O,
            tOrP,
            mma_params.tOrVt,
            smem_copy_params.tOsVt[None, None, None, 0],
            smem_copy_params.smem_thr_copy_V,
        )
