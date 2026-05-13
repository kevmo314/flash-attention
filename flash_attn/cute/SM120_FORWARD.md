# SM120 Forward Kernel Direction

This branch currently enables RTX PRO 6000 / SM120 by routing forward through a
FlashAttentionForwardSm80-derived CuTe kernel with SM120 shared-memory limits.
That is correct as a compatibility path, but it is not the path that should
deliver a large FA4-over-FA2 performance gap.

## Hardware and CuTe Constraints

SM120 is Blackwell, but it is not the same target as datacenter Blackwell SM100.
Do not route SM120 through the SM100 FA4 `tcgen05` / TMEM kernel path.

Relevant constraints from NVIDIA/CUTLASS:

- RTX PRO 6000 Blackwell is compute capability 12.0:
  https://developer.nvidia.com/cuda/gpus
- Compute capability 12.0 has 128 KB shared memory per SM and 99 KB maximum
  shared memory per thread block:
  https://docs.nvidia.com/cuda/archive/12.8.0/blackwell-tuning-guide/index.html
- CUTLASS notes that SM100 datacenter Blackwell and SM120 GeForce/RTX Blackwell
  are different compute capabilities, and SM100 architecture-accelerated kernels
  are not compatible with RTX 50 / SM120 targets:
  https://github.com/NVIDIA/cutlass
- NVIDIA's public guidance for SM12x is that `tcgen05` / TMEM is SM100/SM110,
  while SM120 needs the supported SM12x MMA pipeline instead:
  https://forums.developer.nvidia.com/t/dearest-cutlass-team-when-the-hell-are-you-going-to-properly-fix-tcgen05-fp4-support-for-dgx-spark-gb10-sm121/359598

The installed CuTe DSL enforces the same split:

- `cutlass.cute.nvgpu.tcgen05` MMA ops accept `sm_100f` / `sm_110f` families,
  not `sm_120`.
- `cutlass.cute.nvgpu.warpgroup` MMA ops are restricted to `sm_90a`.
- TMA copy atoms are available for architectures `>= sm_90`, including SM120,
  with CTA group 1.
- The SM120 GeForce-style CUTLASS example uses warp-level
  `cute.nvgpu.warp.MmaF16BF16Op`, TMA GMEM-to-SMEM copies, and a dedicated DMA
  warp. That is the closest template for an SM120-native attention mainloop.

## Target Design

The next forward kernel should be a new SM120-specific path rather than an
SM100 subclass:

1. Keep the current SM80-derived SM120 path as fallback.
2. Add a new SM120 TMA forward kernel for the dense contiguous fast path:
   BF16/FP16, no paged KV, no block sparsity, no SplitKV, no qv, no custom mask.
3. Use CTA group 1 TMA loads for Q/K/V where layouts are TMA-friendly.
4. Start with non-TMA O stores if residue/predication makes TMA stores awkward;
   add TMA O later for aligned dense cases.
5. Use a dedicated DMA warp plus MMA warps, following the SM120 GeForce GEMM
   structure, instead of the current all-warps cp.async SM80-style mainloop.
6. Preserve current SM120 tuned tiles as initial candidates:
   - `head_dim <= 64`: `(128, 128)` noncausal, `(128, 64)` causal/local
   - `head_dim <= 128`: `(128, 32)` noncausal, `(64, 64)` causal/local
   - `head_dim >= 192`: `(64, 32)`
7. Re-autotune tiles after the TMA mainloop exists. The current optimal tile
   choices are specific to the SM80-style cp.async implementation.

## Why This Should Help

The current SM120 path leaves most of FA4's Blackwell advantage unused. It still
uses warp-level SM80-style MMA and per-thread cp.async loads. A proper SM120 path
cannot use SM100 `tcgen05`, but it can still use SM120-supported TMA to reduce
load overhead and dedicate warps to data movement. That is the plausible source
of additional performance beyond the tile tuning already committed.

## Validation Plan

For each milestone:

1. Numerically compare against PyTorch SDPA and FA2 on BF16.
2. Run focused `tests/cute/test_flash_attn.py::test_flash_attn_output` cases for
   `head_dim` 64, 128, 192, and 256, causal and noncausal, MHA and GQA.
3. Benchmark RTX PRO 6000 against FA2 for total sequence length 32768 at
   sequence lengths 4096 and 8192.
4. Keep the fallback path available until the SM120 TMA path is faster and
   correct for a broad shape set.
