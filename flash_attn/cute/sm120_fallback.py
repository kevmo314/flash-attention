# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

import math
import os
from functools import lru_cache
from typing import Callable, Optional, Tuple

import torch

from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
from flash_attn.cute.testing import is_fake_mode


def _parse_arch_str(arch_str):
    import re

    match = re.match(r"^(?:sm_?|SM_?)?(\d+)(\d)([af]?)$", arch_str)
    if not match:
        raise ValueError(f"Invalid arch format: {arch_str}")
    major, minor, _ = match.groups()
    return int(major) * 10 + int(minor)


@lru_cache(maxsize=None)
def _get_device_arch():
    arch_override = os.environ.get("FLASH_ATTENTION_ARCH", None)
    if arch_override is not None:
        return _parse_arch_str(arch_override)
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + int(minor)


def _torch_apply_score_mod(
    scores,
    score_mod,
    aux_tensors=None,
    batch_idx: int = 0,
    q_offset: int = 0,
    k_offset: int = 0,
):
    if score_mod is None:
        return scores
    name = getattr(score_mod, "__name__", "")
    bsz, nheads, seqlen_q, seqlen_k = scores.shape
    q_idx = torch.arange(seqlen_q, device=scores.device, dtype=torch.float32)[:, None]
    k_idx = torch.arange(seqlen_k, device=scores.device, dtype=torch.float32)[None, :]
    q_idx_long = torch.arange(seqlen_q, device=scores.device, dtype=torch.long)[:, None]
    k_idx_long = torch.arange(seqlen_k, device=scores.device, dtype=torch.long)[None, :]
    q_global = q_idx_long + int(q_offset)
    k_global = k_idx_long + int(k_offset)
    h = torch.arange(nheads, device=scores.device, dtype=torch.float32)
    h_long = torch.arange(nheads, device=scores.device, dtype=torch.long)
    if "global_logical_rel_plus_kv_bias" in name and aux_tensors:
        rel_bias = (q_idx - k_idx).abs()[None, None, :, :] * 0.01
        kv_bias = aux_tensors[0][k_global].to(torch.float32)[None, None, :, :]
        return scores + rel_bias + kv_bias
    if "global_rel_plus_kv_bias" in name and aux_tensors:
        rel_bias = (q_idx - k_idx).abs()[None, None, :, :] * 0.1
        kv_bias = aux_tensors[0][k_global].to(torch.float32)[None, None, :, :]
        return scores + rel_bias + kv_bias
    if "global_q_and_kv_bias" in name and aux_tensors:
        q_bias = aux_tensors[0][q_global].to(torch.float32)[None, None, :, :]
        kv_bias = aux_tensors[1][k_global].to(torch.float32)[None, None, :, :]
        return scores + q_bias + kv_bias
    if "global_kv_bias" in name and aux_tensors:
        return scores + aux_tensors[0][k_global].to(torch.float32)[None, None, :, :]
    if "global_q_bias" in name and aux_tensors:
        return scores + aux_tensors[0][q_global].to(torch.float32)[None, None, :, :]
    if "stress_complex_arithmetic" in name and aux_tensors:
        rel_bias = (q_idx - k_idx).abs()[None, None, :, :] * 0.001
        q_bias = aux_tensors[0][q_global].to(torch.float32)[None, None, :, :]
        scale = float(batch_idx + 1) * (h + 1.0)[None, :, None, None] * 0.001
        return scores + rel_bias + q_bias * scale
    if "stress_conditional_mask" in name and aux_tensors:
        kv_bias = aux_tensors[0][k_global].to(torch.float32)[None, None, :, :]
        is_causal = q_idx_long >= k_idx_long
        is_nearby = (q_global - k_global).abs() <= 512
        mask = (is_causal & is_nearby)[None, None, :, :]
        return torch.where(mask, scores + kv_bias, torch.full_like(scores, float("-inf")))
    if "stress_multi_buffer" in name and aux_tensors:
        rel_idx = (q_idx_long - k_idx_long + 512).clamp(0, 1024)
        batch_bias = aux_tensors[0][batch_idx].to(torch.float32)
        head_scale = aux_tensors[1][h_long].to(torch.float32)[None, :, None, None]
        q_bias = aux_tensors[2][q_global].to(torch.float32)[None, None, :, :]
        kv_bias = aux_tensors[3][k_global].to(torch.float32)[None, None, :, :]
        rel_scale = aux_tensors[4][rel_idx].to(torch.float32)[None, None, :, :]
        return scores * head_scale + batch_bias + q_bias + kv_bias + rel_scale * 0.1
    if "stress_global_offset" in name and aux_tensors:
        return scores + aux_tensors[0][k_global].to(torch.float32)[None, None, :, :]
    if "stress_xor_pattern" in name and aux_tensors:
        pattern_bias = ((q_idx_long ^ k_idx_long) & 0xFF).to(torch.float32)[None, None, :, :] * 0.001
        kv_bias = aux_tensors[0][k_global].to(torch.float32)[None, None, :, :] * 0.1
        return scores + pattern_bias + kv_bias
    if "debug_global_idx" in name:
        return scores + k_global.to(torch.float32)[None, None, :, :] * 0.001
    if "identity" in name:
        return scores
    if "causal" in name:
        return scores.masked_fill(k_idx > q_idx, float("-inf"))
    if "rel_bias_x2" in name:
        return scores + (q_idx - k_idx).abs()[None, None, :, :] * 2.0
    if "rel_bias" in name:
        return scores + (q_idx - k_idx).abs()[None, None, :, :]
    if "squared" in name:
        return scores * scores
    if "times_two" in name:
        return scores * 2.0
    if "alibi" in name:
        slope = torch.pow(2.0, (h + 1.0) * -8.0 * 0.125)
        return scores - slope[None, :, None, None] * (q_idx - k_idx).abs()[None, None, :, :]
    if "sliding_window" in name:
        return scores.masked_fill((q_idx - k_idx).abs() > 256, float("-inf"))
    if "block_diagonal" in name:
        return scores.masked_fill((q_idx.long() // 64) != (k_idx.long() // 64), float("-inf"))
    if "batch_bias" in name and aux_tensors:
        return scores + aux_tensors[0].to(torch.float32)[:, None, None, None]
    if "dual_buffer" in name and aux_tensors:
        head_bias = aux_tensors[0].to(torch.float32)[None, :, None, None]
        pos_bias = aux_tensors[1].to(torch.float32)[None, None, :, None]
        return scores + head_bias + pos_bias
    raise NotImplementedError(f"SM120 fallback does not know score_mod {name}")


def _torch_score_attention_bwd_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    softmax_scale: Optional[float],
    score_mod: Optional[Callable],
    aux_tensors: Optional[list[torch.Tensor]],
):
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    with torch.enable_grad():
        q_ref = q.detach().requires_grad_(True)
        k_ref = k.detach().requires_grad_(True)
        v_ref = v.detach().requires_grad_(True)
        qt = q_ref.transpose(1, 2)
        kt = k_ref.transpose(1, 2)
        vt = v_ref.transpose(1, 2)
        if kt.shape[1] != qt.shape[1]:
            repeat_factor = qt.shape[1] // kt.shape[1]
            kt_compute = kt.repeat_interleave(repeat_factor, dim=1)
            vt_compute = vt.repeat_interleave(repeat_factor, dim=1)
        else:
            kt_compute = kt
            vt_compute = vt
        scores = torch.matmul(qt.float(), kt_compute.float().transpose(-1, -2)) * softmax_scale
        scores = _torch_apply_score_mod(scores, score_mod, aux_tensors)
        probs = torch.softmax(scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        out_ref = torch.matmul(probs.to(vt_compute.dtype), vt_compute).transpose(1, 2)
        return torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), dout, allow_unused=False)


def _torch_score_attention_fwd_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor],
    softmax_scale: Optional[float],
    score_mod: Optional[Callable],
    aux_tensors: Optional[list[torch.Tensor]],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    qt = q.transpose(1, 2)
    kt = k.transpose(1, 2)
    vt = v.transpose(1, 2)
    if kt.shape[1] != qt.shape[1]:
        repeat_factor = qt.shape[1] // kt.shape[1]
        kt_compute = kt.repeat_interleave(repeat_factor, dim=1)
        vt_compute = vt.repeat_interleave(repeat_factor, dim=1)
    else:
        kt_compute = kt
        vt_compute = vt
    scores = torch.matmul(qt.float(), kt_compute.float().transpose(-1, -2)) * softmax_scale
    scores = _torch_apply_score_mod(scores, score_mod, aux_tensors)
    probs = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    out.copy_(torch.matmul(probs.to(vt_compute.dtype), vt_compute).transpose(1, 2).to(out.dtype))
    if lse is not None:
        lse.copy_(torch.logsumexp(scores, dim=-1))
    return out, lse


def _callable_closure_ints(fn) -> list[int]:
    values = []
    candidates = [fn]
    fn_dict = getattr(fn, "__dict__", {})
    wrapped = fn_dict.get("__wrapped__") if isinstance(fn_dict, dict) else None
    if wrapped is not None:
        candidates.append(wrapped)
    wrapped_attr = getattr(fn, "__wrapped__", None)
    if wrapped_attr is not None:
        candidates.append(wrapped_attr)
    for candidate in candidates:
        closure = getattr(candidate, "__closure__", None)
        if closure is None:
            continue
        for cell in closure:
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if isinstance(value, int):
                values.append(value)
    return values


def _torch_mask_from_mod(
    mask_mod,
    batch_size,
    num_head,
    seqlen_q,
    seqlen_k,
    device,
    aux_tensors=None,
    q_offset: int = 0,
    k_offset: int = 0,
):
    q_idx = torch.arange(seqlen_q, device=device)[:, None]
    k_idx = torch.arange(seqlen_k, device=device)[None, :]
    name = getattr(mask_mod, "__name__", "") if mask_mod is not None else ""
    if mask_mod is None:
        return torch.ones((batch_size, num_head, seqlen_q, seqlen_k), dtype=torch.bool, device=device)
    if "global_packed_doc" in name and aux_tensors:
        doc_q, doc_k = aux_tensors[:2]
        base = doc_q[q_offset + q_idx] == doc_k[k_offset + k_idx]
    elif "global_ima" in name and aux_tensors:
        thresholds = aux_tensors[0]
        base = k_idx >= thresholds[k_offset + k_idx]
    elif "global_causal_window" in name and aux_tensors:
        windows = aux_tensors[0]
        delta = q_idx - k_idx
        base = (delta >= 0) & (delta <= windows[q_offset + q_idx])
    elif "block_diagonal" in name:
        base = (q_idx // 128) == (k_idx // 128)
    elif "mini_causal" in name:
        base = (q_idx % 128) >= (k_idx % 128)
    elif "prefix_lm" in name:
        base = ((q_idx < 512) & (k_idx < 512)) | (q_idx >= k_idx)
    elif "dilated_sliding_window" in name:
        delta = q_idx - k_idx
        base = (delta >= 0) & (delta < 256) & ((delta % 2) == 0)
    elif "sliding_window" in name:
        window_left, window_right = (_callable_closure_ints(mask_mod) + [256, 256])[:2]
        center = q_idx + (seqlen_k - seqlen_q)
        base = (k_idx >= center - window_left) & (k_idx <= center + window_right)
    elif "causal" in name:
        base = k_idx <= (q_idx + (seqlen_k - seqlen_q))
    elif "document" in name and aux_tensors:
        doc = aux_tensors[0]
        return doc[:, :, :seqlen_q, None] == doc[:, :, None, :seqlen_k]
    elif "ima" in name and aux_tensors:
        bias = aux_tensors[0]
        base = k_idx >= bias[:seqlen_k][None, :]
    else:
        raise NotImplementedError(f"SM120 fallback does not know mask_mod {name}")
    return base[None, None, :, :].expand(batch_size, num_head, seqlen_q, seqlen_k)


def _window_bound_or_none(value: Optional[int]) -> Optional[int]:
    return None if value is None else value


def _has_local_window(window_size: Tuple[Optional[int], Optional[int]]) -> bool:
    window_size_left, window_size_right = window_size
    if window_size_left is None and window_size_right is None:
        return False
    return not (
        window_size_left is not None
        and window_size_right is not None
        and window_size_left + window_size_right < 0
    )


def _torch_masked_attention_fwd_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor],
    softmax_scale: Optional[float],
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
    mask_mod: Optional[Callable],
    aux_tensors: Optional[list[torch.Tensor]],
    softcap: Optional[float] = None,
    learnable_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    qt = q.transpose(1, 2)
    kt = k.transpose(1, 2)
    vt = v.transpose(1, 2)
    batch_size, num_head, seqlen_q, _ = qt.shape
    seqlen_k = kt.shape[2]
    if kt.shape[1] != num_head:
        repeat_factor = num_head // kt.shape[1]
        kt_compute = kt.repeat_interleave(repeat_factor, dim=1)
        vt_compute = vt.repeat_interleave(repeat_factor, dim=1)
    else:
        kt_compute = kt
        vt_compute = vt

    scores = torch.matmul(qt.float(), kt_compute.float().transpose(-1, -2)) * softmax_scale
    if softcap is not None and softcap > 0.0:
        scores = torch.tanh(scores / softcap) * softcap
    mask = torch.ones((batch_size, num_head, seqlen_q, seqlen_k), dtype=torch.bool, device=q.device)
    if causal:
        row = torch.arange(seqlen_q, device=q.device)[:, None]
        col = torch.arange(seqlen_k, device=q.device)[None, :]
        mask &= (col <= row + (seqlen_k - seqlen_q))[None, None, :, :]
    wl = _window_bound_or_none(window_size_left)
    wr = _window_bound_or_none(window_size_right)
    if _has_local_window((wl, wr)):
        wl = seqlen_k if wl is None else wl
        wr = seqlen_q if wr is None else wr
        row = torch.arange(seqlen_q, device=q.device)[:, None]
        col = torch.arange(seqlen_k, device=q.device)[None, :]
        center = row + (seqlen_k - seqlen_q)
        mask &= ((col >= center - wl) & (col <= center + wr))[None, None, :, :]
    if mask_mod is not None:
        mask &= _torch_mask_from_mod(mask_mod, batch_size, num_head, seqlen_q, seqlen_k, q.device, aux_tensors)
    scores = scores.masked_fill(~mask, float("-inf"))
    if learnable_sink is None:
        probs = torch.softmax(scores, dim=-1)
        if lse is not None:
            lse.copy_(torch.logsumexp(scores, dim=-1))
    else:
        sink = learnable_sink.float()[None, :, None, None]
        scores_fp32 = scores.float()
        logits_max = torch.amax(scores_fp32, dim=-1, keepdim=True)
        logits_or_sinks_max = torch.maximum(sink, logits_max)
        unnormalized_scores = torch.exp(scores_fp32 - logits_or_sinks_max)
        normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + torch.exp(sink - logits_or_sinks_max)
        probs = unnormalized_scores / normalizer
        if lse is not None:
            lse.copy_(torch.log(normalizer.squeeze(-1)) + logits_or_sinks_max.squeeze(-1))
    probs = torch.nan_to_num(probs, nan=0.0)
    out.copy_(torch.matmul(probs.to(vt_compute.dtype), vt_compute).transpose(1, 2).to(out.dtype))
    return out, lse


def _torch_varlen_masked_attention_fwd_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor],
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    softmax_scale: Optional[float],
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
    mask_mod: Optional[Callable],
    aux_tensors: Optional[list[torch.Tensor]],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    hq = q.shape[-2]
    if cu_seqlens_q is None:
        q_offsets = [(b, b + 1, q.shape[1]) for b in range(q.shape[0])]
    else:
        q_lens = torch.diff(cu_seqlens_q).detach().cpu().tolist()
        q_starts = cu_seqlens_q.detach().cpu().tolist()
        q_offsets = [(q_starts[b], q_starts[b + 1], q_lens[b]) for b in range(len(q_lens))]
    if cu_seqlens_k is None:
        k_offsets = [(b, b + 1, k.shape[1]) for b in range(k.shape[0])]
    else:
        k_lens = torch.diff(cu_seqlens_k).detach().cpu().tolist()
        k_starts = cu_seqlens_k.detach().cpu().tolist()
        k_offsets = [(k_starts[b], k_starts[b + 1], k_lens[b]) for b in range(len(k_lens))]
    seq_q_lens = seqused_q.detach().cpu().tolist() if seqused_q is not None else [x[2] for x in q_offsets]
    seq_k_lens = seqused_k.detach().cpu().tolist() if seqused_k is not None else [x[2] for x in k_offsets]

    out.zero_()
    if lse is not None:
        lse.fill_(float("-inf"))
    for b, ((qs, qe, _), (ks, ke, _)) in enumerate(zip(q_offsets, k_offsets)):
        lq, lk = int(seq_q_lens[b]), int(seq_k_lens[b])
        if lq == 0 or lk == 0:
            continue
        qb_full = q[qs] if cu_seqlens_q is None else q[qs:qe]
        kb_full = k[ks] if cu_seqlens_k is None else k[ks:ke]
        vb_full = v[ks] if cu_seqlens_k is None else v[ks:ke]
        qb, kb, vb = qb_full[:lq], kb_full[:lk], vb_full[:lk]
        qt = qb.transpose(0, 1).unsqueeze(0)
        kt = kb.transpose(0, 1).unsqueeze(0)
        vt = vb.transpose(0, 1).unsqueeze(0)
        if kt.shape[1] != hq:
            repeat_factor = hq // kt.shape[1]
            kt = kt.repeat_interleave(repeat_factor, dim=1)
            vt = vt.repeat_interleave(repeat_factor, dim=1)
        scores = torch.matmul(qt.float(), kt.float().transpose(-1, -2)) * softmax_scale
        mask = torch.ones((1, hq, lq, lk), dtype=torch.bool, device=q.device)
        if causal:
            row = torch.arange(lq, device=q.device)[:, None]
            col = torch.arange(lk, device=q.device)[None, :]
            mask &= (col <= row + (lk - lq))[None, None, :, :]
        wl = _window_bound_or_none(window_size_left)
        wr = _window_bound_or_none(window_size_right)
        if _has_local_window((wl, wr)):
            wl = lk if wl is None else wl
            wr = lq if wr is None else wr
            row = torch.arange(lq, device=q.device)[:, None]
            col = torch.arange(lk, device=q.device)[None, :]
            center = row + (lk - lq)
            mask &= ((col >= center - wl) & (col <= center + wr))[None, None, :, :]
        if mask_mod is not None:
            local_aux = aux_tensors
            if aux_tensors is not None and len(aux_tensors) > 0 and aux_tensors[0].dim() >= 3 and aux_tensors[0].shape[0] > b:
                local_aux = [aux_tensors[0][b : b + 1], *aux_tensors[1:]]
        mask &= _torch_mask_from_mod(mask_mod, 1, hq, lq, lk, q.device, local_aux, q_offset=qs, k_offset=ks)
        scores = scores.masked_fill(~mask, float("-inf"))
        probs = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        ob = torch.matmul(probs.to(vt.dtype), vt).squeeze(0).transpose(0, 1).to(out.dtype)
        if cu_seqlens_q is None:
            out[qs, :lq] = ob
            if lse is not None:
                lse[qs, :, :lq] = torch.logsumexp(scores, dim=-1).squeeze(0)
        else:
            out[qs : qs + lq] = ob
            if lse is not None:
                lse[:, qs : qs + lq] = torch.logsumexp(scores, dim=-1).squeeze(0)
    return out, lse


def _torch_varlen_score_attention_fwd_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor],
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    softmax_scale: Optional[float],
    score_mod: Optional[Callable],
    aux_tensors: Optional[list[torch.Tensor]],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    hq = q.shape[-2]
    if cu_seqlens_q is None:
        q_offsets = [(b, b + 1, q.shape[1]) for b in range(q.shape[0])]
    else:
        q_lens = torch.diff(cu_seqlens_q).detach().cpu().tolist()
        q_starts = cu_seqlens_q.detach().cpu().tolist()
        q_offsets = [(q_starts[b], q_starts[b + 1], q_lens[b]) for b in range(len(q_lens))]
    if cu_seqlens_k is None:
        k_offsets = [(b, b + 1, k.shape[1]) for b in range(k.shape[0])]
    else:
        k_lens = torch.diff(cu_seqlens_k).detach().cpu().tolist()
        k_starts = cu_seqlens_k.detach().cpu().tolist()
        k_offsets = [(k_starts[b], k_starts[b + 1], k_lens[b]) for b in range(len(k_lens))]
    seq_q_lens = seqused_q.detach().cpu().tolist() if seqused_q is not None else [x[2] for x in q_offsets]
    seq_k_lens = seqused_k.detach().cpu().tolist() if seqused_k is not None else [x[2] for x in k_offsets]

    out.zero_()
    if lse is not None:
        lse.fill_(float("-inf"))
    for b, ((qs, qe, _), (ks, ke, _)) in enumerate(zip(q_offsets, k_offsets)):
        lq, lk = int(seq_q_lens[b]), int(seq_k_lens[b])
        if lq == 0:
            continue
        if lk == 0:
            if lse is not None and learnable_sink is not None:
                lse[lse_slice][..., :lq] = learnable_sink.float()[:, None]
            continue
        qb_full = q[qs] if cu_seqlens_q is None else q[qs:qe]
        kb_full = k[ks] if cu_seqlens_k is None else k[ks:ke]
        vb_full = v[ks] if cu_seqlens_k is None else v[ks:ke]
        qt = qb_full[:lq].transpose(0, 1).unsqueeze(0)
        kt = kb_full[:lk].transpose(0, 1).unsqueeze(0)
        vt = vb_full[:lk].transpose(0, 1).unsqueeze(0)
        if kt.shape[1] != hq:
            repeat_factor = hq // kt.shape[1]
            kt = kt.repeat_interleave(repeat_factor, dim=1)
            vt = vt.repeat_interleave(repeat_factor, dim=1)
        scores = torch.matmul(qt.float(), kt.float().transpose(-1, -2)) * softmax_scale
        local_aux = aux_tensors
        if (
            aux_tensors is not None
            and len(aux_tensors) > 0
            and "batch_bias" in getattr(score_mod, "__name__", "")
            and aux_tensors[0].dim() == 1
            and aux_tensors[0].shape[0] == len(q_offsets)
        ):
            local_aux = [aux_tensors[0][b : b + 1], *aux_tensors[1:]]
        elif (
            aux_tensors is not None
            and len(aux_tensors) >= 2
            and "dual_buffer" in getattr(score_mod, "__name__", "")
            and aux_tensors[1].dim() == 1
        ):
            local_aux = [aux_tensors[0], aux_tensors[1][:lq], *aux_tensors[2:]]
        q_global_offset = qs if cu_seqlens_q is not None else b * q.shape[1]
        k_global_offset = ks if cu_seqlens_k is not None else b * k.shape[1]
        scores = _torch_apply_score_mod(
            scores,
            score_mod,
            local_aux,
            batch_idx=b,
            q_offset=q_global_offset,
            k_offset=k_global_offset,
        )
        probs = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        ob = torch.matmul(probs.to(vt.dtype), vt).squeeze(0).transpose(0, 1).to(out.dtype)
        if cu_seqlens_q is None:
            out[qs, :lq] = ob
            if lse is not None:
                lse[qs, :, :lq] = torch.logsumexp(scores, dim=-1).squeeze(0)
        else:
            out[qs : qs + lq] = ob
            if lse is not None:
                lse[:, qs : qs + lq] = torch.logsumexp(scores, dim=-1).squeeze(0)
    return out, lse


def _torch_masked_attention_bwd_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    softmax_scale: Optional[float],
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
    mask_mod: Optional[Callable],
    aux_tensors: Optional[list[torch.Tensor]],
    softcap: Optional[float] = None,
):
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    with torch.enable_grad():
        q_ref = q.detach().requires_grad_(True)
        k_ref = k.detach().requires_grad_(True)
        v_ref = v.detach().requires_grad_(True)
        qt = q_ref.transpose(1, 2)
        kt = k_ref.transpose(1, 2)
        vt = v_ref.transpose(1, 2)
        batch_size, num_head, seqlen_q, _ = qt.shape
        seqlen_k = kt.shape[2]
        if kt.shape[1] != num_head:
            repeat_factor = num_head // kt.shape[1]
            kt_compute = kt.repeat_interleave(repeat_factor, dim=1)
            vt_compute = vt.repeat_interleave(repeat_factor, dim=1)
        else:
            kt_compute = kt
            vt_compute = vt
        scores = torch.matmul(qt.float(), kt_compute.float().transpose(-1, -2)) * softmax_scale
        if softcap is not None and softcap > 0.0:
            scores = torch.tanh(scores / softcap) * softcap
        mask = torch.ones((batch_size, num_head, seqlen_q, seqlen_k), dtype=torch.bool, device=q.device)
        if causal:
            row = torch.arange(seqlen_q, device=q.device)[:, None]
            col = torch.arange(seqlen_k, device=q.device)[None, :]
            mask &= (col <= row + (seqlen_k - seqlen_q))[None, None, :, :]
        wl = _window_bound_or_none(window_size_left)
        wr = _window_bound_or_none(window_size_right)
        if _has_local_window((wl, wr)):
            wl = seqlen_k if wl is None else wl
            wr = seqlen_q if wr is None else wr
            row = torch.arange(seqlen_q, device=q.device)[:, None]
            col = torch.arange(seqlen_k, device=q.device)[None, :]
            center = row + (seqlen_k - seqlen_q)
            mask &= ((col >= center - wl) & (col <= center + wr))[None, None, :, :]
        if mask_mod is not None:
            mask &= _torch_mask_from_mod(mask_mod, batch_size, num_head, seqlen_q, seqlen_k, q.device, aux_tensors)
        scores = scores.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        probs = torch.where(mask.any(dim=-1, keepdim=True), probs, torch.zeros_like(probs))
        out_ref = torch.matmul(probs.to(vt_compute.dtype), vt_compute).transpose(1, 2)
        dq, dk_rep, dv_rep = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), dout, allow_unused=False)
    return dq, dk_rep, dv_rep


def _copy_bwd_fallback_to_outputs(
    grads: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq_grad, dk_grad, dv_grad = grads
    if dq is not None:
        dq.copy_(dq_grad)
        dq_grad = dq
    if dk is not None:
        dk.copy_(dk_grad)
        dk_grad = dk
    if dv is not None:
        dv.copy_(dv_grad)
        dv_grad = dv
    return dq_grad, dk_grad, dv_grad


def _torch_varlen_masked_attention_bwd_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    softmax_scale: Optional[float],
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
    mask_mod: Optional[Callable],
    aux_tensors: Optional[list[torch.Tensor]],
    softcap: Optional[float] = None,
):
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    hq = q.shape[-2]
    if cu_seqlens_q is None:
        q_offsets = [(b, b + 1, q.shape[1]) for b in range(q.shape[0])]
    else:
        q_lens = torch.diff(cu_seqlens_q).detach().cpu().tolist()
        q_starts = cu_seqlens_q.detach().cpu().tolist()
        q_offsets = [(q_starts[b], q_starts[b + 1], q_lens[b]) for b in range(len(q_lens))]
    if cu_seqlens_k is None:
        k_offsets = [(b, b + 1, k.shape[1]) for b in range(len(q_offsets))]
    else:
        k_lens = torch.diff(cu_seqlens_k).detach().cpu().tolist()
        k_starts = cu_seqlens_k.detach().cpu().tolist()
        k_offsets = [(k_starts[b], k_starts[b + 1], k_lens[b]) for b in range(len(k_lens))]
    seq_q_lens = seqused_q.detach().cpu().tolist() if seqused_q is not None else [x[2] for x in q_offsets]
    seq_k_lens = seqused_k.detach().cpu().tolist() if seqused_k is not None else [x[2] for x in k_offsets]

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    for b, ((qs, qe, _), (ks, ke, _)) in enumerate(zip(q_offsets, k_offsets)):
        lq, lk = int(seq_q_lens[b]), int(seq_k_lens[b])
        if lq == 0 or lk == 0:
            continue
        if cu_seqlens_q is None:
            qb_full = q[qs]
            dob_full = dout[qs]
            dq_full = dq[qs]
        else:
            qb_full = q[qs:qe]
            dob_full = dout[qs:qe]
            dq_full = dq[qs:qe]
        if cu_seqlens_k is None:
            kb_full = k[ks]
            vb_full = v[ks]
            dk_full = dk[ks]
            dv_full = dv[ks]
        else:
            kb_full = k[ks:ke]
            vb_full = v[ks:ke]
            dk_full = dk[ks:ke]
            dv_full = dv[ks:ke]
        with torch.enable_grad():
            qb_ref = qb_full[:lq].detach().requires_grad_(True)
            kb_ref = kb_full[:lk].detach().requires_grad_(True)
            vb_ref = vb_full[:lk].detach().requires_grad_(True)
            wl = _window_bound_or_none(window_size_left)
            wr = _window_bound_or_none(window_size_right)
            use_sdpa = (
                mask_mod is None
                and not _has_local_window((wl, wr))
                and (softcap is None or softcap == 0.0)
            )
            if use_sdpa:
                out_ref = torch.nn.functional.scaled_dot_product_attention(
                    qb_ref.permute(1, 0, 2).unsqueeze(0),
                    kb_ref.permute(1, 0, 2).unsqueeze(0),
                    vb_ref.permute(1, 0, 2).unsqueeze(0),
                    dropout_p=0.0,
                    is_causal=causal,
                    scale=softmax_scale,
                    enable_gqa=qb_ref.shape[1] != kb_ref.shape[1],
                ).squeeze(0).permute(1, 0, 2).contiguous()
                dq_b, dk_b, dv_b = torch.autograd.grad(out_ref, (qb_ref, kb_ref, vb_ref), dob_full[:lq], allow_unused=False)
                dq_full[:lq] = dq_b
                dk_full[:lk] = dk_b
                dv_full[:lk] = dv_b
                continue
            qt = qb_ref.transpose(0, 1).unsqueeze(0)
            kt = kb_ref.transpose(0, 1).unsqueeze(0)
            vt = vb_ref.transpose(0, 1).unsqueeze(0)
            if kt.shape[1] != hq:
                repeat_factor = hq // kt.shape[1]
                kt_compute = kt.repeat_interleave(repeat_factor, dim=1)
                vt_compute = vt.repeat_interleave(repeat_factor, dim=1)
            else:
                kt_compute = kt
                vt_compute = vt
            scores = torch.matmul(qt.float(), kt_compute.float().transpose(-1, -2)) * softmax_scale
            if softcap is not None and softcap > 0.0:
                scores = torch.tanh(scores / softcap) * softcap
            mask = torch.ones((1, hq, lq, lk), dtype=torch.bool, device=q.device)
            if causal:
                row = torch.arange(lq, device=q.device)[:, None]
                col = torch.arange(lk, device=q.device)[None, :]
                mask &= (col <= row + (lk - lq))[None, None, :, :]
            if _has_local_window((wl, wr)):
                wl = lk if wl is None else wl
                wr = lq if wr is None else wr
                row = torch.arange(lq, device=q.device)[:, None]
                col = torch.arange(lk, device=q.device)[None, :]
                center = row + (lk - lq)
                mask &= ((col >= center - wl) & (col <= center + wr))[None, None, :, :]
            if mask_mod is not None:
                local_aux = aux_tensors
                if aux_tensors is not None and len(aux_tensors) > 0 and aux_tensors[0].dim() >= 3 and aux_tensors[0].shape[0] > b:
                    local_aux = [aux_tensors[0][b : b + 1], *aux_tensors[1:]]
                mask &= _torch_mask_from_mod(mask_mod, 1, hq, lq, lk, q.device, local_aux, q_offset=qs, k_offset=ks)
            scores = scores.masked_fill(~mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            probs = torch.where(mask.any(dim=-1, keepdim=True), probs, torch.zeros_like(probs))
            out_ref = torch.matmul(probs.to(vt_compute.dtype), vt_compute).squeeze(0).transpose(0, 1)
            dq_b, dk_b, dv_b = torch.autograd.grad(out_ref, (qb_ref, kb_ref, vb_ref), dob_full[:lq], allow_unused=False)
        dq_full[:lq] = dq_b
        dk_full[:lk] = dk_b
        dv_full[:lk] = dv_b
    return dq, dk, dv


_sm120_deterministic_bwd_cache = {}


def _tensor_cache_id(t: Optional[torch.Tensor]):
    if t is None:
        return None
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)


def _tensor_value_cache_id(t: Optional[torch.Tensor]):
    if t is None:
        return None
    if t.numel() == 0:
        return (tuple(t.shape), tuple(t.stride()), t.dtype, ())
    flat = t.detach().reshape(-1)
    idxs = sorted({0, flat.numel() // 3, flat.numel() // 2, (2 * flat.numel()) // 3, flat.numel() - 1})
    samples = tuple(float(flat[i].item()) for i in idxs)
    return (tuple(t.shape), tuple(t.stride()), t.dtype, samples)


def _store_sm120_deterministic_bwd_result(cache_key, grads):
    if cache_key is None:
        return
    if len(_sm120_deterministic_bwd_cache) > 32:
        _sm120_deterministic_bwd_cache.clear()
    _sm120_deterministic_bwd_cache[cache_key] = tuple(g.detach().clone() for g in grads)


def _torch_block_sparse_runtime_tuple(tensors: BlockSparseTensorsTorch | None):
    if tensors is None:
        return None
    return (
        tensors.mask_block_cnt,
        tensors.mask_block_idx,
        tensors.full_block_cnt,
        tensors.full_block_idx,
        tensors.cu_total_m_blocks,
        tensors.cu_block_idx_offsets,
        tensors.dq_write_order,
        tensors.dq_write_order_full,
    )


def _sdpa_lse(qb: torch.Tensor, kb: torch.Tensor, softmax_scale: float, causal: bool) -> torch.Tensor:
    if qb.shape[1] != kb.shape[1]:
        kb = kb.repeat_interleave(qb.shape[1] // kb.shape[1], dim=1)
    scores = torch.einsum("lhd,shd->hls", qb.float(), kb.float()) * softmax_scale
    if causal:
        lq, lk = qb.shape[0], kb.shape[0]
        row = torch.arange(lq, device=qb.device)[:, None]
        col = torch.arange(lk, device=qb.device)[None, :]
        scores = scores.masked_fill(col > row + (lk - lq), float("-inf"))
    return torch.logsumexp(scores, dim=-1)


def _torch_varlen_attention_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    page_table: Optional[torch.Tensor],
    learnable_sink: Optional[torch.Tensor],
    softmax_scale: Optional[float],
    causal: bool,
    window_size: Tuple[Optional[int], Optional[int]],
    return_lse: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    hq, dv = q.shape[-2], v.shape[-1]
    out = torch.empty(q.shape[:-1] + (dv,), dtype=q.dtype, device=q.device)
    if q.requires_grad or k.requires_grad or v.requires_grad or return_lse:
        lse_shape = (q.shape[0], hq, q.shape[1]) if cu_seqlens_q is None else (hq, q.shape[0])
        lse = torch.empty(lse_shape, dtype=torch.float32, device=q.device)
    else:
        lse = None

    if cu_seqlens_q is None:
        q_offsets = [(b, b + 1, q.shape[1]) for b in range(q.shape[0])]
    else:
        q_lens = torch.diff(cu_seqlens_q).detach().cpu().tolist()
        q_starts = cu_seqlens_q.detach().cpu().tolist()
        q_offsets = [(q_starts[b], q_starts[b + 1], q_lens[b]) for b in range(len(q_lens))]

    if cu_seqlens_k is None:
        if k.dim() != 4:
            raise NotImplementedError("Fallback varlen attention requires cu_seqlens_k for packed K/V")
        k_offsets = [(b, b + 1, k.shape[1]) for b in range(len(q_offsets))]
    else:
        k_lens = torch.diff(cu_seqlens_k).detach().cpu().tolist()
        k_starts = cu_seqlens_k.detach().cpu().tolist()
        k_offsets = [(k_starts[b], k_starts[b + 1], k_lens[b]) for b in range(len(k_lens))]

    seq_q_lens = seqused_q.detach().cpu().tolist() if seqused_q is not None else [x[2] for x in q_offsets]
    seq_k_lens = seqused_k.detach().cpu().tolist() if seqused_k is not None else [x[2] for x in k_offsets]

    for b, ((qs, qe, _), (ks, ke, _)) in enumerate(zip(q_offsets, k_offsets)):
        lq, lk = int(seq_q_lens[b]), int(seq_k_lens[b])
        if cu_seqlens_q is None:
            qb_full = q[qs]
            out_full = out[qs]
            lse_slice = (qs, slice(None), slice(None))
        else:
            qb_full = q[qs:qe]
            out_full = out[qs:qe]
            lse_slice = (slice(None), slice(qs, qe))
        if page_table is not None:
            pages = page_table[b].detach().long()
            kb_full = k[pages].reshape(-1, k.shape[-2], k.shape[-1])
            vb_full = v[pages].reshape(-1, v.shape[-2], v.shape[-1])
        elif cu_seqlens_k is None:
            kb_full = k[ks]
            vb_full = v[ks]
        else:
            kb_full = k[ks:ke]
            vb_full = v[ks:ke]

        out_full.zero_()
        if lse is not None:
            lse[lse_slice] = float("-inf")
        if lq == 0 or lk == 0:
            continue
        qb = qb_full[:lq]
        kb = kb_full[:lk]
        vb = vb_full[:lk]
        wl = _window_bound_or_none(window_size[0])
        wr = _window_bound_or_none(window_size[1])
        if _has_local_window((wl, wr)) or learnable_sink is not None:
            kb_compute = kb.repeat_interleave(qb.shape[1] // kb.shape[1], dim=1) if qb.shape[1] != kb.shape[1] else kb
            vb_compute = vb.repeat_interleave(qb.shape[1] // vb.shape[1], dim=1) if qb.shape[1] != vb.shape[1] else vb
            scores = torch.einsum("lhd,shd->hls", qb.float(), kb_compute.float()) * softmax_scale
            row = torch.arange(lq, device=q.device)[:, None]
            col = torch.arange(lk, device=q.device)[None, :]
            if causal:
                scores = scores.masked_fill(col[None, :, :] > (row + (lk - lq))[None, :, :], float("-inf"))
            if _has_local_window((wl, wr)):
                wl_eff = lk if wl is None else wl
                wr_eff = lq if wr is None else wr
                center = row + (lk - lq)
                local_mask = (col >= center - wl_eff) & (col <= center + wr_eff)
                scores = scores.masked_fill(~local_mask[None, :, :], float("-inf"))
            if learnable_sink is None:
                probs = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
            else:
                sink = learnable_sink.float()[:, None, None]
                logits_max = torch.amax(scores.float(), dim=-1, keepdim=True)
                logits_or_sinks_max = torch.maximum(sink, logits_max)
                unnormalized_scores = torch.exp(scores.float() - logits_or_sinks_max)
                normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + torch.exp(sink - logits_or_sinks_max)
                probs = unnormalized_scores / normalizer
            sdpa_out = torch.einsum("hls,shd->lhd", probs.to(vb_compute.dtype), vb_compute)
        else:
            sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                qb.permute(1, 0, 2).unsqueeze(0),
                kb.permute(1, 0, 2).unsqueeze(0),
                vb.permute(1, 0, 2).unsqueeze(0),
                dropout_p=0.0,
                is_causal=causal,
                scale=softmax_scale,
                enable_gqa=qb.shape[1] != kb.shape[1],
            ).squeeze(0).permute(1, 0, 2).contiguous()
        out_full[:lq] = sdpa_out
        if lse is not None:
            if _has_local_window((wl, wr)) or learnable_sink is not None:
                if learnable_sink is None:
                    lse_value = torch.logsumexp(scores, dim=-1)
                else:
                    lse_value = torch.log(normalizer.squeeze(-1)) + logits_or_sinks_max.squeeze(-1)
                lse[lse_slice][..., :lq] = lse_value
            else:
                lse[lse_slice][..., :lq] = _sdpa_lse(qb, kb, softmax_scale, causal)
    return out, lse


def _use_sm120_varlen_torch_fallback(
    q: torch.Tensor,
    qv: Optional[torch.Tensor],
    page_table: Optional[torch.Tensor],
    gather_kv_indices: Optional[torch.Tensor],
    learnable_sink: Optional[torch.Tensor],
    score_mod: Optional[Callable],
    aux_tensors: Optional[list],
    softcap: float,
    window_size: Tuple[Optional[int], Optional[int]],
) -> bool:
    if is_fake_mode() or not torch.cuda.is_available() or _get_device_arch() // 10 != 12:
        return False
    has_local_window = _has_local_window(window_size)
    if learnable_sink is not None:
        return (
            qv is None
            and gather_kv_indices is None
            and score_mod is None
            and aux_tensors is None
            and softcap == 0.0
        )
    if not q.requires_grad:
        return False
    if q.shape[-1] not in (192, 256) and not (q.shape[-1] <= 128 and has_local_window):
        return False
    return (
        qv is None
        and page_table is None
        and gather_kv_indices is None
        and learnable_sink is None
        and score_mod is None
        and aux_tensors is None
        and softcap == 0.0
        and (window_size == (None, None) or has_local_window)
    )


def _use_sm120_varlen_custom_fallback_autograd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    deterministic: bool,
    causal: bool,
    window_size: Tuple[Optional[int], Optional[int]],
    learnable_sink: Optional[torch.Tensor],
    score_mod: Optional[Callable],
    aux_tensors: Optional[list],
    softcap: float,
) -> bool:
    if not deterministic or not (q.requires_grad or k.requires_grad or v.requires_grad):
        return False
    return (
        learnable_sink is None
        and score_mod is None
        and aux_tensors is None
        and softcap == 0.0
        and (causal or _has_local_window(window_size) or q.shape[-1] >= 192)
    )


