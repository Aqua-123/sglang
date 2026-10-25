"""Hide prefill TP all-reduces behind attention, keeping one MoE pass per layer.

On PCIe-only GPUs a prefill forward spends about as long in all-reduces as in compute.
Splitting whole layers into two token halves (two-batch overlap) reads every expert's weights
twice, which costs more than it hides. Here only the attention block runs per half: half A's
attention output is all-reduced on a side stream while half B's attention computes, and the
MoE output of half B is all-reduced while the next layer's attention computes half A. The MoE
and the dense MLP run once over all tokens.

The layer loop is written out (norms, attention, MLP, all-reduces) instead of going through
the layer boundaries, so it covers plain residual models only: no fusions, no deferred MoE
finalize, no captured layers.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch

from sglang.srt.batch_overlap.two_batch_overlap import (
    TboForwardBatchPreparer,
    _is_two_chunk_split_enabled,
    compute_split_seq_index,
)
from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    get_attn_tp_group,
    get_tp_group,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

_comm_stream: Optional[torch.cuda.Stream] = None


def can_overlap_attn(forward_batch: ForwardBatch) -> bool:
    min_tokens = envs.SGLANG_ATTN_OVERLAP_MIN_TOKENS.get()
    if min_tokens <= 0 or forward_batch.forward_mode != ForwardMode.EXTEND:
        return False
    if forward_batch.extend_num_tokens is None or forward_batch.extend_num_tokens < min_tokens:
        return False
    parallel = get_parallel()
    if parallel.attn_tp_size <= 1 or parallel.attn_dp_enabled:
        return False
    if forward_batch.tbo_children is not None or forward_batch.can_run_tbo:
        return False
    backend = get_attn_backend()
    metadata = getattr(backend, "forward_metadata", None)
    # Image masks index the parent's query ranges, so multimodal prefills keep the plain path.
    return (
        type(backend).__name__ == "FlashAttentionBackend"
        and metadata is not None
        and getattr(metadata, "image_token_ranges", None) is None
    )


def _split_batch(forward_batch: ForwardBatch) -> Optional[Tuple[List[ForwardBatch], int]]:
    """Two child batches cut at a token index (inside a sequence when needed)."""
    extend_lens = forward_batch.extend_seq_lens_cpu
    split_seq_index = compute_split_seq_index(
        forward_mode=forward_batch.forward_mode,
        num_tokens=forward_batch.input_ids.shape[0],
        extend_lens=extend_lens,
        token_num_per_seq=None,
    )
    two_chunk = _is_two_chunk_split_enabled(extend_lens)
    # filter_batch refuses to copy a live residual stream.
    residual_stream, forward_batch.residual_stream = forward_batch.residual_stream, None
    forward_batch.tbo_split_seq_index = split_seq_index
    try:
        split = TboForwardBatchPreparer._compute_split_token_index(forward_batch)
        num_tokens = forward_batch.input_ids.shape[0]
        if not 0 < split < num_tokens:
            return None
        bounds = [(0, split), (split, num_tokens)]
        seq_bounds = [
            (0, split_seq_index + 1 if two_chunk else split_seq_index),
            (split_seq_index, forward_batch.batch_size),
        ]
        children = [
            TboForwardBatchPreparer.filter_batch(
                forward_batch,
                start_token_index=start,
                end_token_index=end,
                start_seq_index=seq_start,
                end_seq_index=seq_end,
                out_num_token_non_padded=torch.tensor(
                    end - start, dtype=torch.int32, device=forward_batch.input_ids.device
                ),
                out_num_token_non_padded_cpu=end - start,
            )
            for (start, end), (seq_start, seq_end) in zip(bounds, seq_bounds)
        ]
        if two_chunk:
            TboForwardBatchPreparer.derive_fields_related_to_seq_len_for_two_chunk(
                forward_batch,
                child_a=children[0],
                child_b=children[1],
                tbo_split_seq_index=split_seq_index,
            )
    finally:
        forward_batch.tbo_split_seq_index = None
        forward_batch.residual_stream = residual_stream
    for child, (start, end) in zip(children, bounds):
        child.tbo_padded_len = end - start
    return children, split


class _Reduced:
    """An all-reduce issued on the comm stream; ``get`` joins it on the compute stream."""

    def __init__(self, group: GroupCoordinator, partial: torch.Tensor, comm: torch.cuda.Stream):
        compute = torch.cuda.current_stream()
        comm.wait_stream(compute)
        with torch.cuda.stream(comm):
            self.value = group.all_reduce(partial)
            self.done = torch.cuda.Event()
            self.done.record(comm)
        partial.record_stream(comm)
        self.value.record_stream(compute)

    def get(self) -> torch.Tensor:
        torch.cuda.current_stream().wait_event(self.done)
        return self.value


def _add_norm(norm, reduced: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """residual += reduced in place; return norm(residual)."""
    out, summed = norm(reduced, residual)
    if summed.data_ptr() != residual.data_ptr():
        residual.copy_(summed)
    return out


def forward_layers_attn_overlap(
    layers, positions: torch.Tensor, forward_batch: ForwardBatch, hidden_states: torch.Tensor
) -> Optional[torch.Tensor]:
    """Run ``layers`` from the embeddings; return the complete residual stream (pre final
    norm), or None when the batch cannot be split."""
    try:
        split = _split_batch(forward_batch)
    except Exception:
        # filter_batch rejects ForwardBatch fields it does not know how to split.
        logger.warning("attention overlap: cannot split batch", exc_info=True)
        split = None
    if split is None:
        return None
    children, cut = split
    bounds = ((0, cut), (cut, hidden_states.shape[0]))

    backend = get_attn_backend()
    parent_metadata = backend.forward_metadata
    child_metadata = []
    for child in children:
        backend.init_forward_metadata(child)
        child_metadata.append(backend.forward_metadata)
    backend.forward_metadata = parent_metadata

    global _comm_stream
    if _comm_stream is None:
        _comm_stream = torch.cuda.Stream(priority=-1)
    comm = _comm_stream
    attn_group, ffn_group = get_attn_tp_group(), get_tp_group()

    residual = hidden_states.clone()
    pending: Optional[List[_Reduced]] = None
    try:
        for layer in layers:
            attn_out = []
            for (lo, hi), child, metadata in zip(bounds, children, child_metadata):
                x = residual[lo:hi]
                if pending is None:
                    h = layer.input_layernorm(x)
                else:
                    h = _add_norm(layer.input_layernorm, pending.pop(0).get(), x)
                backend.forward_metadata = metadata
                partial = layer.self_attn(
                    positions=positions[..., lo:hi], hidden_states=h, forward_batch=child
                )
                attn_out.append(_Reduced(attn_group, partial, comm))
            backend.forward_metadata = parent_metadata
            h = torch.cat(
                [
                    _add_norm(layer.post_attention_layernorm, r.get(), residual[lo:hi])
                    for r, (lo, hi) in zip(attn_out, bounds)
                ]
            )
            partial = layer.mlp(h, forward_batch)
            if not isinstance(partial, torch.Tensor):
                raise RuntimeError("attention overlap needs a plain MLP output")
            pending = [_Reduced(ffn_group, partial[lo:hi], comm) for lo, hi in bounds]
    finally:
        backend.forward_metadata = parent_metadata
    for r, (lo, hi) in zip(pending, bounds):
        residual[lo:hi] += r.get()
    return residual
