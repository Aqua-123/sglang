"""Overlap tensor-parallel all-reduce with compute during prefill.

Without NVLink the per-layer TP all-reduces of a prefill forward can cost as much as its
compute. This splits a plain extend batch into two token halves (A and B, cut inside a
sequence when needed) and runs the decoder layers for each half on its own CUDA stream, with B
one layer behind A: B's layer L waits for A's layer L, so B's attention sees the KV that A wrote
for the same request, while A's next layer, including its all-reduces, runs alongside B.

Unlike two-batch overlap this needs no a2a backend or DP attention: the layers run unchanged,
each half with its own residual stream and attention metadata.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import List, Optional

import torch

from sglang.srt.batch_overlap.two_batch_overlap import (
    TboForwardBatchPreparer,
    _is_two_chunk_split_enabled,
    _model_forward_tbo_merge_outputs,
    _model_forward_tbo_split_inputs,
    compute_split_seq_index,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.environ import envs
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

_streams: Optional[List[torch.cuda.Stream]] = None
_comm_stream: Optional[torch.cuda.Stream] = None


@contextmanager
def _all_reduce_on(comm: torch.cuda.Stream):
    """Issue every all-reduce on ``comm``, a high-priority stream, in launch order.

    The MoE grouped GEMM is persistent and holds every SM until it finishes, so an all-reduce
    that becomes ready while the other half's GEMM is queued must win the block scheduler.
    """
    original = GroupCoordinator.all_reduce

    def all_reduce(self, input_):
        compute = torch.cuda.current_stream()
        comm.wait_stream(compute)
        with torch.cuda.stream(comm):
            output = original(self, input_)
        compute.wait_stream(comm)
        input_.record_stream(comm)
        output.record_stream(compute)
        return output

    GroupCoordinator.all_reduce = all_reduce
    try:
        yield
    finally:
        GroupCoordinator.all_reduce = original


def can_overlap_tp_prefill(forward_batch: ForwardBatch) -> bool:
    min_tokens = envs.SGLANG_TP_PREFILL_OVERLAP_MIN_TOKENS.get()
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
    # Per-half metadata is rebuilt with init_forward_metadata; image masks index the parent's
    # query ranges, so multimodal prefills keep the plain path.
    return (
        type(backend).__name__ == "FlashAttentionBackend"
        and metadata is not None
        and getattr(metadata, "image_token_ranges", None) is None
    )


def _split_batch(forward_batch: ForwardBatch) -> Optional[List[ForwardBatch]]:
    extend_lens = forward_batch.extend_seq_lens_cpu
    split_seq_index = compute_split_seq_index(
        forward_mode=forward_batch.forward_mode,
        num_tokens=forward_batch.input_ids.shape[0],
        extend_lens=extend_lens,
        token_num_per_seq=None,
    )
    two_chunk = _is_two_chunk_split_enabled(extend_lens)
    # The halves get their own residual streams; filter_batch refuses to copy a live one.
    residual_stream, forward_batch.residual_stream = forward_batch.residual_stream, None
    forward_batch.tbo_split_seq_index = split_seq_index
    try:
        split_token_index = TboForwardBatchPreparer._compute_split_token_index(forward_batch)
        num_tokens = forward_batch.input_ids.shape[0]
        if not 0 < split_token_index < num_tokens:
            return None
        bounds = [(0, split_token_index), (split_token_index, num_tokens)]
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
        # Rows are TP-replicated here; the halves need no padding.
        child.tbo_padded_len = end - start
    return children


def forward_layers_tp_prefill_overlap(
    layers, positions: torch.Tensor, forward_batch: ForwardBatch, hidden_states
):
    """Run ``layers`` over ``forward_batch`` with the two halves on separate streams.

    Falls back to running the layers in order when the batch cannot be split.
    """
    try:
        children = _split_batch(forward_batch)
    except Exception:
        # filter_batch rejects ForwardBatch fields it does not know how to split.
        logger.warning("TP prefill overlap: cannot split batch, running unsplit", exc_info=True)
        children = None
    if children is None:
        for layer in layers:
            hidden_states = layer(positions, hidden_states, forward_batch)
        return hidden_states

    backend = get_attn_backend()
    parent_metadata = backend.forward_metadata
    child_metadata = []
    for child in children:
        backend.init_forward_metadata(child)
        child_metadata.append(backend.forward_metadata)
    backend.forward_metadata = parent_metadata

    stream = residual_batch.stream_of(forward_batch)
    pending = stream.pending
    hidden_states, residual = stream.export(hidden_states)
    forward_batch.tbo_children = children
    try:
        parts = _model_forward_tbo_split_inputs(
            hidden_states=hidden_states,
            residual=residual,
            positions=positions,
            forward_batch=forward_batch,
            zero_allocator=None,
            layer_input_rows=layers[0].attn_boundary.incoming_residual_rows,
        )
    finally:
        forward_batch.tbo_children = None
    for part in parts:
        part["hidden_states"], part["forward_batch"].residual_stream = (
            ResidualStream.from_handoff(
                part["hidden_states"],
                part.pop("residual"),
                pending.update if pending is not None else None,
            )
        )
    original_len = hidden_states.shape[0]
    forward_batch.residual_stream = None
    del pending, stream, hidden_states, residual

    global _streams, _comm_stream
    if _streams is None:
        _streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        _comm_stream = torch.cuda.Stream(priority=-1)
    main = torch.cuda.current_stream()
    for side in _streams:
        side.wait_stream(main)
    try:
        with _all_reduce_on(_comm_stream):
            for layer in layers:
                a_done = torch.cuda.Event()
                for index, (part, metadata, side) in enumerate(
                    zip(parts, child_metadata, _streams)
                ):
                    with torch.cuda.stream(side):
                        if index == 1:
                            side.wait_event(a_done)
                        backend.forward_metadata = metadata
                        part["hidden_states"] = layer(
                            part["positions"], part["hidden_states"], part["forward_batch"]
                        )
                        if index == 0:
                            a_done.record(side)
    finally:
        backend.forward_metadata = parent_metadata
        for side in _streams:
            main.wait_stream(side)

    for part in parts:
        part["residual"] = residual_batch.stream_of(part["forward_batch"])
    hidden_states, forward_batch.residual_stream = _model_forward_tbo_merge_outputs(
        *parts, original_len
    )
    for part in parts:
        part["forward_batch"].residual_stream = None
    return hidden_states
