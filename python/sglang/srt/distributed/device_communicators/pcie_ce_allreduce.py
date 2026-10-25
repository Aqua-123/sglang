# SPDX-License-Identifier: Apache-2.0
"""Copy-engine all-reduce for 4 GPUs wired as two PCIe-switch pairs.

Target host: GPUs (0,1) and (2,3) each share a PCIe switch, so P2P works inside
a pair and cross-pair traffic goes through host memory. The two GPUs of a pair
also share one ~56 GB/s uplink to the CPU.

Algorithm, for message halves H0 and H1, where each pair member owns one half:

1. Reduce-scatter inside the pair. Each GPU pushes the partner-owned half of its
   input into the partner's ``recv_rs`` over P2P, then adds its own half to it.
2. Cross-pair exchange. Each GPU copies its partial sum to its slot in shared
   pinned host memory (D2H), the cross peer copies it in (H2D) and adds.
3. All-gather inside the pair. Each GPU pushes its finished half to the
   partner's ``recv_ag``, and the partner copies it into the output.

Every transfer is a copy-engine memcpy, chunked so the stages overlap. Pushes
only: on this fabric a P2P read collapses whenever traffic flows the other way.
The only SM work is two small-grid add kernels per chunk. They still need SM
slots, so a kernel that holds every SM stalls this all-reduce until it ends,
much as it stalls NCCL.

Cross-process ordering uses GPU-side stream memory ops on counters in the
shared host region. Each counter has exactly one writer and only grows, so
nothing is ever reset; ``cuStreamWaitValue32`` GEQ compares the signed
difference, so the 32-bit wrap is harmless.

Every rank computes each element as ``(a + b) + (c + d)`` with the same
operand pairing, and fp addition commutes, so all ranks hold bitwise-equal
results.
"""

import ctypes
import logging
import mmap
import os
import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from cuda.bindings import driver as drv
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

_WORLD_SIZE = 4
_ALIGN = 4096
_FLAG_STRIDE = 64  # one cache line per counter
_RS, _X, _AG, _DONE = range(4)
_STAGES = ("out", "k0", "k1", "h2d", "k2", "loc")
# Streams that launch kernels: k0 quantizes, loc dequantizes in quantized mode.
_KERNEL_STAGES = ("k0", "k1", "k2", "loc")
# Bits per code; the key order gives the kernels' MODE.
_CODE_BITS = {"int8": 8, "fp8": 8, "fp4": 4, "fp6": 6}
_MAX_CHUNKS = 16
# Outbound schedule offsets, tuned on 8x RTX PRO 6000 (lead=lag=2 beat 1, 3, 8).
_LEAD = 2
_LAG = 2
_MIN_BYTES_FLOOR = 1 << 20  # below this a half is too small to chunk
_U32 = 0xFFFFFFFF
_WAIT_GEQ = drv.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ
_WRITE_FENCED = drv.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT


def _ok(result):
    err, *vals = result
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {err}")
    return vals[0] if len(vals) == 1 else vals


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


@dataclass(frozen=True)
class _WireFormat:
    """Block-scaled payload: one scale of ``scale_dtype`` per ``block`` elements."""

    codes: str  # a key of _CODE_BITS
    block: int
    scale_dtype: torch.dtype

    @property
    def mode(self) -> int:
        return list(_CODE_BITS).index(self.codes)

    def code_bytes(self, n: int) -> int:
        return n * _CODE_BITS[self.codes] // 8

    def span(self, a: int, b: int) -> Tuple[int, int]:
        """Byte offset and size on the wire of elements [a, b) of a half."""
        sb = self.scale_dtype.itemsize
        padded = _round_up(b - a, self.block)
        return (
            self.code_bytes(a) + a // self.block * sb,
            self.code_bytes(padded) + padded // self.block * sb,
        )

    def views(self, buf: torch.Tensor, off: int, n: int):
        """Code and scale views of the n-element encoded chunk at byte ``off``."""
        padded = _round_up(n, self.block)
        codes = self.code_bytes(padded)
        scales = buf[
            off + codes : off + codes + padded // self.block * self.scale_dtype.itemsize
        ]
        return buf[off : off + codes], scales.view(self.scale_dtype)


# A bf16 scale per 16 elements costs the same bytes as an fp32 scale per 32,
# and an outlier then spoils 15 neighbours instead of 31. Measured on 4 GPUs
# with synthetic heavy-tailed partials (Student-t(3), [2048, 4096]), the
# all-reduce's relative error is 0.165 for fp4, 0.140 for fp4/16, 0.057 for
# fp6/16 and 0.032 for fp8; on the channels beside 100-1000x outlier channels
# it is 0.31, 0.23, 0.066 and 0.033.
_WIRE_FORMATS = {
    "int8": _WireFormat("int8", 32, torch.float32),
    "fp8": _WireFormat("fp8", 32, torch.float32),
    "fp4": _WireFormat("fp4", 32, torch.float32),
    "fp8/16": _WireFormat("fp8", 16, torch.bfloat16),
    "fp4/16": _WireFormat("fp4", 16, torch.bfloat16),
    "fp6/16": _WireFormat("fp6", 16, torch.bfloat16),
}


@triton.jit
def _add_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    start = tl.program_id(0) * BLOCK
    step = tl.num_programs(0) * BLOCK
    for off in range(start, n, step):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < n
        a = tl.load(a_ptr + offs, mask=mask).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=mask).to(tl.float32)
        tl.store(out_ptr + offs, (a + b).to(out_ptr.dtype.element_ty), mask=mask)


# Quantized wire format. A chunk of E elements travels as one span:
# [round_up(E, QB) codes | one scale per QB-element block], the scale fp32 or
# bf16 per the scale pointer. int8 and fp8 codes take a byte each; fp4 (e2m1)
# codes pack two to a byte, the even element in the low nibble; fp6 (e3m2)
# codes pack four to three bytes, little-endian, the first element lowest.
_FP8 = tl.constexpr(1)
_FP4 = tl.constexpr(2)
_FP6 = tl.constexpr(3)


@triton.jit
def _tile(t, n, QB: tl.constexpr, ROWS: tl.constexpr):
    rows = t * ROWS + tl.arange(0, ROWS)
    offs = rows[:, None] * QB + tl.arange(0, QB)[None, :]
    return rows, offs, offs < n, rows < tl.cdiv(n, QB)


@triton.jit
def _packed(rows, n, QB: tl.constexpr):
    """Byte offsets and mask of a tile's packed fp4 codes."""
    offs = rows[:, None] * (QB // 2) + tl.arange(0, QB // 2)[None, :]
    return offs, offs * 2 < n


@triton.jit
def _e2m1_value(c):
    """Value of a 4-bit e2m1 code: +-{0, .5, 1, 1.5, 2, 3, 4, 6}, sign in bit 3."""
    e = ((c >> 1) & 3).to(tl.float32)
    m = (c & 1).to(tl.float32)
    v = tl.where(e == 0, m * 0.5, tl.exp2(e - 1.0) * (1.0 + m * 0.5))
    return tl.where((c & 8) != 0, -v, v)


@triton.jit
def _e2m1_code(y):
    """Nearest e2m1 code of y in [-6, 6]."""
    a = tl.abs(y)
    c = (
        (a >= 0.25).to(tl.int32)
        + (a >= 0.75).to(tl.int32)
        + (a >= 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32)
        + (a >= 2.5).to(tl.int32)
        + (a >= 3.5).to(tl.int32)
        + (a >= 5.0).to(tl.int32)
    )
    return c | tl.where(y < 0, 8, 0)


@triton.jit
def _pow2(e):
    """2.0 ** e for integer e in the normal fp32 range, exactly."""
    return ((e + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _e3m2_value(c):
    """Value of a 6-bit e3m2 code: sign in bit 5, then 3 exponent and 2
    mantissa bits, exponent bias 3, no inf or nan: up to 28, steps of 1/16
    below 1/4."""
    e = (c >> 2) & 7
    m = c & 3
    v = tl.where(e == 0, m * 0.0625, (4 + m) * _pow2(e - 5))
    return tl.where((c & 32) != 0, -v, v)


@triton.jit
def _e3m2_code(y):
    """Nearest e3m2 code of y in [-28, 28], ties away from zero."""
    a = tl.abs(y)
    # floor(log2(a)), at least that of the smallest normal; the step is a
    # quarter of that power of two.
    e = tl.maximum(((a.to(tl.int32, bitcast=True) >> 23) & 255) - 127, -2)
    k = tl.floor(a * _pow2(2 - e) + 0.5).to(tl.int32)  # 0..4 if e == -2, else 4..8
    c = tl.minimum(4 * e + 8 + k, 31)
    return c | tl.where(y < 0, 32, 0)


@triton.jit
def _fp6_groups(rows, n, QB: tl.constexpr):
    """Byte offset and mask of each 4-code group of a tile's packed fp6 codes."""
    g = tl.arange(0, QB // 4)[None, :]
    return rows[:, None] * (QB // 4 * 3) + g * 3, rows[:, None] * QB + g * 4 < n


@triton.jit
def _decode(
    q_ptr,
    s_ptr,
    rows,
    offs,
    mask,
    row_mask,
    n,
    QB: tl.constexpr,
    ROWS: tl.constexpr,
    MODE: tl.constexpr,
):
    if MODE == _FP4:
        boffs, bmask = _packed(rows, n, QB)
        b = tl.load(q_ptr + boffs, mask=bmask, other=0).to(tl.int32)
        v = tl.reshape(tl.join(_e2m1_value(b & 15), _e2m1_value(b >> 4)), (ROWS, QB))
    elif MODE == _FP6:
        goffs, gmask = _fp6_groups(rows, n, QB)
        w = tl.load(q_ptr + goffs, mask=gmask, other=0).to(tl.int32)
        w |= tl.load(q_ptr + goffs + 1, mask=gmask, other=0).to(tl.int32) << 8
        w |= tl.load(q_ptr + goffs + 2, mask=gmask, other=0).to(tl.int32) << 16
        even = tl.join(w & 63, (w >> 12) & 63)
        odd = tl.join((w >> 6) & 63, w >> 18)
        v = _e3m2_value(tl.reshape(tl.join(even, odd), (ROWS, QB)))
    else:
        q = tl.load(q_ptr + offs, mask=mask, other=0)
        if MODE == _FP8:
            v = q.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        else:
            v = q.to(tl.int8, bitcast=True).to(tl.float32)
    s = tl.load(s_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    return v * s[:, None]


@triton.jit
def _encode(
    v,
    q_ptr,
    s_ptr,
    rows,
    offs,
    mask,
    row_mask,
    n,
    QB: tl.constexpr,
    ROWS: tl.constexpr,
    MODE: tl.constexpr,
):
    """Store v block-scaled; return the values a receiver will decode."""
    if MODE == _FP4:
        qmax = 6.0
    elif MODE == _FP6:
        qmax = 28.0
    elif MODE == _FP8:
        qmax = 448.0
    else:
        qmax = 127.0
    amax = tl.max(tl.abs(tl.where(mask, v, 0.0)), axis=1)
    if s_ptr.dtype.element_ty == tl.float32:
        scale = amax / qmax
        y = v * tl.where(amax > 0, qmax / amax, 0.0)[:, None]
    else:
        # A bf16 scale rounds to nearest, so y may pass qmax by 2^-9 relative;
        # every code kind rounds that back to qmax.
        scale = (amax / qmax).to(s_ptr.dtype.element_ty).to(tl.float32)
        y = v * tl.where(scale > 0, 1.0 / scale, 0.0)[:, None]
    if MODE == _FP4:
        c = _e2m1_code(y)
        lo, hi = tl.split(tl.reshape(c, (ROWS, QB // 2, 2)))
        boffs, bmask = _packed(rows, n, QB)
        tl.store(q_ptr + boffs, (lo | (hi << 4)).to(tl.uint8), mask=bmask)
        qf = _e2m1_value(c)
    elif MODE == _FP6:
        c = _e3m2_code(y)
        # Split each group's codes j = 2i + k by k, then by i.
        even, odd = tl.split(tl.reshape(c, (ROWS, QB // 4, 2, 2)))
        c0, c2 = tl.split(even)
        c1, c3 = tl.split(odd)
        w = c0 | (c1 << 6) | (c2 << 12) | (c3 << 18)
        goffs, gmask = _fp6_groups(rows, n, QB)
        tl.store(q_ptr + goffs, (w & 255).to(tl.uint8), mask=gmask)
        tl.store(q_ptr + goffs + 1, ((w >> 8) & 255).to(tl.uint8), mask=gmask)
        tl.store(q_ptr + goffs + 2, (w >> 16).to(tl.uint8), mask=gmask)
        qf = _e3m2_value(c)
    else:
        if MODE == _FP8:
            qf = y.to(tl.float8e4nv).to(tl.float32)
            q = y.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        else:
            qf = tl.where(y >= 0, y + 0.5, y - 0.5).to(tl.int32).to(tl.float32)
            q = qf.to(tl.int8).to(tl.uint8, bitcast=True)
        tl.store(q_ptr + offs, q, mask=mask)
    tl.store(s_ptr + rows, scale.to(s_ptr.dtype.element_ty), mask=row_mask)
    return qf * scale[:, None]


@triton.jit
def _quant_kernel(
    x_ptr, q_ptr, s_ptr, n, QB: tl.constexpr, ROWS: tl.constexpr, MODE: tl.constexpr
):
    for t in range(tl.program_id(0), tl.cdiv(n, QB * ROWS), tl.num_programs(0)):
        rows, offs, mask, row_mask = _tile(t, n, QB, ROWS)
        v = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        _encode(v, q_ptr, s_ptr, rows, offs, mask, row_mask, n, QB, ROWS, MODE)


@triton.jit
def _decode_add_encode_kernel(
    aq,
    as_,
    x_ptr,
    oq,
    os_,
    n,
    QB: tl.constexpr,
    ROWS: tl.constexpr,
    MODE_IN: tl.constexpr,
    MODE_OUT: tl.constexpr,
):
    """o = encode(decode(a) + x); o may alias a if both use one format."""
    for t in range(tl.program_id(0), tl.cdiv(n, QB * ROWS), tl.num_programs(0)):
        rows, offs, mask, row_mask = _tile(t, n, QB, ROWS)
        v = _decode(aq, as_, rows, offs, mask, row_mask, n, QB, ROWS, MODE_IN)
        v += tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        _encode(v, oq, os_, rows, offs, mask, row_mask, n, QB, ROWS, MODE_OUT)


@triton.jit
def _decode2_add_encode_kernel(
    aq,
    as_,
    bq,
    bs,
    oq,
    os_,
    out_ptr,
    n,
    QB: tl.constexpr,
    ROWS: tl.constexpr,
    MODE_IN: tl.constexpr,
    MODE_OUT: tl.constexpr,
):
    """o = encode(decode(a) + decode(b)); out = what o decodes to. o may alias
    an input if all use one format. The compiler fuses one decode into an FMA,
    so the sum is only bitwise reproducible if every rank passes the operands
    in the same order."""
    for t in range(tl.program_id(0), tl.cdiv(n, QB * ROWS), tl.num_programs(0)):
        rows, offs, mask, row_mask = _tile(t, n, QB, ROWS)
        v = _decode(aq, as_, rows, offs, mask, row_mask, n, QB, ROWS, MODE_IN)
        v += _decode(bq, bs, rows, offs, mask, row_mask, n, QB, ROWS, MODE_IN)
        v = _encode(v, oq, os_, rows, offs, mask, row_mask, n, QB, ROWS, MODE_OUT)
        tl.store(out_ptr + offs, v.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _decode_kernel(
    q_ptr, s_ptr, out_ptr, n, QB: tl.constexpr, ROWS: tl.constexpr, MODE: tl.constexpr
):
    for t in range(tl.program_id(0), tl.cdiv(n, QB * ROWS), tl.num_programs(0)):
        rows, offs, mask, row_mask = _tile(t, n, QB, ROWS)
        v = _decode(q_ptr, s_ptr, rows, offs, mask, row_mask, n, QB, ROWS, MODE)
        tl.store(out_ptr + offs, v.to(out_ptr.dtype.element_ty), mask=mask)


@dataclass(frozen=True)
class _Topology:
    partner: int  # rank in group sharing my PCIe switch
    cross: int  # rank in the other pair that owns the same half
    half: int  # 0 or 1: the half of the message I reduce


def _pair_topology(rank: int, p2p: List[List[bool]]) -> Optional[_Topology]:
    """Topology from the group's P2P matrix, or None if it is not two pairs."""
    peers = [
        [j for j in range(_WORLD_SIZE) if j != i and p2p[i][j]]
        for i in range(_WORLD_SIZE)
    ]
    if any(len(p) != 1 for p in peers):
        return None
    partner = peers[rank][0]
    if peers[partner] != [rank]:
        return None
    half = int(rank > partner)
    others = sorted(set(range(_WORLD_SIZE)) - {rank, partner})
    return _Topology(partner=partner, cross=others[half], half=half)


def _cpulist(text: str) -> set:
    cpus = set()
    for part in text.strip().split(","):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def _first_touch_on_gpu_node(addr: int, size: int, bus_id: str) -> None:
    """Fault the pages in from a CPU on the GPU's NUMA node, so they live there.

    Measured: with the region on the far socket the 67 MB all-reduce slows
    from 2.58 to 2.96 ms. Best effort; placement is left alone if unknown.
    """
    domain, rest = bus_id.split(":", 1)
    try:
        dev = f"/sys/bus/pci/devices/{int(domain, 16):04x}:{rest.lower()}"
        with open(f"{dev}/numa_node") as f:
            node = int(f.read())
        with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
            node_cpus = _cpulist(f.read())
    except (OSError, ValueError):
        return
    old = os.sched_getaffinity(0)
    if node < 0 or not node_cpus & old:
        return
    os.sched_setaffinity(0, node_cpus & old)
    try:
        ctypes.memset(addr, 0, size)
    finally:
        os.sched_setaffinity(0, old)


class _RawCuda:
    """Exposes a raw device allocation to ``torch.as_tensor`` without copying."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
            "strides": None,
        }


def _chunks(lo: int, hi: int, n_chunks: int, align: int) -> List[Tuple[int, int]]:
    """Split [lo, hi) into exactly n_chunks aligned ranges, so both halves agree."""
    bounds = [lo + (hi - lo) * i // n_chunks // align * align for i in range(n_chunks)]
    return list(zip(bounds, bounds[1:] + [hi]))


def _outbound_schedule(n_chunks: int, lead: int, lag: int) -> List[Tuple[str, int]]:
    """Order of one GPU's outbound copies (RS push, D2H, AG push).

    They all run on one copy engine, so they go on one stream in this order
    rather than leaving the interleaving to engine arbitration. RS pushes run
    ``lead`` chunks ahead of the D2H that needs them; an AG push trails the D2H
    of the same chunk by ``lag`` to absorb cross-pair skew.
    """
    sched = [("rs", c) for c in range(min(lead, n_chunks))]
    for c in range(n_chunks):
        if c + lead < n_chunks:
            sched.append(("rs", c + lead))
        sched.append(("d2h", c))
        if c >= lag:
            sched.append(("ag", c - lag))
    sched += [("ag", c) for c in range(max(0, n_chunks - lag), n_chunks)]
    return sched


class PcieCeAllReduce:
    """Sum all-reduce over 4 ranks driven by copy engines.

    ``group`` is a CPU (gloo) group over the 4 ranks; it carries the one-time
    handshake only. Messages larger than ``capacity_bytes`` run in segments.
    """

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        min_bytes: int = 16 << 20,
        capacity_bytes: int = 128 << 20,
        chunk_bytes: int = 4 << 20,
        num_ctas: int = 16,
        quant: Optional[str] = None,
    ):
        """``quant`` sends every payload block-scaled in a format of
        ``_WIRE_FORMATS``: "int8", "fp8" and "fp4" (e2m1) with an fp32 scale
        per 32 elements, or "fp8/16", "fp6/16" (e3m2) and "fp4/16" with a bf16
        scale per 16. int8 and fp8 halve link bytes, fp6 cuts them to under
        half and fp4 to under a third. Each hop re-quantizes, so the result
        carries three rounding steps; ranks stay bitwise equal. On activations
        with outlier channels the float codes are the more accurate: an outlier
        sets its block's int8 step for every other element in the block, while
        a float code keeps a per-element exponent.

        "rs+cross+ag" picks a format per stage: the reduce-scatter and
        all-gather inside the pairs, and the partial sums that cross the shared
        uplink. "pair+cross" is short for "pair+cross+pair". Only the cross
        format crosses the uplink, but every stage loads each GPU's own link,
        so a wider pair format still costs time on large messages."""
        fmts = [_WIRE_FORMATS[f] for f in quant.split("+")] if quant else [None]
        if len(fmts) < 3:
            fmts = [fmts[0], fmts[-1], fmts[0]]
        self.rs_fmt, self.x_fmt, self.ag_fmt = fmts
        if quant:
            assert len({f.block for f in fmts}) == 1, quant
            assert _ALIGN // 4 % fmts[0].block == 0, quant
        self.disabled = True
        self.quant = quant
        self.group = group
        self.device = torch.device(device)
        self.min_bytes = min_bytes
        self.chunk_bytes = chunk_bytes
        self.num_ctas = num_ctas
        self.rank = dist.get_rank(group)
        world = dist.get_world_size(group)
        self._dev_ptr = 0
        self._peer_ptr = 0
        self._host_addr = 0
        self._mmap = None
        if world != _WORLD_SIZE:
            logger.info("PCIe CE all-reduce needs world size 4, got %d.", world)
            return

        with torch.cuda.device(self.device):
            torch.cuda.current_stream().synchronize()
            topo = self._discover_topology()
            if topo is None:
                logger.warning(
                    "PCIe CE all-reduce disabled: the group is not two P2P pairs "
                    "without P2P between them."
                )
                return
            self.topo = topo
            # A half is at most half the capacity plus the segment and half
            # rounding (under 16 KiB even for fp32).
            self.half_cap = _round_up(capacity_bytes // 2 + (64 << 10), _ALIGN)
            # Re-encoding into another format cannot run in place, so a cross
            # format that differs gets its own region of recv_rs, recv_x and the
            # host slots, past the rs and ag payloads. recv_ag doubles as the
            # reduce-scatter staging area; that too moves past the ag payload
            # if the two formats differ.
            self._x_base = self._stage_base = 0
            if quant:
                # Wire bytes of a half of 16-bit elements in each format.
                rs, x, ag = (f.span(0, self.half_cap // 2)[1] for f in fmts)
                if len(set(fmts)) > 1:
                    self._x_base = _round_up(max(rs, ag), _ALIGN)
                if fmts[0] != fmts[2]:
                    self._stage_base = _round_up(ag, _ALIGN)
                assert self._x_base + x <= self.half_cap, quant
                assert self._stage_base + rs <= self.half_cap, quant
            self.capacity_bytes = capacity_bytes
            self._setup_device_buffers()
            self._setup_host_region()
            # Add kernels go on top-priority streams (torch clamps -100 to the
            # highest): with the SMs saturated, their pending blocks are
            # scheduled ahead of the compute stream's.
            self._streams = [
                torch.cuda.Stream(
                    self.device, priority=-100 if st in _KERNEL_STAGES else 0
                )
                for st in _STAGES
            ]
            self._cu = [drv.CUstream(st.cuda_stream) for st in self._streams]
            no_timing = drv.CUevent_flags.CU_EVENT_DISABLE_TIMING
            # encoded / summed / landed / reduced per chunk, then the join.
            self._events = [
                _ok(drv.cuEventCreate(no_timing))
                for _ in range(4 * _MAX_CHUNKS + len(_STAGES) + 1)
            ]
            self._calls = 0
            self._chunk_seq = 0
            dist.barrier(group=group)
        self.disabled = False

    # ---- setup -------------------------------------------------------------

    def _discover_topology(self) -> Optional[_Topology]:
        my_bus = _ok(drv.cuDeviceGetPCIBusId(64, self.device.index)).split(b"\0")[0]
        self._bus_id = my_bus.decode()
        buses = [None] * _WORLD_SIZE
        dist.all_gather_object(buses, my_bus, group=self.group)
        row = []
        for j, b in enumerate(buses):
            if j == self.rank:
                row.append(False)
                continue
            err, other = drv.cuDeviceGetByPCIBusId(b)
            row.append(
                err == drv.CUresult.CUDA_SUCCESS
                and torch.cuda.can_device_access_peer(self.device.index, int(other))
            )
        p2p = [None] * _WORLD_SIZE
        dist.all_gather_object(p2p, row, group=self.group)
        return _pair_topology(self.rank, p2p)

    def _setup_device_buffers(self) -> None:
        # [recv_rs | recv_ag | recv_x]; the partner writes the first two.
        nbytes = 3 * self.half_cap
        self._dev_ptr = int(_ok(drv.cuMemAlloc(nbytes)))
        buf = torch.as_tensor(_RawCuda(self._dev_ptr, nbytes), device=self.device)
        self._recv_rs, self._recv_ag, self._recv_x = buf.split(self.half_cap)
        # cuda-python exposes the 64-byte handle only through its pointer.
        handle = ctypes.string_at(
            _ok(drv.cuIpcGetMemHandle(self._dev_ptr)).getPtr(), 64
        )
        handles = [None] * _WORLD_SIZE
        dist.all_gather_object(handles, handle, group=self.group)
        peer = drv.CUipcMemHandle()
        ctypes.memmove(peer.getPtr(), handles[self.topo.partner], 64)
        self._peer_ptr = int(
            _ok(
                drv.cuIpcOpenMemHandle(
                    peer, drv.CUipcMem_flags.CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
                )
            )
        )

    def _setup_host_region(self) -> None:
        flags_bytes = _round_up(_WORLD_SIZE * 4 * _FLAG_STRIDE, _ALIGN)
        size = flags_bytes + _WORLD_SIZE * self.half_cap
        name = [
            f"/dev/shm/sglang_pcie_ce_{uuid.uuid4().hex}" if self.rank == 0 else None
        ]
        dist.broadcast_object_list(
            name, src=dist.get_global_rank(self.group, 0), group=self.group
        )
        path = name[0]
        if self.rank == 0:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.ftruncate(fd, size)
            os.close(fd)
        dist.barrier(group=self.group)
        fd = os.open(path, os.O_RDWR)
        self._mmap = mmap.mmap(fd, size)
        os.close(fd)
        dist.barrier(group=self.group)
        self._host_buf = (ctypes.c_char * size).from_buffer(self._mmap)
        self._host_addr = ctypes.addressof(self._host_buf)
        if self.rank == 0:
            os.unlink(path)
            _first_touch_on_gpu_node(self._host_addr, size, self._bus_id)
        dist.barrier(group=self.group)
        _ok(
            drv.cuMemHostRegister(
                self._host_addr,
                size,
                drv.CU_MEMHOSTREGISTER_PORTABLE | drv.CU_MEMHOSTREGISTER_DEVICEMAP,
            )
        )
        flags_dev = int(_ok(drv.cuMemHostGetDevicePointer(self._host_addr, 0)))
        self._flag = [
            [flags_dev + (r * 4 + k) * _FLAG_STRIDE for k in range(4)]
            for r in range(_WORLD_SIZE)
        ]
        self._slot = [
            self._host_addr + flags_bytes + r * self.half_cap
            for r in range(_WORLD_SIZE)
        ]

    # ---- public API ----------------------------------------------------------

    def should_use(self, tensor: torch.Tensor) -> bool:
        return self.fits(tensor) and not torch.cuda.is_current_stream_capturing()

    def fits(self, tensor: torch.Tensor) -> bool:
        """Whether tensor suits this all-reduce. Its per-call counters are
        baked into whatever it enqueues, so it must run eagerly: outside graph
        capture, or as an eager break of a breakable graph."""
        return (
            not self.disabled
            and tensor.is_cuda
            and tensor.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and tensor.is_contiguous()
            and tensor.numel() * tensor.element_size()
            >= max(self.min_bytes, _MIN_BYTES_FLOOR)
        )

    def all_reduce(
        self, tensor: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Sum over the group into ``out``, a new tensor by default. ``out`` may be
        ``tensor``: each output range is written only after events that follow
        the last read of the same input range."""
        if out is None:
            out = torch.empty_like(tensor)
        flat_in, flat_out = tensor.view(-1), out.view(-1)
        n = flat_in.numel()
        # Equal segments, so none is too small to split into halves.
        n_segs = -(-n * tensor.element_size() // self.capacity_bytes)
        seg = _round_up(-(-n // n_segs), _ALIGN)
        for lo in range(0, n, seg):
            self._all_reduce_segment(flat_in[lo : lo + seg], flat_out[lo : lo + seg])
        return out

    def destroy(self) -> None:
        if self.disabled:
            return
        self.disabled = True
        torch.cuda.synchronize(self.device)
        for ev in self._events:
            drv.cuEventDestroy(ev)
        drv.cuMemHostUnregister(self._host_addr)
        drv.cuIpcCloseMemHandle(self._peer_ptr)
        # The partner must unmap my buffer before I free it.
        dist.barrier(group=self.group)
        drv.cuMemFree(self._dev_ptr)
        del self._host_buf
        self._mmap.close()

    # ---- one pipelined all-reduce ------------------------------------------

    def _all_reduce_segment(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        topo = self.topo
        me, partner, cross = self.rank, topo.partner, topo.cross
        es = inp.element_size()
        n = inp.numel()
        align = _ALIGN // es
        mid = n // 2 // align * align
        halves = ((0, mid), (mid, n))
        quant = self.quant is not None
        rs_fmt, x_fmt, ag_fmt = self.rs_fmt, self.x_fmt, self.ag_fmt
        n_chunks = mid * es // self.chunk_bytes
        if quant and x_fmt.codes in ("fp4", "fp6"):
            # About 3 MB of wire per chunk is fastest: fp4 took 0.68 ms for 4096
            # tokens in 1 or 4 chunks and 0.53 ms in 2. fp6/16 ran as fast in 3
            # MB chunks as in the 4 MB ones below, with fewer launches.
            wire = x_fmt.span(0, mid)[1]
            n_chunks = min(4, -(-wire // (3 << 20)))
        elif quant:
            # A quantized chunk puts half the bytes on the wire, so halve the
            # chunk count and the host launches with it, keeping up to 4
            # chunks to overlap the stages: 67 MB enqueues in 0.69 ms instead
            # of 1.3 ms, and 2 chunks made 33 MB 24% slower than 4.
            n_chunks = max(min(n_chunks, 4), n_chunks // 2)
        n_chunks = max(1, min(_MAX_CHUNKS, n_chunks))
        mine = _chunks(*halves[topo.half], n_chunks, align)
        theirs = _chunks(*halves[1 - topo.half], n_chunks, align)
        my_lo, their_lo = mine[0][0], theirs[0][0]

        call = self._calls + 1
        seq = self._chunk_seq
        self._calls = call
        self._chunk_seq += n_chunks
        prev = (call - 1) & _U32
        # Pair members take turns on D2H: they share one uplink, and the copy
        # engine that runs a D2H also runs the P2P pushes. Taking turns lets each
        # D2H run at full uplink speed while the other member's engine pushes.
        turn = topo.half

        in_ptr, out_ptr = inp.data_ptr(), out.data_ptr()
        rs_ptr, ag_ptr, x_ptr = (
            self._recv_rs.data_ptr(),
            self._recv_ag.data_ptr(),
            self._recv_x.data_ptr(),
        )
        peer_rs, peer_ag = self._peer_ptr, self._peer_ptr + self.half_cap
        my_slot, cross_slot = self._slot[me], self._slot[cross]
        flag = self._flag
        memcpy = drv.cuMemcpyAsync
        wait, write = drv.cuStreamWaitValue32, drv.cuStreamWriteValue32
        record, wait_event = drv.cuEventRecord, drv.cuStreamWaitEvent
        s_out, s_k0, s_k1, s_h2d, s_k2, s_loc = self._cu
        t_k0, t_k1, t_k2, t_loc = (self._streams[i] for i in (1, 2, 4, 5))
        ev = self._events
        encoded, summed, landed, reduced = (ev[i * _MAX_CHUNKS :] for i in range(4))
        start, joins = ev[4 * _MAX_CHUNKS], ev[4 * _MAX_CHUNKS + 1 :]

        def span(fmt: Optional[_WireFormat], a: int, b: int) -> Tuple[int, int]:
            return fmt.span(a, b) if quant else (a * es, (b - a) * es)

        caller = drv.CUstream(torch.cuda.current_stream(self.device).cuda_stream)
        record(start, caller)
        for s in self._cu:
            wait_event(s, start, 0)
        # Remote buffers I write must be free from the previous call.
        wait(s_out, flag[partner][_DONE], prev, _WAIT_GEQ)
        wait(s_out, flag[cross][_DONE], prev, _WAIT_GEQ)

        for kind, c in _outbound_schedule(n_chunks, _LEAD, _LAG):
            tag = (seq + c + 1) & _U32
            if kind == "rs":
                # 1a. partner's half of my input -> partner's recv_rs
                lo, hi = theirs[c]
                off, nbytes = span(rs_fmt, lo - their_lo, hi - their_lo)
                src = in_ptr + lo * es
                if quant:
                    # Staged in my recv_ag: the partner's all-gather of this
                    # chunk lands there only after this push has completed.
                    with torch.cuda.stream(t_k0):
                        off_stage = self._stage_base + off
                        codes = rs_fmt.views(self._recv_ag, off_stage, hi - lo)
                        self._codec(
                            _quant_kernel,
                            inp[lo:hi],
                            *codes,
                            n=hi - lo,
                            MODE=rs_fmt.mode,
                        )
                    record(encoded[c], s_k0)
                    wait_event(s_out, encoded[c], 0)
                    src = ag_ptr + off_stage
                memcpy(peer_rs + off, src, nbytes, s_out)
                write(s_out, flag[me][_RS], tag, _WRITE_FENCED)
                continue
            lo, hi = mine[c]
            a, b = lo - my_lo, hi - my_lo
            off, nbytes = span(ag_fmt, a, b)
            if kind == "ag":
                # 3a. my finished half -> partner's recv_ag
                wait_event(s_out, reduced[c], 0)
                src = x_ptr + off if quant else out_ptr + lo * es
                memcpy(peer_ag + off, src, nbytes, s_out)
                write(s_out, flag[me][_AG], tag, _WRITE_FENCED)
                # 3b. partner's finished half -> out
                lo, hi = theirs[c]
                off, nbytes = span(ag_fmt, lo - their_lo, hi - their_lo)
                wait(s_loc, flag[partner][_AG], tag, _WAIT_GEQ)
                if quant:
                    with torch.cuda.stream(t_loc):
                        codes = ag_fmt.views(self._recv_ag, off, hi - lo)
                        self._codec(
                            _decode_kernel,
                            *codes,
                            out[lo:hi],
                            n=hi - lo,
                            MODE=ag_fmt.mode,
                        )
                else:
                    memcpy(out_ptr + lo * es, ag_ptr + off, nbytes, s_loc)
                continue
            # The cross-pair payload's place in recv_rs, the host slots and recv_x.
            xoff, xbytes = span(x_fmt, a, b)
            xoff += self._x_base
            # 1b. recv_rs = recv_rs + my half of my input
            wait(s_k1, flag[partner][_RS], tag, _WAIT_GEQ)
            with torch.cuda.stream(t_k1):
                if quant:
                    self._codec(
                        _decode_add_encode_kernel,
                        *rs_fmt.views(self._recv_rs, span(rs_fmt, a, b)[0], b - a),
                        inp[lo:hi],
                        *x_fmt.views(self._recv_rs, xoff, b - a),
                        n=b - a,
                        MODE_IN=rs_fmt.mode,
                        MODE_OUT=x_fmt.mode,
                    )
                else:
                    part = self._recv_rs.view(inp.dtype)[a:b]
                    self._add(part, inp[lo:hi], part)
            record(summed[c], s_k1)
            # 2a. partial sum -> my host slot, alternating with the partner
            wait_event(s_out, summed[c], 0)
            wait(s_out, flag[partner][_X], (seq + c + turn) & _U32, _WAIT_GEQ)
            memcpy(my_slot + xoff, rs_ptr + xoff, xbytes, s_out)
            write(s_out, flag[me][_X], tag, _WRITE_FENCED)
            # 2b. cross peer's partial sum -> recv_x
            wait(s_h2d, flag[cross][_X], tag, _WAIT_GEQ)
            memcpy(x_ptr + xoff, cross_slot + xoff, xbytes, s_h2d)
            record(landed[c], s_h2d)
            # 2c. out[my half] = recv_rs + recv_x
            wait_event(s_k2, summed[c], 0)
            wait_event(s_k2, landed[c], 0)
            with torch.cuda.stream(t_k2):
                if quant:
                    # The lower rank's pair goes first on both cross peers.
                    # The encoded result stays in recv_x for the all-gather,
                    # so the partner decodes exactly my out.
                    ours = x_fmt.views(self._recv_rs, xoff, b - a)
                    theirs_x = x_fmt.views(self._recv_x, xoff, b - a)
                    first, second = (ours, theirs_x) if me < cross else (theirs_x, ours)
                    self._codec(
                        _decode2_add_encode_kernel,
                        *first,
                        *second,
                        *ag_fmt.views(self._recv_x, off, b - a),
                        out[lo:hi],
                        n=b - a,
                        MODE_IN=x_fmt.mode,
                        MODE_OUT=ag_fmt.mode,
                    )
                else:
                    rs, x = (
                        t.view(inp.dtype)[a:b] for t in (self._recv_rs, self._recv_x)
                    )
                    self._add(rs, x, out[lo:hi])
            record(reduced[c], s_k2)

        for s, j in zip(self._cu, joins):
            record(j, s)
            wait_event(caller, j, 0)
        write(caller, flag[me][_DONE], call & _U32, _WRITE_FENCED)

    def _add(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
        # 4-warp blocks: under a saturating GEMM, 1-4 warps finished a 1 MB
        # all-reduce in 1.4-1.6 ms against 2.2-2.3 ms for 8 warps; idle is equal.
        _add_kernel[(self.num_ctas,)](a, b, out, a.numel(), BLOCK=2048, num_warps=4)

    def _codec(self, kernel, *args, n: int, **modes: int) -> None:
        qb = self.rs_fmt.block
        kernel[(self.num_ctas,)](*args, n, QB=qb, ROWS=2048 // qb, num_warps=4, **modes)
