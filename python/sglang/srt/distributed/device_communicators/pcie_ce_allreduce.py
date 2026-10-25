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
_STAGES = ("out", "k1", "h2d", "k2", "loc")
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
    ):
        self.disabled = True
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
            self.capacity_bytes = capacity_bytes
            self._setup_device_buffers()
            self._setup_host_region()
            # Add kernels go on top-priority streams (torch clamps -100 to the
            # highest): with the SMs saturated, their pending blocks are
            # scheduled ahead of the compute stream's.
            self._streams = [
                torch.cuda.Stream(
                    self.device, priority=-100 if st in ("k1", "k2") else 0
                )
                for st in _STAGES
            ]
            self._cu = [drv.CUstream(st.cuda_stream) for st in self._streams]
            no_timing = drv.CUevent_flags.CU_EVENT_DISABLE_TIMING
            # summed / landed / reduced per chunk, then one per stream for the join.
            self._events = [
                _ok(drv.cuEventCreate(no_timing))
                for _ in range(3 * _MAX_CHUNKS + len(_STAGES) + 1)
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
        return (
            not self.disabled
            and tensor.is_cuda
            and tensor.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and tensor.is_contiguous()
            and tensor.numel() * tensor.element_size()
            >= max(self.min_bytes, _MIN_BYTES_FLOOR)
            and not torch.cuda.is_current_stream_capturing()
        )

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Out-of-place sum over the group; ``tensor`` is not modified."""
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
        n_chunks = max(1, min(_MAX_CHUNKS, mid * es // self.chunk_bytes))
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
        recv_rs = self._recv_rs.view(inp.dtype)
        recv_x = self._recv_x.view(inp.dtype)
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
        s_out, s_k1, s_h2d, s_k2, s_loc = self._cu
        t_k1, t_k2 = self._streams[1], self._streams[3]
        ev = self._events
        summed, landed, reduced = ev[0:], ev[_MAX_CHUNKS:], ev[2 * _MAX_CHUNKS :]
        start, joins = ev[3 * _MAX_CHUNKS], ev[3 * _MAX_CHUNKS + 1 :]

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
                memcpy(
                    peer_rs + (lo - their_lo) * es,
                    in_ptr + lo * es,
                    (hi - lo) * es,
                    s_out,
                )
                write(s_out, flag[me][_RS], tag, _WRITE_FENCED)
                continue
            lo, hi = mine[c]
            a, b = lo - my_lo, hi - my_lo
            off, nbytes = a * es, (hi - lo) * es
            if kind == "ag":
                # 3a. my finished half -> partner's recv_ag
                wait_event(s_out, reduced[c], 0)
                memcpy(peer_ag + off, out_ptr + lo * es, nbytes, s_out)
                write(s_out, flag[me][_AG], tag, _WRITE_FENCED)
                # 3b. partner's finished half -> out
                lo, hi = theirs[c]
                wait(s_loc, flag[partner][_AG], tag, _WAIT_GEQ)
                memcpy(
                    out_ptr + lo * es,
                    ag_ptr + (lo - their_lo) * es,
                    (hi - lo) * es,
                    s_loc,
                )
                continue
            # 1b. recv_rs += my half of my input
            wait(s_k1, flag[partner][_RS], tag, _WAIT_GEQ)
            with torch.cuda.stream(t_k1):
                self._add(recv_rs[a:b], inp[lo:hi], recv_rs[a:b])
            record(summed[c], s_k1)
            # 2a. partial sum -> my host slot, alternating with the partner
            wait_event(s_out, summed[c], 0)
            wait(s_out, flag[partner][_X], (seq + c + turn) & _U32, _WAIT_GEQ)
            memcpy(my_slot + off, rs_ptr + off, nbytes, s_out)
            write(s_out, flag[me][_X], tag, _WRITE_FENCED)
            # 2b. cross peer's partial sum -> recv_x
            wait(s_h2d, flag[cross][_X], tag, _WAIT_GEQ)
            memcpy(x_ptr + off, cross_slot + off, nbytes, s_h2d)
            record(landed[c], s_h2d)
            # 2c. out[my half] = recv_rs + recv_x
            wait_event(s_k2, summed[c], 0)
            wait_event(s_k2, landed[c], 0)
            with torch.cuda.stream(t_k2):
                self._add(recv_rs[a:b], recv_x[a:b], out[lo:hi])
            record(reduced[c], s_k2)

        for s, j in zip(self._cu, joins):
            record(j, s)
            wait_event(caller, j, 0)
        write(caller, flag[me][_DONE], call & _U32, _WRITE_FENCED)

    def _add(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
        # 4-warp blocks: under a saturating GEMM, 1-4 warps finished a 1 MB
        # all-reduce in 1.4-1.6 ms against 2.2-2.3 ms for 8 warps; idle is equal.
        _add_kernel[(self.num_ctas,)](a, b, out, a.numel(), BLOCK=2048, num_warps=4)
