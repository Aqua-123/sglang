# SPDX-License-Identifier: Apache-2.0
"""One-kernel all-reduce for 4 GPUs wired as two PCIe-switch pairs.

Same topology and algorithm as ``pcie_ce_allreduce`` (reduce-scatter inside the pair over
P2P, cross-pair exchange through pinned host memory, all-gather inside the pair), but every
transfer is a store from SMs and every handoff is a flag polled inside the kernel. The
copy-engine version pays ~0.4 ms of stream-handoff latency per call; this one has none, at
the cost of SM time, and it runs inside captured CUDA graphs. It serves decode-sized and
medium reductions; the copy engines stay faster for large prefill messages.

Each block owns a fixed slice of the message and synchronizes only with the blocks that own
the same slice on the other ranks, so there is no grid-wide barrier. Flags carry a per-block
epoch kept in device memory (graph replays advance it too); every rank runs the same sequence
of reductions, so the epochs agree. Receive buffers and host slots alternate by epoch parity,
so a fast rank never overwrites data a slower rank has yet to read.

Every rank computes each element as ``bf16(a + b) + bf16(c + d)`` with the pair holding rank
0 first, so all ranks hold bitwise-equal results.
"""

import ctypes
import logging
import mmap
import os
import uuid
from typing import Optional

import torch
import torch.distributed as dist
from cuda.bindings import driver as drv
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.pcie_ce_allreduce import (
    _ALIGN,
    _WORLD_SIZE,
    _first_touch_on_gpu_node,
    _ok,
    _pair_topology,
    _round_up,
)

logger = logging.getLogger(__name__)

_MAX_BLOCKS = 128
_THREADS = 512
_VEC_ELEMS = 8  # bf16 per 16-byte vector

_CUDA_SRC = r"""
#include <cuda_bf16.h>
#include <stdint.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

__device__ __forceinline__ uint32_t ld_acquire(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint4 ld_volatile(const uint4* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ void wait_flag(const uint32_t* p, uint32_t epoch) {
  while ((int32_t)(ld_acquire(p) - epoch) < 0) {
  }
}
__device__ __forceinline__ uint4 add8(uint4 a, uint4 b) {
  const __nv_bfloat162* x = reinterpret_cast<const __nv_bfloat162*>(&a);
  const __nv_bfloat162* y = reinterpret_cast<const __nv_bfloat162*>(&b);
  uint4 r;
  __nv_bfloat162* z = reinterpret_cast<__nv_bfloat162*>(&r);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 fx = __bfloat1622float2(x[i]), fy = __bfloat1622float2(y[i]);
    z[i] = __floats2bfloat162_rn(fx.x + fy.x, fx.y + fy.y);
  }
  return r;
}

struct Args {
  const uint4* in;
  uint4* out;
  uint32_t* counters;        // [blocks], local, this rank's epochs
  uint32_t* my_flags;        // [2][MAXB] rs, ag flags the partner sets in my memory
  uint32_t* peer_flags;      // the partner's my_flags, mapped over P2P
  uint4* my_rs;              // [2][half_cap] partner pushes its copy of my half here
  uint4* peer_rs;
  uint4* my_ag;              // [2][half_cap] partner pushes its finished half here
  uint4* peer_ag;
  uint32_t* host_flags;      // [4][MAXB], host mapped
  uint4* host_slots;         // [4][2][half_cap], host mapped
  int64_t half_cap;          // vectors per half buffer
  int64_t nvh;               // vectors per half of this message
  int me, cross, half, low_first;
};

// Loads of a 16-byte vector cross PCIe and take microseconds; issue U of them before using
// any, or each thread waits one round trip per vector.
#define U 4
#define FOR_U(i, LOAD, USE)                                                        \
  for (int64_t base = lo + threadIdx.x; base < hi; base += blockDim.x * U) {      \
    uint4 v[U];                                                                   \
    _Pragma("unroll") for (int u = 0; u < U; u++) {                               \
      int64_t i = base + u * blockDim.x;                                          \
      if (i < hi) LOAD                                                            \
    }                                                                             \
    _Pragma("unroll") for (int u = 0; u < U; u++) {                               \
      int64_t i = base + u * blockDim.x;                                          \
      if (i < hi) USE                                                             \
    }                                                                             \
  }

template <int MAXB>
__global__ void __launch_bounds__(512) pair_host_all_reduce(Args a) {
  __shared__ uint32_t s_epoch;
  const int b = blockIdx.x, nb = gridDim.x;
  if (threadIdx.x == 0) s_epoch = a.counters[b] + 1;
  __syncthreads();
  const uint32_t epoch = s_epoch;
  const int par = epoch & 1;
  const int64_t per = (a.nvh + nb - 1) / nb;
  const int64_t lo = b * per, hi = min(a.nvh, lo + per);
  const int64_t mine = a.half ? a.nvh : 0, theirs = a.half ? 0 : a.nvh;

  // 1. Partner's half of my input -> partner's rs buffer.
  uint4* prs = a.peer_rs + par * a.half_cap;
  FOR_U(i, {
    v[u] = a.in[theirs + i];
  }, {
    prs[i] = v[u];
  });
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    st_release(a.peer_flags + b, epoch);
    wait_flag(a.my_flags + b, epoch);
  }
  __syncthreads();

  // 2. Pair partial of my half -> out (scratch) and my host slot.
  const uint4* mrs = a.my_rs + par * a.half_cap;
  uint4* slot = a.host_slots + ((int64_t)a.me * 2 + par) * a.half_cap;
  FOR_U(i, {
    v[u] = ld_volatile(mrs + i);
  }, {
    uint4 s = add8(a.in[mine + i], v[u]);
    a.out[mine + i] = s;
    slot[i] = s;
  });
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    st_release(a.host_flags + a.me * MAXB + b, epoch);
    wait_flag(a.host_flags + a.cross * MAXB + b, epoch);
  }
  __syncthreads();

  // 3. Add the cross pair's partial; push the result to the partner.
  const uint4* xs = a.host_slots + ((int64_t)a.cross * 2 + par) * a.half_cap;
  uint4* pag = a.peer_ag + par * a.half_cap;
  FOR_U(i, {
    v[u] = ld_volatile(xs + i);
  }, {
    uint4 m = a.out[mine + i];
    uint4 t = a.low_first ? add8(m, v[u]) : add8(v[u], m);
    a.out[mine + i] = t;
    pag[i] = t;
  });
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    st_release(a.peer_flags + MAXB + b, epoch);
    wait_flag(a.my_flags + MAXB + b, epoch);
  }
  __syncthreads();

  // 4. Partner's finished half -> out.
  const uint4* mag = a.my_ag + par * a.half_cap;
  FOR_U(i, {
    v[u] = ld_volatile(mag + i);
  }, {
    a.out[theirs + i] = v[u];
  });
  if (threadIdx.x == 0) a.counters[b] = epoch;
}

void all_reduce(torch::Tensor in, torch::Tensor out, torch::Tensor counters, int64_t my_dev,
                int64_t peer_dev, int64_t host, int64_t half_cap, int64_t blocks, int64_t me,
                int64_t cross, int64_t half, int64_t low_first) {
  const int MAXB = 128;
  Args a;
  int64_t flags_bytes = 2 * MAXB * 4;
  int64_t region = (flags_bytes + 4095) / 4096 * 4096;
  a.in = reinterpret_cast<const uint4*>(in.data_ptr());
  a.out = reinterpret_cast<uint4*>(out.data_ptr());
  a.counters = reinterpret_cast<uint32_t*>(counters.data_ptr());
  a.my_flags = reinterpret_cast<uint32_t*>(my_dev);
  a.peer_flags = reinterpret_cast<uint32_t*>(peer_dev);
  a.half_cap = half_cap;
  a.my_rs = reinterpret_cast<uint4*>(my_dev + region);
  a.peer_rs = reinterpret_cast<uint4*>(peer_dev + region);
  a.my_ag = a.my_rs + 2 * half_cap;
  a.peer_ag = a.peer_rs + 2 * half_cap;
  int64_t host_flags_bytes = (4 * MAXB * 4 + 4095) / 4096 * 4096;
  a.host_flags = reinterpret_cast<uint32_t*>(host);
  a.host_slots = reinterpret_cast<uint4*>(host + host_flags_bytes);
  a.nvh = in.numel() / 16;
  a.me = me;
  a.cross = cross;
  a.half = half;
  a.low_first = low_first;
  auto stream = at::cuda::getCurrentCUDAStream();
  pair_host_all_reduce<128><<<blocks, 512, 0, stream>>>(a);
}
"""

_CPP_SRC = """
void all_reduce(torch::Tensor in, torch::Tensor out, torch::Tensor counters, int64_t my_dev,
                int64_t peer_dev, int64_t host, int64_t half_cap, int64_t blocks, int64_t me,
                int64_t cross, int64_t half, int64_t low_first);
"""

_ext = None


def _load_ext():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load_inline

        major, minor = torch.cuda.get_device_capability()
        _ext = load_inline(
            name=f"pcie_pair_allreduce_v2_sm{major}{minor}",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["all_reduce"],
            extra_cuda_cflags=[
                "-O3",
                f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}",
            ],
            verbose=False,
        )
    return _ext


class PciePairAllReduce:
    """Sum all-reduce of bf16 tensors over 4 ranks, one kernel per call.

    ``group`` is a CPU (gloo) group over the 4 ranks; it carries the one-time handshake
    only. Serves tensors up to ``max_bytes``.
    """

    def __init__(self, group: ProcessGroup, device: torch.device, max_bytes: int = 8 << 20):
        self.disabled = True
        self.group = group
        self.device = torch.device(device)
        self.max_bytes = max_bytes
        self.rank = dist.get_rank(group)
        self._dev_ptr = self._peer_ptr = self._host_addr = 0
        self._mmap = None
        if dist.get_world_size(group) != _WORLD_SIZE:
            return
        with torch.cuda.device(self.device):
            topo = self._discover_topology()
            if topo is None:
                logger.warning("PCIe pair all-reduce disabled: not two P2P pairs.")
                return
            self.topo = topo
            self.low_first = int(min(self.rank, topo.partner) == 0)
            self._ext = _load_ext()
            # One half of the largest message, in 16-byte vectors.
            self.half_cap = _round_up(max_bytes // 2, _ALIGN) // 16
            self._setup_device_buffer()
            self._setup_host_region()
            self._counters = torch.zeros(_MAX_BLOCKS, dtype=torch.int32, device=self.device)
            torch.cuda.synchronize(self.device)
            dist.barrier(group=group)
        self.disabled = False

    def _discover_topology(self):
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

    def _setup_device_buffer(self) -> None:
        # [flags (rs, ag) | rs[2][half] | ag[2][half]]; the partner writes all of it.
        nbytes = _ALIGN + 4 * self.half_cap * 16
        self._dev_ptr = int(_ok(drv.cuMemAlloc(nbytes)))
        _ok(drv.cuMemsetD8(self._dev_ptr, 0, nbytes))
        handle = ctypes.string_at(_ok(drv.cuIpcGetMemHandle(self._dev_ptr)).getPtr(), 64)
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
        size = _ALIGN + _WORLD_SIZE * 2 * self.half_cap * 16
        name = [f"/dev/shm/sglang_pcie_pair_{uuid.uuid4().hex}" if self.rank == 0 else None]
        dist.broadcast_object_list(name, src=dist.get_global_rank(self.group, 0), group=self.group)
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
        self._host_dev = int(_ok(drv.cuMemHostGetDevicePointer(self._host_addr, 0)))

    def should_use(self, tensor: torch.Tensor) -> bool:
        return (
            not self.disabled
            and tensor.is_cuda
            and tensor.dtype == torch.bfloat16
            and tensor.is_contiguous()
            and tensor.numel() % 16 == 0
            and tensor.numel() * 2 <= self.max_bytes
        )

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(tensor)
        nvh = tensor.numel() // 16
        # Measured on 8x RTX PRO 6000: 8 blocks win up to 256 tokens of hidden 4096
        # (44 us vs 75 us with 64 at 64 tokens), 32 blocks beyond.
        blocks = 8 if nvh <= 65536 else 32
        self._ext.all_reduce(
            tensor, out, self._counters, self._dev_ptr, self._peer_ptr, self._host_dev,
            self.half_cap, blocks, self.rank, self.topo.cross, self.topo.half, self.low_first,
        )
        return out

    def destroy(self) -> None:
        if self.disabled:
            return
        self.disabled = True
        torch.cuda.synchronize(self.device)
        drv.cuMemHostUnregister(self._host_addr)
        drv.cuIpcCloseMemHandle(self._peer_ptr)
        dist.barrier(group=self.group)
        drv.cuMemFree(self._dev_ptr)
        del self._host_buf
        self._mmap.close()
