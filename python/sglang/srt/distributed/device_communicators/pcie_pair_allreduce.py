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

Optionally, large messages travel as e4m3 codes with a bf16 scale per 16 or 32 elements. The
pair partials are quantized once, and both ranks that own a half add the same decoded values,
so the results stay bitwise equal across ranks.
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
#include <cuda_fp16.h>
#include <cuda_fp8.h>
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

// FP8 wire format: groups of VG vectors (8*VG elements) share one bf16 scale, rounded up from
// amax/448 so no code saturates. A half buffer holds the codes (nvh/2 vectors) followed by the
// group scales. Wire buffers are planar (vector j of group g at j * groups + g) so each warp
// store is one contiguous run; strided 16-byte stores split into small PCIe writes. Every value
// a rank reduces is decoded from codes all ranks see, and fp addition commutes, so all ranks
// still hold bitwise-equal results.
__device__ __forceinline__ uint16_t ld_volatile_u16(const uint16_t* p) {
  uint16_t v;
  asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(v) : "l"(p));
  return v;
}
template <int VG>
__device__ __forceinline__ void unpack_bf16(const uint4 (&v)[VG], float (&x)[VG * 8]) {
#pragma unroll
  for (int j = 0; j < VG; j++) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v[j]);
#pragma unroll
    for (int k = 0; k < 4; k++) {
      float2 f = __bfloat1622float2(h[k]);
      x[j * 8 + 2 * k] = f.x;
      x[j * 8 + 2 * k + 1] = f.y;
    }
  }
}
template <int VG>
__device__ __forceinline__ void pack_bf16(const float (&x)[VG * 8], uint4 (&v)[VG]) {
#pragma unroll
  for (int j = 0; j < VG; j++) {
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&v[j]);
#pragma unroll
    for (int k = 0; k < 4; k++) h[k] = __floats2bfloat162_rn(x[j * 8 + 2 * k], x[j * 8 + 2 * k + 1]);
  }
}
template <int VG>
__device__ __forceinline__ uint16_t encode(const float (&x)[VG * 8], uint4 (&c)[VG / 2]) {
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < VG * 8; i++) amax = fmaxf(amax, fabsf(x[i]));
  const __nv_bfloat16 s = __float2bfloat16_ru(amax * (1.f / 448.f));
  const float sf = __bfloat162float(s);
  const float inv = sf > 0.f ? 1.f / sf : 0.f;
  uint32_t w[VG * 2];
#pragma unroll
  for (int i = 0; i < VG * 2; i++) {
    const uint32_t lo = __nv_cvt_float2_to_fp8x2(
        make_float2(x[4 * i] * inv, x[4 * i + 1] * inv), __NV_SATFINITE, __NV_E4M3);
    const uint32_t hi = __nv_cvt_float2_to_fp8x2(
        make_float2(x[4 * i + 2] * inv, x[4 * i + 3] * inv), __NV_SATFINITE, __NV_E4M3);
    w[i] = lo | (hi << 16);
  }
#pragma unroll
  for (int m = 0; m < VG / 2; m++) c[m] = make_uint4(w[4 * m], w[4 * m + 1], w[4 * m + 2], w[4 * m + 3]);
  return __bfloat16_as_ushort(s);
}
template <int VG>
__device__ __forceinline__ void decode(const uint4 (&c)[VG / 2], uint16_t s, float (&x)[VG * 8]) {
  const float sf = __bfloat162float(__ushort_as_bfloat16(s));
#pragma unroll
  for (int m = 0; m < VG / 2; m++) {
    const uint32_t w[4] = {c[m].x, c[m].y, c[m].z, c[m].w};
#pragma unroll
    for (int k = 0; k < 8; k++) {
      __half2_raw r = __nv_cvt_fp8x2_to_halfraw2(
          static_cast<__nv_fp8x2_storage_t>(w[k / 2] >> (16 * (k % 2))), __NV_E4M3);
      const float2 f = __half22float2(__half2(r));
      x[16 * m + 2 * k] = f.x * sf;
      x[16 * m + 2 * k + 1] = f.y * sf;
    }
  }
}
// Sum of two fp32 vectors rounded to bf16, the same rounding add8 applies.
template <int N>
__device__ __forceinline__ void add_round(float (&x)[N], const float (&y)[N]) {
#pragma unroll
  for (int i = 0; i < N; i++) x[i] = __bfloat162float(__float2bfloat16_rn(x[i] + y[i]));
}

#define UF 2
#define FOR_G(G, LOAD, USE)                                                        \
  for (int64_t base = lo + threadIdx.x; base < hi; base += blockDim.x * UF) {     \
    _Pragma("unroll") for (int u = 0; u < UF; u++) {                              \
      int64_t G = base + u * blockDim.x;                                          \
      if (G < hi) LOAD                                                            \
    }                                                                             \
    _Pragma("unroll") for (int u = 0; u < UF; u++) {                              \
      int64_t G = base + u * blockDim.x;                                          \
      if (G < hi) USE                                                             \
    }                                                                             \
  }

template <int MAXB, int VG, bool AG_FP8>
__global__ void __launch_bounds__(512) pair_host_all_reduce_fp8(Args a) {
  constexpr int CV = VG / 2;
  __shared__ uint32_t s_epoch;
  const int b = blockIdx.x, nb = gridDim.x;
  if (threadIdx.x == 0) s_epoch = a.counters[b] + 1;
  __syncthreads();
  const uint32_t epoch = s_epoch;
  const int par = epoch & 1;
  const int64_t ng = a.nvh / VG;
  const int64_t per = (ng + nb - 1) / nb;
  const int64_t lo = b * per, hi = min(ng, lo + per);
  const int64_t mine = a.half ? a.nvh : 0, theirs = a.half ? 0 : a.nvh;
  // Codes start each half buffer; the group scales follow them.
  auto scales = [&](const uint4* buf) {
    return reinterpret_cast<uint16_t*>(const_cast<uint4*>(buf) + a.nvh / 2);
  };
  uint4 v[UF][VG];
  uint4 c[UF][CV];
  uint16_t s[UF];
  float x[VG * 8], y[VG * 8];

  // 1. Partner's half of my input, encoded -> partner's rs buffer.
  uint4* prs = a.peer_rs + par * a.half_cap;
  FOR_G(g, {
    _Pragma("unroll") for (int j = 0; j < VG; j++) v[u][j] = a.in[theirs + g * VG + j];
  }, {
    unpack_bf16<VG>(v[u], x);
    s[u] = encode<VG>(x, c[u]);
    _Pragma("unroll") for (int j = 0; j < CV; j++) prs[j * ng + g] = c[u][j];
    scales(prs)[g] = s[u];
  });
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    st_release(a.peer_flags + b, epoch);
    wait_flag(a.my_flags + b, epoch);
  }
  __syncthreads();

  // 2. Pair partial of my half, encoded -> my host slot, and its codes -> out (scratch).
  uint4* mrs = a.my_rs + par * a.half_cap;
  uint4* slot = a.host_slots + ((int64_t)a.me * 2 + par) * a.half_cap;
  FOR_G(g, {
    _Pragma("unroll") for (int j = 0; j < CV; j++) c[u][j] = ld_volatile(mrs + j * ng + g);
    s[u] = ld_volatile_u16(scales(mrs) + g);
    _Pragma("unroll") for (int j = 0; j < VG; j++) v[u][j] = a.in[mine + g * VG + j];
  }, {
    unpack_bf16<VG>(v[u], x);
    decode<VG>(c[u], s[u], y);
    add_round(x, y);
    s[u] = encode<VG>(x, c[u]);
    _Pragma("unroll") for (int j = 0; j < CV; j++) {
      slot[j * ng + g] = c[u][j];
      a.out[mine + g * VG + j] = c[u][j];
    }
    scales(slot)[g] = s[u];
    reinterpret_cast<uint16_t*>(a.out + mine + g * VG + CV)[0] = s[u];
  });
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    st_release(a.host_flags + a.me * MAXB + b, epoch);
    wait_flag(a.host_flags + a.cross * MAXB + b, epoch);
  }
  __syncthreads();

  // 3. Both pair partials decoded and added; the result -> out and the partner.
  const uint4* xs = a.host_slots + ((int64_t)a.cross * 2 + par) * a.half_cap;
  uint4* pag = a.peer_ag + par * a.half_cap;
  FOR_G(g, {
    _Pragma("unroll") for (int j = 0; j < CV; j++) c[u][j] = ld_volatile(xs + j * ng + g);
    s[u] = ld_volatile_u16(scales(xs) + g);
  }, {
    decode<VG>(c[u], s[u], y);
    uint4 own[CV];
    _Pragma("unroll") for (int j = 0; j < CV; j++) own[j] = a.out[mine + g * VG + j];
    decode<VG>(own, reinterpret_cast<const uint16_t*>(a.out + mine + g * VG + CV)[0], x);
    add_round(x, y);
    if constexpr (AG_FP8) {
      s[u] = encode<VG>(x, c[u]);
      _Pragma("unroll") for (int j = 0; j < CV; j++) pag[j * ng + g] = c[u][j];
      scales(pag)[g] = s[u];
      decode<VG>(c[u], s[u], x);
    }
    pack_bf16<VG>(x, v[u]);
    _Pragma("unroll") for (int j = 0; j < VG; j++) {
      a.out[mine + g * VG + j] = v[u][j];
      if constexpr (!AG_FP8) pag[j * ng + g] = v[u][j];
    }
  });
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    st_release(a.peer_flags + MAXB + b, epoch);
    wait_flag(a.my_flags + MAXB + b, epoch);
  }
  __syncthreads();

  // 4. Partner's finished half -> out.
  uint4* mag = a.my_ag + par * a.half_cap;
  FOR_G(g, {
    if constexpr (AG_FP8) {
      _Pragma("unroll") for (int j = 0; j < CV; j++) c[u][j] = ld_volatile(mag + j * ng + g);
      s[u] = ld_volatile_u16(scales(mag) + g);
    } else {
      _Pragma("unroll") for (int j = 0; j < VG; j++) v[u][j] = ld_volatile(mag + j * ng + g);
    }
  }, {
    if constexpr (AG_FP8) {
      decode<VG>(c[u], s[u], x);
      pack_bf16<VG>(x, v[u]);
    }
    _Pragma("unroll") for (int j = 0; j < VG; j++) a.out[theirs + g * VG + j] = v[u][j];
  });
  if (threadIdx.x == 0) a.counters[b] = epoch;
}

static Args make_args(torch::Tensor in, torch::Tensor out, torch::Tensor counters,
                      int64_t my_dev, int64_t peer_dev, int64_t host, int64_t half_cap,
                      int64_t me, int64_t cross, int64_t half, int64_t low_first) {
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
  return a;
}

void all_reduce(torch::Tensor in, torch::Tensor out, torch::Tensor counters, int64_t my_dev,
                int64_t peer_dev, int64_t host, int64_t half_cap, int64_t blocks, int64_t me,
                int64_t cross, int64_t half, int64_t low_first) {
  Args a = make_args(in, out, counters, my_dev, peer_dev, host, half_cap, me, cross, half,
                     low_first);
  auto stream = at::cuda::getCurrentCUDAStream();
  pair_host_all_reduce<128><<<blocks, 512, 0, stream>>>(a);
}

void all_reduce_fp8(torch::Tensor in, torch::Tensor out, torch::Tensor counters,
                    int64_t my_dev, int64_t peer_dev, int64_t host, int64_t half_cap,
                    int64_t blocks, int64_t me, int64_t cross, int64_t half, int64_t low_first,
                    int64_t group, int64_t quant_ag) {
  Args a = make_args(in, out, counters, my_dev, peer_dev, host, half_cap, me, cross, half,
                     low_first);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (group == 16 && quant_ag) {
    pair_host_all_reduce_fp8<128, 2, true><<<blocks, 512, 0, stream>>>(a);
  } else if (group == 16) {
    pair_host_all_reduce_fp8<128, 2, false><<<blocks, 512, 0, stream>>>(a);
  } else if (group == 32 && quant_ag) {
    pair_host_all_reduce_fp8<128, 4, true><<<blocks, 512, 0, stream>>>(a);
  } else if (group == 32) {
    pair_host_all_reduce_fp8<128, 4, false><<<blocks, 512, 0, stream>>>(a);
  } else {
    TORCH_CHECK(false, "fp8 scale group must be 16 or 32");
  }
}
"""

_CPP_SRC = """
void all_reduce(torch::Tensor in, torch::Tensor out, torch::Tensor counters, int64_t my_dev,
                int64_t peer_dev, int64_t host, int64_t half_cap, int64_t blocks, int64_t me,
                int64_t cross, int64_t half, int64_t low_first);
void all_reduce_fp8(torch::Tensor in, torch::Tensor out, torch::Tensor counters,
                    int64_t my_dev, int64_t peer_dev, int64_t host, int64_t half_cap,
                    int64_t blocks, int64_t me, int64_t cross, int64_t half, int64_t low_first,
                    int64_t group, int64_t quant_ag);
"""

_ext = None


def _load_ext():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load_inline

        major, minor = torch.cuda.get_device_capability()
        _ext = load_inline(
            name=f"pcie_pair_allreduce_v3_sm{major}{minor}",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["all_reduce", "all_reduce_fp8"],
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
    only. Serves tensors up to ``max_bytes``. Tensors of at least ``fp8_min_bytes`` (0: never)
    cross the reduce-scatter and the host exchange as e4m3 codes with one bf16 scale per
    ``fp8_group`` elements, and the all-gather too when ``fp8_all_gather`` is set.
    """

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_bytes: int = 8 << 20,
        fp8_min_bytes: int = 0,
        fp8_group: int = 16,
        fp8_all_gather: bool = False,
    ):
        self.disabled = True
        self.group = group
        self.device = torch.device(device)
        self.max_bytes = max_bytes
        if fp8_group not in (16, 32):
            raise ValueError(f"fp8_group must be 16 or 32, got {fp8_group}")
        self.fp8_min_bytes = fp8_min_bytes
        self.fp8_group = fp8_group
        self.fp8_all_gather = fp8_all_gather
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
        quantize = (
            self.fp8_min_bytes > 0
            and tensor.numel() * 2 >= self.fp8_min_bytes
            and nvh % (self.fp8_group // _VEC_ELEMS) == 0
        )
        # Measured on 8x RTX PRO 6000: 8 blocks win up to 256 tokens of hidden 4096
        # (44 us vs 75 us with 64 at 64 tokens), 32 blocks beyond. The FP8 kernel takes 8 at
        # every size from 64 to 512 tokens (192 tokens: 79 us vs 82 with 16 and 87 with 32).
        blocks = 8 if quantize or nvh <= 65536 else 32
        args = (
            tensor, out, self._counters, self._dev_ptr, self._peer_ptr, self._host_dev,
            self.half_cap, blocks, self.rank, self.topo.cross, self.topo.half, self.low_first,
        )
        if quantize:
            self._ext.all_reduce_fp8(*args, self.fp8_group, int(self.fp8_all_gather))
        else:
            self._ext.all_reduce(*args)
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
