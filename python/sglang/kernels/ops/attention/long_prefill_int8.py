# SPDX-License-Identifier: Apache-2.0
"""Causal prefill attention with INT8 Q.K^T for long cached prefixes (SM120, head dims 192/128).

At long context a prefill chunk spends most of its time in Q.K^T and P.V of the full-attention
layers. This path quantizes Q per row and K per token to INT8 (K centered first: subtracting one
vector from every key adds a per-row constant to the scores, which softmax ignores), runs Q.K^T
on INT8 tensor cores and keeps P.V in BF16. V is read straight from the paged cache. Before
quantizing, K channels are scaled by c and Q channels by 1/c (Q.K unchanged), which evens out the
channel ranges between the two so outlier channels cost less INT8 resolution.

SGLANG_LONG_PREFILL_INT8_MIN_KV turns it on for batches holding a request with at least that many
KV tokens; 0 (the default) keeps every prefill on FA4. Shorter requests batched with such a request
take this path too.
"""

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

LOG2E = 1.4426950408889634
BM, BN, BT = 64, 64, 64
MIN_KV = int(os.environ.get("SGLANG_LONG_PREFILL_INT8_MIN_KV", "0"))
CHECK = os.environ.get("SGLANG_LONG_PREFILL_INT8_CHECK") == "1"


@triton.jit(do_not_specialize=["n"])
def _quant_k(K, IDX, MEAN, C, K8, KS, n, kpad, s_kt, s_kh, BT: tl.constexpr):
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    h = tl.program_id(1)
    valid = t < n
    slot = tl.load(IDX + t).to(tl.int64)
    d1 = tl.arange(0, 128)
    d2 = 128 + tl.arange(0, 64)
    base = K + slot[:, None] * s_kt + h * s_kh
    k1 = tl.load(base + d1[None, :]).to(tl.float32) - tl.load(MEAN + h * 192 + d1)[None, :]
    k2 = tl.load(base + d2[None, :]).to(tl.float32) - tl.load(MEAN + h * 192 + d2)[None, :]
    k1 = tl.where(valid[:, None], k1, 0.0) * tl.load(C + h * 192 + d1)[None, :]
    k2 = tl.where(valid[:, None], k2, 0.0) * tl.load(C + h * 192 + d2)[None, :]
    sc = tl.maximum(tl.maximum(tl.max(tl.abs(k1), 1), tl.max(tl.abs(k2), 1)), 1e-6) / 127.0
    out = K8 + h * kpad * 192 + t[:, None] * 192
    tl.store(out + d1[None, :], libdevice.rint(k1 / sc[:, None]).to(tl.int8))
    tl.store(out + d2[None, :], libdevice.rint(k2 / sc[:, None]).to(tl.int8))
    tl.store(KS + h * kpad + t, sc)


@triton.jit(do_not_specialize=["sq", "sk", "off", "kpad"])
def _attn(Q, CI, K8, KS, V, IDX, O, sq, sk, off, kpad, qk_scale,
          s_qt, s_qh, s_vt, s_vh, s_ot, s_oh,
          GQA: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pm = tl.program_id(0)
    h = tl.program_id(1)
    kh = h // GQA
    rm = pm * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    d1 = tl.arange(0, 128)
    d2 = 128 + tl.arange(0, 64)
    dv = tl.arange(0, 128)
    rv = rm < sq
    qp = Q + rm[:, None] * s_qt + h * s_qh
    q1 = tl.load(qp + d1[None, :], mask=rv[:, None], other=0.0).to(tl.float32) * tl.load(CI + kh * 192 + d1)[None, :]
    q2 = tl.load(qp + d2[None, :], mask=rv[:, None], other=0.0).to(tl.float32) * tl.load(CI + kh * 192 + d2)[None, :]
    qs = tl.maximum(tl.maximum(tl.max(tl.abs(q1), 1), tl.max(tl.abs(q2), 1)), 1e-6) / 127.0
    q1 = libdevice.rint(q1 / qs[:, None]).to(tl.int8)
    q2 = libdevice.rint(q2 / qs[:, None]).to(tl.int8)
    qs = qs * qk_scale
    m_i = tl.full([BM], -float("inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, 128], tl.float32)
    kp = K8 + kh * kpad * 192
    full_end = off + pm * BM
    end = tl.minimum(off + (pm + 1) * BM, sk)
    for start in range(0, end, BN):
        cols = start + rn
        k1 = tl.load(kp + cols[:, None] * 192 + d1[None, :])
        k2 = tl.load(kp + cols[:, None] * 192 + d2[None, :])
        s = tl.dot(q1, tl.trans(k1), out_dtype=tl.int32) + tl.dot(q2, tl.trans(k2), out_dtype=tl.int32)
        s = s.to(tl.float32) * tl.load(KS + kh * kpad + cols)[None, :] * qs[:, None]
        if start + BN > full_end:
            s = tl.where((off + rm[:, None]) >= cols[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        slot = tl.load(IDX + cols).to(tl.int64)
        v = tl.load(V + slot[:, None] * s_vt + kh * s_vh + dv[None, :])
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    out = acc / l_i[:, None]
    tl.store(O + rm[:, None] * s_ot + h * s_oh + dv[None, :], out.to(tl.bfloat16), mask=rv[:, None])


def wants(prefix_lens, ext_lens) -> bool:
    return MIN_KV > 0 and max(p + e for p, e in zip(prefix_lens, ext_lens)) >= MIN_KV


def long_prefill_attention(q, key_cache, value_cache, page_table, prefix_lens, ext_lens, sm_scale, out=None):
    """q [T, Hq, 192]; key/value_cache [slots, Hkv, 192|128]; page_table [B, >=kv] token slots.

    Requests follow the batch order of q. Returns out [T, Hq, 128]."""
    T, hq, _ = q.shape
    hk = key_cache.shape[1]
    o = out if out is not None else q.new_empty(T, hq, 128)
    qo = 0
    for b, (pre, ext) in enumerate(zip(prefix_lens, ext_lens)):
        sk = pre + ext
        kpad = triton.cdiv(sk, BT) * BT
        row = page_table[b, :sk]
        idx = torch.cat([row, row[-1:].expand(kpad - sk)])
        sample = row[:: max(1, sk // 4096)].long()
        ksample = key_cache[sample].float()
        mean = ksample.mean(0)
        qb, ob = q[qo:qo + ext], o[qo:qo + ext]
        qa = qb.float().abs().amax(0).view(hk, hq // hk, 192).amax(1)
        ka = (ksample - mean).abs().amax(0)
        c = (qa.clamp_min(1e-4) / ka.clamp_min(1e-4)).sqrt()
        mean, c, ci = mean.contiguous(), c.contiguous(), (1 / c).contiguous()
        k8 = torch.empty(hk, kpad, 192, dtype=torch.int8, device=q.device)
        ks = torch.empty(hk, kpad, dtype=torch.float32, device=q.device)
        _quant_k[(kpad // BT, hk)](key_cache, idx, mean, c, k8, ks, sk, kpad,
                                   key_cache.stride(0), key_cache.stride(1), BT=BT, num_warps=4)
        _attn[(triton.cdiv(ext, BM), hq)](
            qb, ci, k8, ks, value_cache, idx, ob, ext, sk, pre, kpad, sm_scale * LOG2E,
            qb.stride(0), qb.stride(1), value_cache.stride(0), value_cache.stride(1), ob.stride(0), ob.stride(1),
            GQA=hq // hk, BM=BM, BN=BN, num_warps=4, num_stages=2)
        qo += ext
    return o


def report(layer_id, forward_batch, out, ref):
    """Log this layer's error against the BF16 kernel (diagnostics, SGLANG_LONG_PREFILL_INT8_CHECK=1)."""
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    ref = ref[0] if isinstance(ref, tuple) else ref
    e = (out.float() - ref.float()).pow(2).mean().sqrt() / ref.float().pow(2).mean().sqrt()
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), 0)
    kv = max(p + n for p, n in zip(forward_batch.extend_prefix_lens_cpu, forward_batch.extend_seq_lens_cpu))
    print(f"LPCHECK layer {layer_id} kv {kv} rel_rms {e.item():.5f} cos {cos.item():.6f}", flush=True)
