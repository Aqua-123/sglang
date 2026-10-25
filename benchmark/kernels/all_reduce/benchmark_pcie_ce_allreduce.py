"""Benchmark PcieCeAllReduce against NCCL on 4 GPUs, idle and under SM contention.

    torchrun --nproc-per-node 4 benchmark_pcie_ce_allreduce.py [--sizes-tokens ...]

Messages are [tokens, 4096] bf16, so 128/512/2048/4096/8192 tokens are
1/4/16/33/67 MB. Under contention every GPU loops a bf16 matmul on a side
stream for the whole timed window. ``--quant fp4 fp8/16+fp4/16`` adds the
quantized wire modes and an accuracy table on hidden-state-like activations.
"""

import argparse
import os
import time

import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators.pcie_ce_allreduce import (
    PcieCeAllReduce,
)

HIDDEN = 4096


def exact_reference(x: torch.Tensor, topo) -> torch.Tensor:
    """bf16((a+b) + (c+d)) with the pairing the CE path uses."""
    xs = [torch.empty_like(x) for _ in range(4)]
    dist.all_gather(xs, x)
    rank = dist.get_rank()
    pairs = [sorted((rank, topo.partner))]
    pairs.append(sorted(set(range(4)) - set(pairs[0])))
    sums = [(xs[i].float() + xs[j].float()).to(x.dtype).float() for i, j in pairs]
    return (sums[0] + sums[1]).to(x.dtype)


def all_ranks_max(v: float) -> float:
    t = torch.tensor([v], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def check_correctness(comm, tokens_list, rounds):
    rank = dist.get_rank()
    rows = {}
    for t in tokens_list:
        g = torch.Generator(device="cuda").manual_seed(1000 + rank)
        x = torch.randn(t, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=g)
        nccl = x.clone()
        dist.all_reduce(nccl)
        got = comm.all_reduce(x)
        exact = exact_reference(x, comm.topo)
        gathered = [torch.empty_like(got) for _ in range(4)]
        dist.all_gather(gathered, got)
        same = all(torch.equal(gathered[0], g_) for g_ in gathered)
        rows[t] = (
            all_ranks_max((got.float() - nccl.float()).abs().max().item()),
            all_ranks_max(float(not torch.equal(got, exact))) == 0.0,
            same,
        )
    # Back-to-back calls with no sync between them, verified afterwards: catches
    # flag or buffer-reuse races that a synced single call would hide.
    mismatches = 0
    for r in range(rounds):
        sizes = [tokens_list[(r + i) % len(tokens_list)] for i in range(6)]
        ins, outs = [], []
        for i, t in enumerate(sizes):
            g = torch.Generator(device="cuda").manual_seed(r * 100 + i * 10 + rank)
            ins.append(
                torch.randn(t, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=g)
            )
            outs.append(comm.all_reduce(ins[-1]))
        for x, y in zip(ins, outs):
            mismatches += int(not torch.equal(y, exact_reference(x, comm.topo)))
    mismatches = int(all_ranks_max(float(mismatches)))
    return rows, mismatches


def time_loop(fn, iters):
    torch.cuda.synchronize()
    dist.barrier()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    host = time.perf_counter()
    for _ in range(iters):
        fn()
    host = (time.perf_counter() - host) * 1e3 / iters
    end.record()
    end.synchronize()
    time_loop.host_ms = all_ranks_max(host)
    return all_ranks_max(start.elapsed_time(end) / iters)


class Gemm:
    def __init__(self, n):
        self.a = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
        self.b = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
        self.c = torch.empty(n, n, dtype=torch.bfloat16, device="cuda")
        self.stream = torch.cuda.Stream()
        self.flops = 2 * n**3
        for _ in range(3):
            torch.matmul(self.a, self.b, out=self.c)
        torch.cuda.synchronize()
        self.ms = time_loop(lambda: torch.matmul(self.a, self.b, out=self.c), 10)

    def enqueue(self, count, after):
        self.stream.wait_event(after)
        with torch.cuda.stream(self.stream):
            for _ in range(count):
                torch.matmul(self.a, self.b, out=self.c)
            return self.stream.record_event(torch.cuda.Event(enable_timing=True))


def time_under_contention(fn, iters, gemm, idle_ms):
    """All-reduce ms/iter while every GPU loops a matmul for the whole window."""
    for _ in range(3):
        fn()
    budget = 6.0
    while True:
        torch.cuda.synchronize()
        dist.barrier()
        t0 = torch.cuda.Event(enable_timing=True)
        t0.record()
        n_gemm = int(budget * iters * idle_ms / gemm.ms) + 4
        gemm_end = gemm.enqueue(n_gemm, t0)
        for _ in range(iters):
            fn()
        ar_end = torch.cuda.Event(enable_timing=True)
        ar_end.record()
        torch.cuda.synchronize()
        ar_ms = t0.elapsed_time(ar_end)
        covered = all_ranks_max(float(t0.elapsed_time(gemm_end) < ar_ms)) == 0.0
        if covered:
            return all_ranks_max(ar_ms / iters)
        budget *= 4


def hidden_like(tokens: int, n_outliers: int, seed: int) -> torch.Tensor:
    """A TP rank's partial hidden state: Student-t(3) values, per-token scale
    spread, and outlier channels 50-100x the typical magnitude whose sign and
    size are shared across ranks, as massive activations are."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    z = torch.randn(tokens, HIDDEN, device="cuda", generator=g)
    chi2 = torch.randn(3, tokens, HIDDEN, device="cuda", generator=g).pow(2).sum(0)
    token_scale = torch.exp(0.5 * torch.randn(tokens, 1, device="cuda", generator=g))
    x = z / (chi2 / 3).sqrt() * token_scale
    shared = torch.Generator(device="cuda").manual_seed(7)
    ch = outlier_channels(n_outliers)
    size = 50 + 50 * torch.rand(n_outliers, device="cuda", generator=shared)
    sign = torch.randint(0, 2, (n_outliers,), device="cuda", generator=shared) * 2 - 1
    jitter = 1 + 0.1 * torch.randn(tokens, n_outliers, device="cuda", generator=g)
    x[:, ch] = sign * size * jitter * token_scale
    return x.bfloat16()


def outlier_channels(n: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(11)
    return torch.randperm(HIDDEN, device="cuda", generator=g)[:n]


def token_rel_err(got: torch.Tensor, exact: torch.Tensor):
    """Per-token ||got - exact|| / ||exact||, as (mean, max) over tokens."""
    err = (got.double() - exact).norm(dim=-1) / exact.norm(dim=-1)
    return all_ranks_max(err.mean().item()), all_ranks_max(err.max().item())


def accuracy(methods, tokens, n_outliers):
    rank = dist.get_rank()
    x = hidden_like(tokens, n_outliers, seed=500 + rank)
    xs = [torch.empty_like(x) for _ in range(4)]
    dist.all_gather(xs, x)
    exact = sum(t.double() for t in xs)
    normal = torch.ones(HIDDEN, dtype=torch.bool, device="cuda")
    normal[outlier_channels(n_outliers)] = False
    rows = {}
    for name, fn in methods.items():
        got = fn(x.clone())
        gathered = [torch.empty_like(got) for _ in range(4)]
        dist.all_gather(gathered, got)
        rows[name] = (
            token_rel_err(got, exact)
            + token_rel_err(got[:, normal], exact[:, normal])
            + (all(torch.equal(gathered[0], g) for g in gathered),)
        )
    return rows


def stress_quantized(comm, tokens_list, rounds):
    """Back-to-back calls verified afterwards. Races show up as stale chunks,
    far beyond quantization error, or as ranks that disagree."""
    rank, worst, unequal = dist.get_rank(), 0.0, 0
    for r in range(rounds):
        ins, outs = [], []
        for i in range(6):
            t = tokens_list[(r + i) % len(tokens_list)]
            g = torch.Generator(device="cuda").manual_seed(r * 100 + i * 10 + rank)
            ins.append(
                torch.randn(t, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=g)
            )
            outs.append(comm.all_reduce(ins[-1]))
        for x, y in zip(ins, outs):
            xs = [torch.empty_like(x) for _ in range(4)]
            dist.all_gather(xs, x)
            worst = max(worst, token_rel_err(y, sum(t.double() for t in xs))[1])
            ys = [torch.empty_like(y) for _ in range(4)]
            dist.all_gather(ys, y)
            unequal += int(not all(torch.equal(ys[0], v) for v in ys))
    return worst, unequal


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sizes-tokens", type=int, nargs="+", default=[128, 512, 2048, 4096, 8192]
    )
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--gemm-n", type=int, default=8192)
    p.add_argument("--chunk-mb", type=float, default=4)
    p.add_argument("--ctas", type=int, default=16)
    p.add_argument("--quant", nargs="*", default=[])
    p.add_argument("--accuracy-tokens", type=int, default=4096)
    p.add_argument("--outliers", type=int, default=8)
    p.add_argument("--skip-contention", action="store_true")
    args = p.parse_args()

    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank = dist.get_rank()
    cpu_group = dist.new_group(backend="gloo")
    comms = {
        name: PcieCeAllReduce(
            cpu_group,
            torch.device("cuda", torch.cuda.current_device()),
            min_bytes=0,
            chunk_bytes=int(args.chunk_mb * (1 << 20)),
            num_ctas=args.ctas,
            quant=quant,
        )
        for name, quant in [("ce", None)] + [(f"ce-{q}", q) for q in args.quant]
    }
    assert not any(c.disabled for c in comms.values()), (
        "CE all-reduce did not initialize"
    )
    comm = comms["ce"]
    warm = torch.zeros(64, HIDDEN, dtype=torch.bfloat16, device="cuda")
    dist.all_reduce(warm)

    def nccl(x):
        dist.all_reduce(x)
        return x

    methods = {"nccl": nccl, **{k: c.all_reduce for k, c in comms.items()}}
    rows, mismatches = check_correctness(comm, args.sizes_tokens, args.rounds)
    stress = {
        k: stress_quantized(c, args.sizes_tokens, args.rounds)
        for k, c in comms.items()
        if c.quant
    }
    acc = accuracy(methods, args.accuracy_tokens, args.outliers)
    gemm = None if args.skip_contention else Gemm(args.gemm_n)
    if rank == 0:
        print(f"topology {comm.topo} chunk {args.chunk_mb} MB ctas {args.ctas}")
        print(
            f"stress ce: {args.rounds * 6} back-to-back calls, {mismatches} mismatches vs exact reference"
        )
        for k, (worst, unequal) in stress.items():
            print(
                f"stress {k}: worst per-token rel err {worst:.4f}, {unequal} calls with unequal ranks"
            )
        print(
            f"accuracy, {args.accuracy_tokens} tokens hidden-like ({args.outliers} outlier channels), "
            "per-token rel err vs exact sum (mean / max):"
        )
        print(
            f"{'':>8} {'all ch mean':>11} {'max':>9} {'normal ch mean':>14} {'max':>9} ranks="
        )
        for k, (m, mx, nm, nmx, same) in acc.items():
            print(f"{k:>8} {m:11.2e} {mx:9.2e} {nm:14.2e} {nmx:9.2e} {same}")
        if gemm:
            print(f"contention gemm {args.gemm_n}^3 bf16: {gemm.ms:.2f} ms")
        busy = "" if gemm is None else " | busy " + " ".join(f"{k:>8}" for k in methods)
        names = " ".join(f"{k:>8}" for k in methods)
        print(
            f"{'MB':>6} {'maxdiff':>8} {'exact':>5} | idle {names} | host {names}{busy}"
        )
    for t in args.sizes_tokens:
        x = torch.randn(t, HIDDEN, dtype=torch.bfloat16, device="cuda")
        scratch = torch.zeros_like(x)
        fns = {
            k: (lambda f=f: f(scratch if f is nccl else x)) for k, f in methods.items()
        }
        idle, host = {}, {}
        for k, f in fns.items():
            idle[k] = time_loop(f, args.iters)
            host[k] = time_loop.host_ms
        line = f"{x.numel() * 2 / 1e6:6.1f} {rows[t][0]:8.4f} {str(rows[t][1] and rows[t][2]):>5} | idle "
        line += " ".join(f"{v:8.3f}" for v in idle.values())
        line += " | host " + " ".join(f"{v:8.3f}" for v in host.values())
        if gemm is not None:
            line += " | busy " + " ".join(
                f"{time_under_contention(f, args.iters, gemm, idle[k]):8.3f}"
                for k, f in fns.items()
            )
        if rank == 0:
            print(line, flush=True)
    for c in comms.values():
        c.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
