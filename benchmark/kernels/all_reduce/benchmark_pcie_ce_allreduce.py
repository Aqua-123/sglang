"""Benchmark PcieCeAllReduce against NCCL on 4 GPUs, idle and under SM contention.

    torchrun --nproc-per-node 4 benchmark_pcie_ce_allreduce.py [--sizes-tokens ...]

Messages are [tokens, 4096] bf16, so 128/512/2048/4096/8192 tokens are
1/4/16/33/67 MB. Under contention every GPU loops a bf16 matmul on a side
stream for the whole timed window.
"""

import argparse
import os

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
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
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
    p.add_argument("--skip-contention", action="store_true")
    args = p.parse_args()

    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank = dist.get_rank()
    cpu_group = dist.new_group(backend="gloo")
    comm = PcieCeAllReduce(
        cpu_group,
        torch.device("cuda", torch.cuda.current_device()),
        min_bytes=0,
        chunk_bytes=int(args.chunk_mb * (1 << 20)),
        num_ctas=args.ctas,
    )
    assert not comm.disabled, "CE all-reduce did not initialize"
    warm = torch.zeros(64, HIDDEN, dtype=torch.bfloat16, device="cuda")
    dist.all_reduce(warm)

    rows, mismatches = check_correctness(comm, args.sizes_tokens, args.rounds)
    gemm = None if args.skip_contention else Gemm(args.gemm_n)
    if rank == 0:
        print(f"topology {comm.topo} chunk {args.chunk_mb} MB ctas {args.ctas}")
        print(
            f"stress: {args.rounds * 6} back-to-back calls, {mismatches} mismatches vs exact reference"
        )
        if gemm:
            print(
                f"contention gemm {args.gemm_n}^3 bf16: {gemm.ms:.2f} ms, {gemm.flops / gemm.ms / 1e9:.0f} TFLOPS"
            )
        print(
            f"{'MB':>6} {'maxdiff':>8} {'exact':>5} {'ranks=':>6} | {'idle CE':>8} {'NCCL':>8} {'x':>5}"
            + ("" if gemm is None else f" | {'busy CE':>8} {'NCCL':>8} {'x':>5}")
        )
    for t in args.sizes_tokens:
        x = torch.randn(t, HIDDEN, dtype=torch.bfloat16, device="cuda")
        scratch = torch.zeros_like(x)
        ce = lambda: comm.all_reduce(x)
        nccl = lambda: dist.all_reduce(scratch)
        ce_ms, nccl_ms = time_loop(ce, args.iters), time_loop(nccl, args.iters)
        line = (
            f"{x.numel() * 2 / 1e6:6.1f} {rows[t][0]:8.4f} {str(rows[t][1]):>5} {str(rows[t][2]):>6} | "
            f"{ce_ms:8.3f} {nccl_ms:8.3f} {nccl_ms / ce_ms:5.2f}"
        )
        if gemm is not None:
            ce_b = time_under_contention(ce, args.iters, gemm, ce_ms)
            nccl_b = time_under_contention(nccl, args.iters, gemm, nccl_ms)
            line += f" | {ce_b:8.3f} {nccl_b:8.3f} {nccl_b / ce_b:5.2f}"
        if rank == 0:
            print(line, flush=True)
    comm.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
