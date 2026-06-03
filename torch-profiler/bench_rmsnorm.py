"""Benchmark: PyTorch native RMSNorm vs cuTile vs Triton at different sequence lengths."""
import argparse
import time
import torch
import torch.nn as nn
import triton
import triton.language as tl
import cupy as cp
import cuda.tile as ct
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm


# ──────────────────────────────────────────────────────────────
# TRITON RMSNorm Kernel
# ──────────────────────────────────────────────────────────────
@triton.jit
def triton_rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride,       # row stride (number of elements between rows)
    N,            # number of columns (hidden_size)
    eps,          # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row = tl.program_id(0)
    X_ptr += row * stride
    Y_ptr += row * stride

    # Pass 1: Compute sum of squares
    sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += x * x

    # Compute inverse RMS: 1 / sqrt(mean(x^2) + eps)
    mean_sq = tl.sum(sum_sq) / N
    inv_rms = 1.0 / tl.sqrt(mean_sq + eps)

    # Pass 2: Normalize and scale
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * inv_rms * w
        tl.store(Y_ptr + cols, y, mask=mask)


class TritonRMSNorm(nn.Module):
    def __init__(self, original_norm):
        super().__init__()
        self.weight = original_norm.weight
        self.variance_epsilon = original_norm.variance_epsilon
        self.hidden_size = original_norm.weight.shape[0]

    def forward(self, hidden_states):
        input_shape = hidden_states.shape
        x = hidden_states.view(-1, self.hidden_size)
        num_rows = x.shape[0]
        y = torch.empty_like(x)

        # Pick a BLOCK_SIZE that covers the full row in one pass if possible
        BLOCK_SIZE = triton.next_power_of_2(self.hidden_size)

        # Launch one program per row
        triton_rmsnorm_kernel[(num_rows,)](
            x, self.weight, y,
            x.stride(0),
            self.hidden_size,
            self.variance_epsilon,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return y.view(input_shape)


# ──────────────────────────────────────────────────────────────
# cuTile RMSNorm (same as before)
# ──────────────────────────────────────────────────────────────
class CuTileRMSNorm(nn.Module):
    def __init__(self, original_norm):
        super().__init__()
        self.weight = original_norm.weight
        self.variance_epsilon = original_norm.variance_epsilon
        self.hidden_size = original_norm.weight.shape[0]
        self.block_rows = 64
        self.block_cols = 128
        self.kernel = self._compile_kernel(self.hidden_size, self.block_rows, self.block_cols)
        self.weight_cp = cp.from_dlpack(
            torch.utils.dlpack.to_dlpack(self.weight.data.unsqueeze(0))
        )

    def _compile_kernel(self, cols, block_rows, block_cols):
        @ct.kernel
        def rmsnorm_kernel(inp: ct.Tile, weight: ct.Tile, out: ct.Tile, eps: float):
            pid = ct.bid(0)
            running_sum_sq = ct.zeros(shape=(block_rows, 1), dtype=ct.float32)
            num_col_tiles = ct.cdiv(cols, block_cols)
            for k in range(num_col_tiles):
                x = ct.load(inp, (pid, k), shape=(block_rows, block_cols))
                x_sq = ct.mul(x, x)
                tile_sum_sq = ct.sum(x_sq, axis=1, keepdims=True)
                running_sum_sq = ct.add(running_sum_sq, tile_sum_sq)
            mean_sq = ct.truediv(running_sum_sq, float(cols))
            inv_rms = ct.rsqrt(ct.add(mean_sq, eps))
            for k in range(num_col_tiles):
                x = ct.load(inp, (pid, k), shape=(block_rows, block_cols))
                w = ct.load(weight, (0, k), shape=(1, block_cols))
                x_norm = ct.mul(x, inv_rms)
                res = ct.mul(x_norm, w)
                ct.store(out, (pid, k), res)
        return rmsnorm_kernel

    def forward(self, hidden_states):
        input_shape = hidden_states.shape
        x_2d = hidden_states.view(-1, self.hidden_size).float()
        original_rows = x_2d.shape[0]
        pad_rows = (self.block_rows - (original_rows % self.block_rows)) % self.block_rows
        if pad_rows > 0:
            x_2d = torch.nn.functional.pad(x_2d, (0, 0, 0, pad_rows))
        padded_rows = x_2d.shape[0]
        y_2d = torch.empty_like(x_2d)
        x_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x_2d))
        y_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(y_2d))
        grid = (padded_rows // self.block_rows, 1, 1)
        stream = torch.cuda.current_stream()
        ct.launch(stream, grid, self.kernel, (x_cp, self.weight_cp, y_cp, self.variance_epsilon))
        if pad_rows > 0:
            y_2d = y_2d[:original_rows, :]
        return y_2d.view(input_shape).to(hidden_states.dtype)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def replace_rmsnorm(model, replacement_cls):
    count = 0
    for name, module in model.named_modules():
        for child_name, child_module in module.named_children():
            if isinstance(child_module, Qwen2RMSNorm):
                setattr(module, child_name, replacement_cls(child_module))
                count += 1
    return count


def make_input_ids(tokenizer, seq_len):
    seed = "Tell me a story about a programmer writing highly optimized GPU kernels."
    seed_ids = tokenizer.encode(seed)
    repeated = (seed_ids * ((seq_len // len(seed_ids)) + 1))[:seq_len]
    return torch.tensor([repeated], device="cuda")


def benchmark_forward(model, input_ids, warmup=5, runs=10):
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_ids=input_ids)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        with torch.no_grad():
            _ = model(input_ids=input_ids)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / runs * 1000


def bench_variant(label, model, tokenizer, seq_lengths, warmup, runs):
    print(f"\n{'=' * 80}")
    print(f"  Benchmarking: {label}")
    print(f"{'=' * 80}")
    times = {}
    for seq_len in seq_lengths:
        input_ids = make_input_ids(tokenizer, seq_len)
        t = benchmark_forward(model, input_ids, warmup=warmup, runs=runs)
        times[seq_len] = t
        print(f"  seq_len={seq_len:>5d}  |  {t:>8.2f} ms/forward")
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    seq_lengths = [64, 128, 256, 512, 1024]
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ── 1. Original PyTorch ──
    print(f"Loading model: {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda().eval()
    orig_times = bench_variant("ORIGINAL PyTorch RMSNorm", model, tokenizer, seq_lengths, args.warmup, args.runs)
    del model; torch.cuda.empty_cache()

    # ── 2. cuTile ──
    print(f"\nReloading model for cuTile...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda().eval()
    n = replace_rmsnorm(model, CuTileRMSNorm)
    print(f"Replaced {n} layers.")
    cutile_times = bench_variant("cuTile RMSNorm", model, tokenizer, seq_lengths, args.warmup, args.runs)
    del model; torch.cuda.empty_cache()

    # ── 3. Triton ──
    print(f"\nReloading model for Triton...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda().eval()
    n = replace_rmsnorm(model, TritonRMSNorm)
    print(f"Replaced {n} layers.")
    triton_times = bench_variant("Triton RMSNorm", model, tokenizer, seq_lengths, args.warmup, args.runs)
    del model; torch.cuda.empty_cache()

    # ── Comparison ──
    print(f"\n{'=' * 100}")
    print(f"  COMPARISON TABLE")
    print(f"{'=' * 100}")
    print(f"  {'Seq Len':>8s}  |  {'Original':>10s}  |  {'cuTile':>10s}  |  {'Triton':>10s}  |  {'cuTile vs Orig':>14s}  |  {'Triton vs Orig':>14s}")
    print(f"  {'-'*8}  |  {'-'*10}  |  {'-'*10}  |  {'-'*10}  |  {'-'*14}  |  {'-'*14}")
    for s in seq_lengths:
        o, c, t = orig_times[s], cutile_times[s], triton_times[s]
        cs = o / c
        ts = o / t
        cm = "✓" if cs > 1 else "✗"
        tm = "✓" if ts > 1 else "✗"
        print(f"  {s:>8d}  |  {o:>8.2f}ms  |  {c:>8.2f}ms  |  {t:>8.2f}ms  |  {cs:>7.2f}x {cm:>5s}  |  {ts:>7.2f}x {tm:>5s}")

    print(f"\n{'=' * 100}")
    print("  DONE!")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
