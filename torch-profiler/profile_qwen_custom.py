import argparse
import os
import torch
import torch.nn as nn
import cupy as cp
import cuda.tile as ct
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm


class CuTileRMSNorm(nn.Module):
    def __init__(self, original_norm):
        super().__init__()
        self.weight = original_norm.weight
        self.variance_epsilon = original_norm.variance_epsilon
        self.hidden_size = original_norm.weight.shape[0]

        self.block_rows = 64
        self.block_cols = 128
        
        self.kernel = self._compile_kernel(self.hidden_size, self.block_rows, self.block_cols)

        # Share original weight tensor with CuPy using DLPack (zero-copy)
        # Weight shape is (hidden_size,), add a batch dim: (1, hidden_size)
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

        # Pad rows to be a multiple of block_rows
        pad_rows = (self.block_rows - (original_rows % self.block_rows)) % self.block_rows
        if pad_rows > 0:
            x_2d = torch.nn.functional.pad(x_2d, (0, 0, 0, pad_rows))

        padded_rows = x_2d.shape[0]
        y_2d = torch.empty_like(x_2d)

        # DLPack zero-copy sharing
        x_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x_2d))
        y_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(y_2d))

        # Launch the custom kernel on PyTorch's current CUDA stream
        grid = (padded_rows // self.block_rows, 1, 1)
        stream = torch.cuda.current_stream()

        ct.launch(stream, grid, self.kernel, (x_cp, self.weight_cp, y_cp, self.variance_epsilon))

        # Slice back to original dimensions if we padded
        if pad_rows > 0:
            y_2d = y_2d[:original_rows, :]

        return y_2d.view(input_shape).to(hidden_states.dtype)


def replace_rmsnorm_with_cutile(model):
    replaced_count = 0
    # Walk modules to swap RMSNorm instances
    for name, module in model.named_modules():
        for child_name, child_module in module.named_children():
            if isinstance(child_module, Qwen2RMSNorm):
                setattr(module, child_name, CuTileRMSNorm(child_module))
                replaced_count += 1
    print(f"Successfully replaced {replaced_count} Qwen2RMSNorm instances with CuTileRMSNorm.")


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--warmup_steps", type=int, default=3)
    p.add_argument("--active_steps", type=int, default=3)
    p.add_argument("--trace_dir", default="./traces/qwen_custom_profile")
    return p.parse_args()


def main():
    args = parse_arguments()
    os.makedirs(args.trace_dir, exist_ok=True)

    print(f"Loading tokenizer and model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).cuda()

    replace_rmsnorm_with_cutile(model)

    print("Preparing input prompt...")
    prompt = "Tell me a story about a programmer writing highly optimized GPU kernels using tiling techniques in CUDA."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print(f"Performing {args.warmup_steps} warmup steps...")
    for i in range(args.warmup_steps):
        with torch.no_grad():
            _ = model(**inputs)
    torch.cuda.synchronize()

    print("Starting profiling session...")
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=args.active_steps, repeat=1)
    
    trace_path = os.path.join(args.trace_dir, "qwen_custom_trace.json")
    table_path = os.path.join(args.trace_dir, "qwen_custom_operators.txt")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        total_steps = 1 + 1 + args.active_steps
        for step in range(total_steps):
            with torch.no_grad():
                _ = model(**inputs)
            prof.step()
            print(f"Profiler step {step + 1}/{total_steps} completed.")

    torch.cuda.synchronize()

    print(f"Saving Chrome Trace to: {trace_path}")
    prof.export_chrome_trace(trace_path)

    print(f"Saving operator execution averages table to: {table_path}")
    with open(table_path, "w") as f:
        f.write(prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total", row_limit=30))

    print("\nSUCCESS! Custom Profiling complete. Please compare with original operators list.")


if __name__ == "__main__":
    main()
