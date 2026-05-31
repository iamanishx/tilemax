import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--warmup_steps", type=int, default=3)
    p.add_argument("--active_steps", type=int, default=3)
    p.add_argument("--trace_dir", default="./traces/qwen_profile")
    return p.parse_args()


def main():
    args = parse_arguments()
    os.makedirs(args.trace_dir, exist_ok=True)

    print(f"Loading tokenizer and model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).cuda()

    print("Preparing input prompt...")
    prompt = "Tell me a story about a programmer writing highly optimized GPU kernels using tiling techniques in CUDA."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # 1. Warm up the GPU
    # This ensures CUDA contexts are initialized, autotuning is done,
    # and model weights/buffers are warm.
    print(f"Performing {args.warmup_steps} warmup steps...")
    for i in range(args.warmup_steps):
        with torch.no_grad():
            _ = model(**inputs)
    torch.cuda.synchronize()

    # 2. Setup PyTorch Profiler
    print("Starting profiling session...")
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=args.active_steps, repeat=1)
    
    trace_path = os.path.join(args.trace_dir, "qwen_trace.json")
    table_path = os.path.join(args.trace_dir, "qwen_operators.txt")

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
        # We run warmup_steps + active_steps + wait times to satisfy the schedule
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
        # Sort by total CUDA time to see exactly which kernels are taking the most GPU time
        f.write(prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total", row_limit=30))

    print("\nSUCCESS! Profiling complete. Please examine your trace and operator table.")


if __name__ == "__main__":
    main()
