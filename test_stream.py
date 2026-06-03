import cuda.tile as ct
import cupy as cp
import torch

@ct.kernel
def dummy_kernel(inp, out):
    pid = ct.bid(0)
    x = ct.load(inp, (pid, 0), shape=(64, 128))
    ct.store(out, (pid, 0), x)

print("Trying torch.cuda.current_stream()...")
try:
    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), dummy_kernel, (cp.zeros((64, 128)), cp.zeros((64, 128))))
    print("SUCCESS with torch.cuda.current_stream()")
except Exception as e:
    print(f"FAILED with torch.cuda.current_stream(): {e}")

print("Trying torch.cuda.current_stream().cuda_stream (raw int)...")
try:
    stream_ptr = torch.cuda.current_stream().cuda_stream
    ct.launch(stream_ptr, (1, 1, 1), dummy_kernel, (cp.zeros((64, 128)), cp.zeros((64, 128))))
    print("SUCCESS with raw int stream")
except Exception as e:
    print(f"FAILED with raw int stream: {e}")

print("Trying cp.cuda.get_current_stream()...")
try:
    stream = cp.cuda.get_current_stream()
    ct.launch(stream, (1, 1, 1), dummy_kernel, (cp.zeros((64, 128)), cp.zeros((64, 128))))
    print("SUCCESS with cupy get_current_stream")
except Exception as e:
    print(f"FAILED with cupy get_current_stream: {e}")
