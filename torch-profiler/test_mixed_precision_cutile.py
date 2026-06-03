import cuda.tile as ct
import cupy as cp

ROWS, COLS = 64, 128
BLOCK_ROWS, BLOCK_COLS = 64, 128

@ct.kernel
def test_kernel(inp: ct.Tile, out: ct.Tile):
    pid = ct.bid(0)
    running_sum = ct.zeros(shape=(BLOCK_ROWS, 1), dtype=ct.float32)
    x = ct.load(inp, (pid, 0), shape=(BLOCK_ROWS, BLOCK_COLS))
    # Test x.astype(ct.float32) or ct.astype(x, ct.float32)
    x_f32 = ct.astype(x, ct.float32)
    tile_sum = ct.sum(x_f32, axis=1, keepdims=True)
    running_sum = ct.add(running_sum, tile_sum)
    # Store back to out (which is float16, so let's convert back to float16)
    out_tile = ct.astype(x_f32, ct.float16)
    ct.store(out, (pid, 0), out_tile)

def main():
    import torch
    print("Testing passing PyTorch tensors directly to ct.launch...")
    # float16 PyTorch tensors on CUDA
    x = torch.randn(ROWS, COLS, dtype=torch.float16, device="cuda")
    y = torch.zeros(ROWS, COLS, dtype=torch.float16, device="cuda")
    
    grid = (1, 1, 1)
    # Get standard PyTorch stream or CUDA stream
    stream = torch.cuda.current_stream()
    try:
        ct.launch(stream, grid, test_kernel, (x, y))
        torch.cuda.synchronize()
        print("PyTorch tensor launch successful!")
        diff = (x - y).abs().max().item()
        print(f"Max diff: {diff}")
        assert torch.allclose(x, y, atol=1e-5, rtol=1e-5)
        print("Values match perfectly!")
    except Exception as e:
        print(f"Failed to launch with PyTorch tensors directly: {e}")

if __name__ == "__main__":
    main()
