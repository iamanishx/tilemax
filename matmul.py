import cuda.tile as ct
import cupy as cp

# Full matrix dimensions
M, N, K = 256, 256, 256

# How big each block's workload is
BLOCK_M = 64   # rows each block handles
BLOCK_N = 64   # cols each block handles
BLOCK_K = 32   # how much of K we chew through per iteration

@ct.kernel
def matmul_kernel(A, B, C):
    # Which block am I? (my row-block index and col-block index)
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    # My running total, starts at zero
    acc = ct.zeros(shape=(BLOCK_M, BLOCK_N), dtype=ct.float32)

    # Walk along K in small steps, loading a piece of A and B each time
    num_k_tiles = ct.cdiv(K, BLOCK_K)
    for k in range(num_k_tiles):
        # Grab my row-strip chunk from A
        a_tile = ct.load(A, (pid_m, k), shape=(BLOCK_M, BLOCK_K))
        # Grab my col-strip chunk from B
        b_tile = ct.load(B, (k, pid_n), shape=(BLOCK_K, BLOCK_N))
        # Multiply and add to running total
        acc = ct.mma(a_tile, b_tile, acc)

    # Done accumulating, write my 64x64 result chunk back to C
    ct.store(C, (pid_m, pid_n), acc)

def main():
    print(f"Multiplying {M}x{K} and {K}x{N} matrices...")

    # Create random inputs on GPU
    a = cp.random.randn(M, K).astype(cp.float16)
    b = cp.random.randn(K, N).astype(cp.float16)
    c = cp.zeros((M, N), dtype=cp.float32)

    # Launch a 4x4 grid of blocks (each block computes a 64x64 piece of C)
    grid = (ct.cdiv(M, BLOCK_M), ct.cdiv(N, BLOCK_N), 1)

    print(f"Grid: {grid}")
    print(f"Tile size: {BLOCK_M}x{BLOCK_N}")

    stream = cp.cuda.get_current_stream()
    ct.launch(stream, grid, matmul_kernel, (a, b, c))
    stream.synchronize()

    # Verify against numpy on CPU
    a_np = a.get().astype('float32')
    b_np = b.get().astype('float32')
    expected = a_np @ b_np

    cp.testing.assert_allclose(c, expected, atol=1e-1, rtol=1e-2)
    print("SUCCESS! cuTile Matmul matched NumPy result.")

if __name__ == "__main__":
    main()
