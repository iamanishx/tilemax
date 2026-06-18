import cuda.tile as ct
import cupy as cp

M, N, K = 256, 256, 256

BLOCK_M = 64   
BLOCK_N = 64  
BLOCK_K = 32  

@ct.kernel
def matmul_kernel(A, B, C):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    acc = ct.zeros(shape=(BLOCK_M, BLOCK_N), dtype=ct.float32)

    num_k_tiles = ct.cdiv(K, BLOCK_K)
    for k in range(num_k_tiles):
        a_tile = ct.load(A, (pid_m, k), shape=(BLOCK_M, BLOCK_K))
        b_tile = ct.load(B, (k, pid_n), shape=(BLOCK_K, BLOCK_N))
        acc = ct.mma(a_tile, b_tile, acc)

    ct.store(C, (pid_m, pid_n), acc)

def main():
    print(f"Multiplying {M}x{K} and {K}x{N} matrices...")

    a = cp.random.randn(M, K).astype(cp.float16)
    b = cp.random.randn(K, N).astype(cp.float16)
    c = cp.zeros((M, N), dtype=cp.float32)

    grid = (ct.cdiv(M, BLOCK_M), ct.cdiv(N, BLOCK_N), 1)

    print(f"Grid: {grid}")
    print(f"Tile size: {BLOCK_M}x{BLOCK_N}")

    stream = cp.cuda.get_current_stream()
    ct.launch(stream, grid, matmul_kernel, (a, b, c))
    stream.synchronize()

    a_np = a.get().astype('float32')
    b_np = b.get().astype('float32')
    expected = a_np @ b_np

    cp.testing.assert_allclose(c, expected, atol=1e-1, rtol=1e-2)
    print("SUCCESS! cuTile Matmul matched NumPy result.")

if __name__ == "__main__":
    main()
