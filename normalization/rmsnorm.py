import cuda.tile as ct
import cupy as cp
import numpy as np

ROWS, COLS = 512, 512

BLOCK_ROWS = 64
BLOCK_COLS = 128


@ct.kernel
def rmsnorm_kernel(inp: ct.Tile, weight: ct.Tile, out: ct.Tile, eps: float):
    pid = ct.bid(0)

    running_sum_sq = ct.zeros(shape=(BLOCK_ROWS, 1), dtype=ct.float32)

    num_col_tiles = ct.cdiv(COLS, BLOCK_COLS)
    for k in range(num_col_tiles):
        x = ct.load(inp, (pid, k), shape=(BLOCK_ROWS, BLOCK_COLS))
        x_sq = ct.mul(x, x)
        tile_sum_sq = ct.sum(x_sq, axis=1, keepdims=True)
        running_sum_sq = ct.add(running_sum_sq, tile_sum_sq)

    mean_sq = ct.truediv(running_sum_sq, float(COLS))
    inv_rms = ct.rsqrt(ct.add(mean_sq, eps))

    for k in range(num_col_tiles):
        x = ct.load(inp, (pid, k), shape=(BLOCK_ROWS, BLOCK_COLS))
        w = ct.load(weight, (0, k), shape=(1, BLOCK_COLS))
        x_norm = ct.mul(x, inv_rms)
        res = ct.mul(x_norm, w)
        ct.store(out, (pid, k), res)


def main():
    print(f"Running RMSNorm on a {ROWS}x{COLS} matrix...")

    x = cp.random.randn(ROWS, COLS).astype(cp.float32)
    gamma = cp.random.randn(1, COLS).astype(cp.float32)
    y = cp.zeros((ROWS, COLS), dtype=cp.float32)
    eps = 1e-5

    grid = (ct.cdiv(ROWS, BLOCK_ROWS), 1, 1)
    print(f"Grid: {grid} | Tile: {BLOCK_ROWS}x{BLOCK_COLS}")

    stream = cp.cuda.get_current_stream()
    ct.launch(stream, grid, rmsnorm_kernel, (x, gamma, y, eps))
    stream.synchronize()

    x_np = x.get()
    gamma_np = gamma.get()
    
    rms_np = np.sqrt(np.mean(x_np**2, axis=1, keepdims=True) + eps)
    expected = (x_np / rms_np) * gamma_np

    cp.testing.assert_allclose(y, expected, atol=1e-5, rtol=1e-4)
    print("SUCCESS! cuTile RMSNorm matched NumPy result.")


if __name__ == "__main__":
    main()
