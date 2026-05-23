"""Numerically stable softmax kernel using cuTile reductions + element-wise ops.

API checklist (softmax vs matmul):
    ct.reduce   — generic reduction with custom op
    ct.max      — row-wise max for numerical stability
    ct.sum      — row-wise sum for denominator
    ct.exp      — element-wise exponential
    ct.sub      — element-wise subtract (x - max)
    ct.truediv  — element-wise divide (normalize)
    ct.where    — mask illegal values (optional, for masked softmax)

The key trick: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
Subtracting the max prevents overflow in exp().
"""

import cuda.tile as ct
import cupy as cp

ROWS, COLS = 512, 512

# Tile sizes
BLOCK_ROWS = 64  # rows per block
BLOCK_COLS = 128  # cols we process per iteration (softmax reduces across cols)


@ct.kernel
def softmax_kernel(inp: ct.Tile, out: ct.Tile):
    """Numerically stable row-wise softmax.

    Each block handles BLOCK_ROWS rows. It walks across the column dimension
    in tiles, computing max → exp → sum → normalize.
    """

    pid = ct.bid(0)

    # ---------- STEP 1: Find row-wise maximum ----------
    # We need max per row across ALL columns for numerical stability.
    # Start with -inf, then walk column tiles to find max.
    running_max = ct.full(
        shape=(BLOCK_ROWS, 1), fill_value=float("-inf"), dtype=ct.float32
    )

    num_col_tiles = ct.cdiv(COLS, BLOCK_COLS)
    for k in range(num_col_tiles):
        # Load a 64x128 tile of input
        x = ct.load(inp, (pid, k), shape=(BLOCK_ROWS, BLOCK_COLS))
        # Reduce to row-wise max: shape goes from (64, 128) → (64, 1)
        tile_max = ct.max(x, axis=1, keepdims=True)
        # Element-wise max with running max
        running_max = ct.maximum(running_max, tile_max)

    # ---------- STEP 2: Compute exp(x - max) and sum ----------
    running_sum = ct.zeros(shape=(BLOCK_ROWS, 1), dtype=ct.float32)

    for k in range(num_col_tiles):
        x = ct.load(inp, (pid, k), shape=(BLOCK_ROWS, BLOCK_COLS))
        # Subtract max (stability trick) and exponentiate
        x = ct.sub(x, running_max)
        x = ct.exp(x)
        # Accumulate row-wise sum
        running_sum = ct.add(running_sum, ct.sum(x, axis=1, keepdims=True))
        # Store the intermediate exp(x) values back out
        ct.store(out, (pid, k), x)

    # ---------- STEP 3: Normalize (divide by sum) ----------
    for k in range(num_col_tiles):
        x = ct.load(out, (pid, k), shape=(BLOCK_ROWS, BLOCK_COLS))
        x = ct.truediv(x, running_sum)
        ct.store(out, (pid, k), x)


def main():
    print(f"Computing row-wise softmax on {ROWS}x{COLS} matrix...")

    x = cp.random.randn(ROWS, COLS).astype(cp.float32)
    y = cp.zeros((ROWS, COLS), dtype=cp.float32)

    # One block per 64 rows
    grid = (ct.cdiv(ROWS, BLOCK_ROWS), 1, 1)
    print(f"Grid: {grid} | Tile: {BLOCK_ROWS}x{BLOCK_COLS}")

    stream = cp.cuda.get_current_stream()
    ct.launch(stream, grid, softmax_kernel, (x, y))
    stream.synchronize()

    expected = cp.exp(x - x.max(axis=1, keepdims=True))
    expected = expected / expected.sum(axis=1, keepdims=True)

    cp.testing.assert_allclose(y, expected, atol=1e-5, rtol=1e-4)
    print("SUCCESS! cuTile Softmax matched NumPy result.")


if __name__ == "__main__":
    main()
