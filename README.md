### Mental Model: How GPU Matmul Actually Works

You have a massive matrix multiply to do. One CPU core would take forever.
A GPU has thousands of tiny cores but they share limited fast memory. The
whole game is about splitting the work into pieces small enough to fit in
fast memory, and keeping all the cores busy.

### HBM (High Bandwidth Memory)

This is the GPU's main memory. Big (like 24GB to 80GB) but relatively slow
to access. Think of it as a warehouse. Your full matrices A, B, C live here.

### SM (Streaming Multiprocessor)

The actual compute units on the GPU. A modern GPU has maybe 80 to 130 SMs.
Each SM can run one or more blocks at a time. Think of SMs as workers in a
factory. They each have their own small workspace.

### SRAM (Shared Memory / Registers)

This is the tiny fast memory sitting right next to the SM. Maybe 128KB to
228KB per SM. Extremely fast to read from, but tiny compared to HBM. Think
of it as the worker's desk. Only what's on the desk can be worked on quickly.

### TMA (Tensor Memory Accelerator)

A hardware unit (on newer GPUs like H100) that handles moving rectangular
tiles of data between HBM and SRAM without bothering the compute cores. Think
of it as a robot arm that fetches boxes from the warehouse and puts them on
the worker's desk while the worker keeps doing math.


### Grid

When you launch a kernel, you launch a grid. A grid is just the total number
of blocks you want to run. For our matmul:

    grid = (4, 4, 1) means 16 blocks total arranged in a 4x4 pattern

Each block gets a coordinate. Block(2,1) knows it handles a specific chunk
of the output. The grid answers "how do we split the whole job."

### Block

One block is one chunk of work assigned to one SM. A block computes one piece
of the output matrix. In our case, each block computes a 64x64 piece of C.

The block runs on a single SM. It has access to that SM's shared memory.
Multiple blocks can share an SM if they fit, but one block never spans two SMs.

### Tile

A tile is the actual chunk of data a block loads at one time. The block can't
load everything it needs at once (the K dimension is too long), so it loads
in small rectangular tiles, does partial math, and loads the next tile.

In our code, each block loads:
    a 64x32 tile from A
    a 32x64 tile from B
    multiplies them (adds to accumulator)
    repeats 8 times (because 256/32 = 8)

The tile answers "how much fits on the desk at once."


## How Data Flows: HBM to SRAM and Back

Here's what actually happens for one block, step by step:

    1. Block starts. Accumulator lives in registers (fastest memory, on the SM).

    2. ct.load(A, ...) triggers a memory request.
       Data travels: HBM -> L2 cache -> SRAM (shared memory or registers)
       On newer GPUs with TMA, this happens asynchronously. The hardware
       fetches the tile while the SM does other work.

    3. ct.load(B, ...) same thing. Another tile pulled from HBM to SRAM.

    4. ct.mma() the actual math. Both tiles are now in fast memory. The SM's
       tensor cores multiply them and add to the accumulator. This is fast
       because everything is local.

    5. Repeat steps 2 to 4 for each K tile (8 times in our case).

    6. ct.store(C, ...) writes the final 64x64 accumulator back.
       Data travels: Registers/SRAM -> L2 cache -> HBM

The whole point of tiling is to minimize how often we go to HBM. Each element
of A gets reused across the N dimension, each element of B gets reused across
the M dimension. By loading tiles into fast memory, we do many multiplies per
byte loaded.


### Why This Matters: The Numbers

HBM bandwidth on an A100: ~2 TB/s
SRAM bandwidth on an A100: ~19 TB/s (per SM, aggregated)
Tensor core throughput: ~312 TFLOPS (FP16)

If you load from HBM every time you need a number, the compute cores sit idle
waiting for data. Tiling ensures you load once, use many times. This is the
entire reason for the complexity of GPU kernels.


### Putting It All Together

    Matrix (lives in HBM, the warehouse)
        |
        | ct.launch: divide work into a grid of blocks
        v
    Grid of Blocks (each assigned to an SM, a worker)
        |
        | ct.load: pull tiles from HBM to SRAM (the desk)
        v
    Tiles in SRAM (small, fast, close to compute)
        |
        | ct.mma: tensor cores do the actual math
        v
    Accumulator in Registers (fastest, per thread)
        |
        | ct.store: write finished result back to HBM
        v
    Output Matrix (back in HBM, ready to use)


Grid splits the output, blocks own the splits, tiles are what fit in fast
memory at once, and the whole dance is about not waiting on slow memory.
