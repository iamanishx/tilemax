# `matmul_cute.py` — Visual Walkthrough (Mermaid)

Diagrams of how the SIMT GEMM in `matmul_cute.py` actually executes, from host
setup down to per-thread work. Pair with `learning.md`.

Problem: `C[256,256] = A[256,256] @ B[256,256]`, tiles `128×128×8`, `256` threads/CTA.

---

## 1. End-to-end flow (host → device → verify)

```mermaid
flowchart TD
    subgraph HOST["main()  —  host / Python"]
        A1["NumPy randn A, B  (libcurand-free)"]
        A2["cp.asarray → GPU buffers"]
        A3["from_dlpack(a), from_dlpack(b.T), from_dlpack(c)<br/>(views: pointer + layout, no copy)"]
        A4["build cuda.CUstream from cupy stream"]
        A5["gemm(mA, mB, mC, stream)"]
    end

    subgraph JIT["SimpleGemm.__call__  —  @cute.jit (traced → PTX/SASS)"]
        B1["op = MmaUniversalOp(Float32)  (SIMT FMA atom)"]
        B2["atoms_layout = (16,16,1)  → TV layout / 256 threads"]
        B3["tiled_mma = make_tiled_mma(op, atoms_layout)"]
        B4["grid = (M/128, N/128, 1) = (2,2,1)"]
        B5["kernel(...).launch(grid=2x2, block=256, stream)"]
    end

    subgraph DEV["SimpleGemm.kernel  —  @cute.kernel (runs on GPU)"]
        C1["each CTA computes one 128x128 tile of C"]
    end

    subgraph VERIFY["back on host"]
        D1["stream.synchronize()"]
        D2["expected = a_np @ b_np"]
        D3["assert_allclose → SUCCESS"]
    end

    A1 --> A2 --> A3 --> A4 --> A5 --> B1
    B1 --> B2 --> B3 --> B4 --> B5 --> C1
    C1 --> D1 --> D2 --> D3
```

---

## 2. Grid decomposition — which CTA owns which part of C

`grid = (2, 2)` → 4 CTAs, each owns a `128×128` tile of the `256×256` output.

```mermaid
flowchart LR
    subgraph C["C  (256 x 256)"]
        direction TB
        subgraph R0[" "]
            T00["CTA (0,0)<br/>rows 0-127<br/>cols 0-127"]
            T01["CTA (0,1)<br/>rows 0-127<br/>cols 128-255"]
        end
        subgraph R1[" "]
            T10["CTA (1,0)<br/>rows 128-255<br/>cols 0-127"]
            T11["CTA (1,1)<br/>rows 128-255<br/>cols 128-255"]
        end
    end
    note["bidx = row-tile index&#10;bidy = col-tile index"] -.-> C
```

Each CTA: `bidx, bidy = block_idx()` → it knows exactly its 128×128 region.

---

## 3. Tiling the global tensors with `local_tile`

For CTA `(bidx, bidy)`, the three operands are sliced into per-CTA tiles.
Note B is passed as `(N, K)` (the `b.T` view) so its K mode is trailing.

```mermaid
flowchart TD
    subgraph GLOBAL["Global memory (full matrices)"]
        mA["mA  (M=256, K=256)"]
        mB["mB  (N=256, K=256)  ← b.T view"]
        mC["mC  (M=256, N=256)"]
    end

    subgraph TILES["This CTA's tiles (bidx, bidy)"]
        gA["gA = local_tile(mA,(128,8),(bidx,None))<br/>shape (128, 8, k_tiles=32)"]
        gB["gB = local_tile(mB,(128,8),(bidy,None))<br/>shape (128, 8, k_tiles=32)"]
        gC["gC = local_tile(mC,(128,128),(bidx,bidy))<br/>shape (128, 128)"]
    end

    mA -->|"pick row-band bidx, keep all K"| gA
    mB -->|"pick row-band bidy, keep all K"| gB
    mC -->|"pick the (bidx,bidy) tile"| gC
```

`k_tiles = 256 / 8 = 32` → the third mode is the K-loop iteration count.

---

## 4. TV layout — how 256 threads partition the 128×128 tile

`atoms_layout = (16,16,1)` arranges threads as a 16×16 grid. Each thread owns an
`8×8` sub-block of the output tile (`128/16 = 8`).

```mermaid
flowchart TD
    TM["tiled_mma  (from op + 16x16 atoms_layout)"]
    SL["thr_mma = tiled_mma.get_slice(tidx)<br/>'I am thread tidx — give me MY view'"]
    PA["partition_A(gA[..,k]) → tCrA  (this thread's A elems)"]
    PB["partition_B(gB[..,k]) → tCrB  (this thread's B elems)"]
    PC["partition_C(gC)       → tCgC  (this thread's C elems, 8x8)"]
    FR["acc = make_fragment_C(tCgC)<br/>register accumulator, filled 0.0"]

    TM --> SL
    SL --> PA
    SL --> PB
    SL --> PC --> FR
```

> This TV layout is the answer to "which thread/warp does what" — change
> `atoms_layout` and you change the assignment. Warp `w` = threads `32w..32w+31`.

---

## 5. The K-loop — accumulate across the contraction dim

```mermaid
sequenceDiagram
    autonumber
    participant R as Registers (acc, 8x8 per thread)
    participant L as K-loop (k = 0..31)
    participant G as Global tiles gA, gB

    Note over R: acc.fill(0.0)
    loop for each k-tile (32 total)
        G->>L: tCrA = partition_A(gA[.., .., k])
        G->>L: tCrB = partition_B(gB[.., .., k])
        L->>R: cute.gemm(tiled_mma, acc, tCrA, tCrB, acc)
        Note over R: acc += A_tile · B_tile
    end
    Note over R: loop done → acc holds full dot products
```

Contraction: `acc[m,n] += Σ_k A[m,k] · B[n,k]`. Because B is `(N,K)`,
`B[n,k] = b[k,n]`, so this equals `(a @ b)[m,n]`. ✅

---

## 6. Memory journey of the data

```mermaid
flowchart LR
    HBM["Global / HBM<br/>A, B, C"] -->|"partition_A/B reads tiles"| REG1["Registers<br/>tCrA, tCrB"]
    REG1 -->|"cute.gemm (FMA)"| ACC["Registers<br/>acc (accumulator)"]
    ACC -->|"cute.copy(copy_atom, acc, tCgC)"| HBM2["Global / HBM<br/>C"]

    classDef slow fill:#fde,stroke:#b55;
    classDef fast fill:#dfe,stroke:#5b5;
    class HBM,HBM2 slow;
    class REG1,ACC fast;
```

> ⚠️ This basic version goes **global → registers → global** with no shared-
> memory stage, so A/B tiles get re-read from slow HBM. The fast version inserts
> an SMEM tile (`SmemAllocator` + `cp.async`) between HBM and registers, and uses
> tensor cores instead of FMA. See `learning.md` Stages 3–4.

---

## 7. Concept → code map

```mermaid
mindmap
  root((matmul_cute.py))
    Launch
      grid 2x2 = CTAs
      block 256 = threads
      cuda.CUstream
    Layouts
      from_dlpack = view
      local_tile = per-CTA tile
      b.T = B as N,K
    Threads
      atoms_layout 16x16 = TV layout
      get_slice = this thread
      partition_A/B/C
    Compute
      MmaUniversalOp = SIMT FMA
      make_fragment_C = acc in regs
      K-loop + cute.gemm
      cute.copy = write back
```

---

*Render these in any Mermaid-aware viewer (GitHub, VS Code Mermaid preview, Obsidian).*
