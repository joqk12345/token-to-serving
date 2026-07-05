# Part II Figure Specs

Scope: Chapters 4-6

These specs cover the GPU execution and attention-acceleration chapters. The goal is to make memory movement, execution hierarchy, and algorithmic scheduling visible.

## Chapter 4

### `fig-04-host-device-lifecycle`

- Chapter: 4
- Purpose: Show the lifecycle of a minimal CUDA program.
- Core message: CPU host code allocates device memory, copies data, launches a kernel, copies results back, and frees memory.
- Visual form: Sequence diagram.
- Layout: left lane for host memory/CPU, right lane for GPU/device memory, arrows for `cudaMalloc`, `cudaMemcpy`, kernel launch, result copy, `cudaFree`.
- Labels: host, device, device memory, host-to-device copy, kernel launch, device-to-host copy.
- Source anchors: `llmsys-03-cuda-memory-lifecycle`, `llmsys-02-cuda-programming-model`
- Production note: Keep API names visible but do not turn the figure into CUDA documentation.

### `fig-04-grid-block-thread`

- Chapter: 4
- Purpose: Show CUDA launch hierarchy and global thread indexing.
- Core message: A grid contains blocks; blocks contain threads; each thread computes a global index.
- Visual form: Hierarchy plus formula callout.
- Layout: large grid rectangle with several block rectangles; zoom into one block with thread cells; callout with `i = blockDim.x * blockIdx.x + threadIdx.x`.
- Labels: grid, block, thread, `blockIdx`, `blockDim`, `threadIdx`, global index.
- Source anchors: `llmsys-02-warp-execution`, `llmsys-03-thread-indexing`, `llmsys-03-vector-addition`
- Production note: Use one-dimensional indexing in the main figure. Put 2D indexing in prose or a small inset if needed.

### `fig-04-sm-warp-scheduling`

- Chapter: 4
- Purpose: Show blocks assigned to SMs and threads executed in warps.
- Core message: GPU hardware schedules structured groups of threads, not arbitrary independent scalar work.
- Visual form: Execution schematic.
- Layout: grid blocks on the left, arrows to several SMs on the right, one SM expanded into warps.
- Labels: SM, thread block, warp, scheduler, registers, shared memory.
- Source anchors: `llmsys-02-gpu-architecture`, `llmsys-02-warp-execution`
- Production note: Keep NVIDIA-specific wording clear. Do not imply every accelerator uses CUDA warp semantics.

### `fig-04-memory-paths`

- Chapter: 4
- Purpose: Show the memory paths relevant to GPU programs.
- Core message: Performance depends on where data lives and how often it moves.
- Visual form: Memory hierarchy diagram.
- Layout: host memory -> interconnect -> GPU global memory/HBM -> L2 -> shared memory -> registers.
- Labels: host memory, PCIe/NVLink, GPU memory, L2, shared memory, registers, thread, block.
- Source anchors: `llmsys-02-cpu-gpu-data-movement`, `llmsys-02-gpu-server-components`, `llmsys-02-gpu-architecture`
- Production note: Avoid exact bandwidth numbers unless the caption names the specific hardware source.

## Chapter 5

### `fig-05-memory-bound-intuition`

- Chapter: 5
- Purpose: Explain memory-bound versus compute-bound kernels.
- Core message: A kernel can have enough arithmetic units available and still wait on memory traffic.
- Visual form: Two-lane comparison.
- Layout: left lane shows low arithmetic per byte with idle compute; right lane shows high reuse with busier compute.
- Labels: arithmetic, bytes moved, memory-bound, compute-bound, arithmetic intensity.
- Source anchors: `llmsys-04-memory-access-efficiency`, `llmsys-04-naive-matmul-intensity`
- Production note: This is conceptual, not a formal roofline chart.

### `fig-05-naive-vs-tiled-matmul`

- Chapter: 5
- Purpose: Compare repeated global-memory reads with shared-memory tile reuse.
- Core message: Tiling changes where operands live while they are reused.
- Visual form: Side-by-side dataflow.
- Layout: naive side shows each output reading rows/columns from global memory; tiled side shows `A` and `B` tiles loaded into shared memory, reused by a block, then accumulated.
- Labels: global memory, shared memory, tile, partial sum, `__syncthreads`.
- Source anchors: `llmsys-04-tiling-shared-memory`
- Production note: Keep matrix sizes tiny and symbolic.

### `fig-05-coalescing-transpose`

- Chapter: 5
- Purpose: Show why transpose can be correct but memory-inefficient.
- Core message: Adjacent warp lanes should access adjacent addresses when possible.
- Visual form: Warp access diagram.
- Layout: row-major matrix, one row-wise access path marked coalesced, one column/stride path marked uncoalesced, shared-memory tile in the middle.
- Labels: warp lanes, adjacent addresses, strided addresses, shared-memory tile, transpose.
- Source anchors: `llmsys-04-coalesced-access`, `llmsys-04-bank-conflict`
- Production note: Treat bank conflicts as an inset or sidebar, not the main visual.

### `fig-05-transformer-operator-map`

- Chapter: 5
- Purpose: Map Transformer block operations to kernel categories.
- Core message: Transformer performance includes GEMM, elementwise, reduction, and memory-management work.
- Visual form: Annotated Transformer block.
- Layout: attention and FFN blocks with color-coded tags for GEMM, elementwise, reduction, memory movement.
- Labels: QKV projection, softmax, LayerNorm, dropout, residual, FFN, GEMM, reduction, fusion candidate.
- Source anchors: `llmsys-10-transformer-operator-stack`
- Production note: Do not make the diagram look like a full architecture diagram from Chapter 2; focus on operator classes.

### `fig-05-kernel-fusion`

- Chapter: 5
- Purpose: Show how fusion reduces launch and intermediate-memory boundaries.
- Core message: Fusing simple operations can avoid writing and rereading intermediates.
- Visual form: Before/after trace.
- Layout: before: kernel 1 writes `C`, kernel 2 reads `C`; after: one fused kernel writes `E`.
- Labels: kernel launch, intermediate tensor, global memory write, global memory read, fused kernel.
- Source anchors: `llmsys-10-kernel-fusion`, `llmsys-10-fused-embedding`
- Production note: Keep the example `E = A + B + D`, then mention embedding fusion in caption if needed.

## Chapter 6

### `fig-06-standard-attention-hbm`

- Chapter: 6
- Purpose: Show standard attention materializing `S` and `P`.
- Core message: Standard attention writes large `N x N` intermediates to HBM.
- Visual form: Dataflow diagram.
- Layout: `Q,K -> S = QK^T -> P = softmax(S) -> O = PV`, with `S` and `P` shown as large HBM-resident matrices.
- Labels: `Q`, `K`, `V`, `S`, `P`, `O`, HBM, `N x N`.
- Source anchors: `dao-2022-standard-attention-materialization`
- Production note: Make the `N x N` size visually larger than `N x d`.

### `fig-06-flashattention-tiling`

- Chapter: 6
- Purpose: Show FlashAttention moving Q/K/V blocks through SRAM.
- Core message: FlashAttention avoids materializing the full attention matrix by computing attention blockwise on chip.
- Visual form: Blocked matrix diagram.
- Layout: HBM contains full Q/K/V; SRAM holds one Q block and one K/V block; output block is updated and written back.
- Labels: HBM, SRAM, Q block, K block, V block, output block, tile loop.
- Source anchors: `dao-2022-flashattention-tiling`, `dao-2022-flashattention-io-awareness`
- Production note: Do not imply the algorithm changes dense attention into sparse attention.

### `fig-06-online-softmax`

- Chapter: 6
- Purpose: Show how blockwise softmax remains exact through running statistics.
- Core message: Running max and normalization terms let local score blocks combine into a global row-wise softmax.
- Visual form: Step diagram with formulas.
- Layout: block 1 has `(m_old, l_old, O_old)`, block 2 has `(m_block, l_block, O_block)`, merge step produces `(m_new, l_new, O_new)`.
- Labels: running max, normalization sum, rescale old output, add block output, exact softmax.
- Source anchors: `dao-2022-online-softmax`
- Production note: Keep formulas aligned with Chapter 6 prose and verify notation before final drawing.

### `fig-06-forward-backward-memory`

- Chapter: 6
- Purpose: Compare standard backward storage with FlashAttention recomputation.
- Core message: FlashAttention stores smaller statistics and recomputes blocks instead of storing the full attention matrix.
- Visual form: Before/after memory diagram.
- Layout: standard side stores `P: N x N`; FlashAttention side stores `O`, running statistics, and reuses Q/K/V blocks during backward.
- Labels: attention matrix, normalization stats, recompute, backward, HBM traffic.
- Source anchors: `dao-2022-backward-recomputation`
- Production note: Do not say recomputation is universally faster; phrase as a memory-traffic tradeoff.

### `fig-06-modern-hardware-sidebar`

- Chapter: 6
- Purpose: Summarize FA2/FA3/FA4 as evolution of the same co-design pattern.
- Core message: Later FlashAttention versions respond to new active bottlenecks: work partitioning, asynchrony/FP8, and asymmetric hardware scaling.
- Visual form: Three-item sidebar timeline.
- Layout: FA2 -> FA3 -> FA4, each with one bottleneck and one design response.
- Labels: work partitioning, parallelism, Hopper asynchrony, FP8, Blackwell asymmetric scaling, pipelining.
- Source anchors: `dao-2023-flashattention-2`, `shah-2024-flashattention-3`, `zadouri-2026-flashattention-4`
- Production note: Do not include speedup numbers in the figure unless the caption gives benchmark conditions.

## Cross-Chapter Notes

- Use consistent HBM/SRAM/shared-memory terminology.
- When a figure says SRAM in Chapter 6, make clear it follows FlashAttention paper terminology.
- Avoid generic GPU photos or decorative chip art.
- Keep formulas in figures only when they clarify a memory/scheduling point.
- Captions must state when a statement is CUDA/NVIDIA-specific.

Owner: Book Architect  
Purpose: Part II figure planning  
Evidence grade: A for structural decisions; figure content must be checked against source cards before drawing  
Assumptions: Figures are technical book diagrams, not lecture slides  
Open questions: Whether Chapter 6 final figure includes full online softmax formulas or an abbreviated merge rule  
Handoff: Writer / illustrator for first draft diagrams
