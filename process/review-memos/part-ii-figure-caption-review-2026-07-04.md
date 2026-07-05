# Part II Figure Caption Review

Date: 2026-07-04

Scope: `book/figures/part-ii-figure-specs.md`

## Verdict

Part II figure specs are ready for first-pass diagram production. The captions below are source-aligned and avoid benchmark, hardware-limit, or universal-speedup claims.

Do not add numeric bandwidth, utilization, or speedup values to captions unless a figure-specific source card and benchmark condition are added.

## Chapter 4 Captions

### `fig-04-host-device-lifecycle`

Caption: A minimal CUDA program is split between host orchestration and device execution: the CPU allocates device memory, copies inputs to the GPU, launches a kernel, copies results back, and frees device memory.

Source anchors: `nvidia-2026-cuda-programming-model`, `llmsys-03-cuda-memory-lifecycle`

Risk: Keep this as a conceptual lifecycle. Do not imply every optimized CUDA program performs blocking copies in exactly this order.

### `fig-04-grid-block-thread`

Caption: A CUDA kernel launch creates a grid of thread blocks. Each thread computes its identity from block and thread indices, then uses that identity to choose the data element it owns.

Source anchors: `nvidia-2026-cuda-programming-model`, `llmsys-03-thread-indexing`

Risk: Keep the main formula one-dimensional. Two-dimensional indexing can appear in prose or an inset.

### `fig-04-sm-warp-scheduling`

Caption: Thread blocks are scheduled onto streaming multiprocessors, and threads within a block execute in warps under CUDA's SIMT model.

Source anchors: `nvidia-2026-cuda-programming-model`, `llmsys-02-warp-execution`

Risk: Say CUDA/NVIDIA in the caption or nearby text. Do not generalize warp semantics to all accelerators.

### `fig-04-memory-paths`

Caption: GPU performance depends on where data lives: host memory, interconnects, GPU global memory, caches, shared memory, and registers all impose different movement costs and visibility rules.

Source anchors: `nvidia-2026-cuda-programming-model`, `llmsys-02-cpu-gpu-data-movement`, `llmsys-02-gpu-architecture`

Risk: Avoid exact bandwidth numbers in the figure.

## Chapter 5 Captions

### `fig-05-memory-bound-intuition`

Caption: A kernel with little arithmetic per byte moved can be memory-bound: the arithmetic units may wait for data even when peak compute throughput is high.

Source anchors: `llmsys-04-memory-access-efficiency`

Risk: This is not a formal roofline model. Keep the graphic qualitative.

### `fig-05-naive-vs-tiled-matmul`

Caption: Naive matrix multiplication repeatedly reads operands from global memory, while a tiled kernel loads smaller blocks into shared memory so threads in a block can reuse them.

Source anchors: `llmsys-04-naive-matmul-intensity`, `llmsys-04-tiling-shared-memory`

Risk: Do not imply this is how production GEMM kernels are fully implemented; this is the teaching pattern.

### `fig-05-coalescing-transpose`

Caption: Matrix transpose exposes the difference between correct indexing and efficient memory access: coalesced warp accesses align neighboring threads with neighboring addresses, while strided accesses can require more transactions.

Source anchors: `llmsys-04-coalesced-access`, `llmsys-04-bank-conflict`

Risk: Bank-conflict behavior is architecture-specific. Keep bank conflict as an inset or note.

### `fig-05-transformer-operator-map`

Caption: A Transformer block is a mixed workload: GEMM-heavy projections sit beside elementwise operations, reductions such as softmax and LayerNorm, and memory-management choices.

Source anchors: `llmsys-10-transformer-operator-stack`

Risk: This is an operator taxonomy, not a full Transformer architecture diagram.

### `fig-05-kernel-fusion`

Caption: Kernel fusion removes unnecessary intermediate tensor boundaries: instead of writing `C` to global memory and reading it back, a fused kernel can compute the final result in one pass.

Source anchors: `llmsys-10-kernel-fusion`, `llmsys-10-fused-embedding`

Risk: Avoid claiming fusion is categorically faster. Register pressure, shape variation, and occupancy can change the result.

## Chapter 6 Captions

### `fig-06-standard-attention-hbm`

Caption: Standard attention commonly materializes the score matrix `S = QK^T` and probability matrix `P = softmax(S)` as `N x N` intermediates in HBM before computing `O = PV`.

Source anchors: `dao-2022-standard-attention-materialization`

Risk: The word "commonly" avoids overclaiming all implementations.

### `fig-06-flashattention-tiling`

Caption: FlashAttention computes exact dense attention block by block, moving Q/K/V tiles through on-chip SRAM and avoiding materialization of the full `N x N` attention matrix in HBM.

Source anchors: `dao-2022-flashattention-tiling`, `dao-2022-flashattention-io-awareness`

Risk: Make clear this is not sparse or approximate attention.

### `fig-06-online-softmax`

Caption: Online softmax keeps a running maximum and normalization sum so each new score block can be rescaled into the same row-wise denominator as the previous blocks.

Source anchors: `dao-2022-online-softmax`, `chapter-6-online-softmax-formula-check-2026-07-04`

Risk: Use the same `m_old`, `l_old`, `O_old`, `m_block`, `l_block`, `O_block` notation as Chapter 6. State that `O_old` is normalized and `O_block` is unnormalized.

### `fig-06-forward-backward-memory`

Caption: FlashAttention trades storage for recomputation in backward: instead of storing the full attention-probability matrix, it stores smaller normalization statistics and recomputes attention blocks when gradients are needed.

Source anchors: `dao-2022-backward-recomputation`

Risk: Avoid claiming recomputation is categorically faster. The benefit depends on memory traffic and hardware balance.

### `fig-06-modern-hardware-sidebar`

Caption: Later FlashAttention versions continue the same co-design pattern: FA2 improves work partitioning, FA3 targets Hopper asynchrony and low precision, and FA4 responds to Blackwell-era asymmetric hardware scaling.

Source anchors: `dao-2023-flashattention-2`, `shah-2024-flashattention-3`, `zadouri-2026-flashattention-4`

Risk: Do not include speedup numbers in this sidebar without benchmark conditions.

## Remaining Review Notes

- Captions are cleared for draft diagrams.
- Diagram artwork must still be checked for terminology drift, especially `shared memory` versus `SRAM`.
- If the final book uses rendered formulas inside `fig-06-online-softmax`, rerun formula review on the exact artwork text.

Owner: Technical Reviewer  
Purpose: Part II figure-caption review  
Evidence grade: A for review process; captions inherit their source anchors  
Open questions: Whether final artwork includes formulas or keeps formula detail in prose  
Handoff: Illustrator / diagram producer
