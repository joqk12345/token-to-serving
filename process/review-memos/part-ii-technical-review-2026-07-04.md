# Part II Technical Review

Date: 2026-07-04

Scope: Chapters 4-6

- Chapter 4: `04-gpu-programming-model`
- Chapter 5: `05-kernels-memory-transformer-blocks`
- Chapter 6: `06-flashattention-transformer-acceleration`

## Verdict

Part II is ready to continue into Chapter 7, but not ready for technical-review clearance.

The chapter sequence is sound:

```text
GPU programming model
-> memory and kernel performance patterns
-> FlashAttention as IO-aware co-design case study
```

The current drafts are good enough as book-shaping drafts. The remaining issues are technical evidence and precision risks rather than chapter-order problems.

## Chapter 4 Review

### What Works

- The chapter introduces host/device, kernel launch, grid/block/thread, SM/warp, and memory hierarchy in the right order.
- The code snippets are minimal and support the prose rather than taking over the chapter.
- The chapter correctly avoids production-kernel tuning, leaving tiling and coalescing for Chapter 5.

### Risks

- CUDA API details are supported by course cards only. Before final publication, add official NVIDIA CUDA Programming Guide source cards for:
  - function qualifiers: `__global__`, `__device__`, `__host__`;
  - kernel launch configuration;
  - thread/block/grid built-ins;
  - memory spaces;
  - warp size and SIMT wording.
- Hardware-specific statements should remain NVIDIA-qualified. The chapter currently does this in the key warp passage, but final review should scan for accidental generalization.
- The matrix indexing snippets are explanatory, not tested examples. If runnable code is later added, it should live in an appendix or examples directory with build instructions.

## Chapter 5 Review

### What Works

- The progression from arithmetic intensity to tiling, coalescing, cuBLAS, fusion, reductions, mixed precision, and memory reuse is coherent.
- The chapter does not overuse LightSeq benchmark numbers, which keeps the evidence risk controlled.
- The text gives readers a reusable resource-diagnosis frame rather than a bag of tricks.

### Risks

- The `0.25 FLOP/B` matrix multiplication example is useful but simplified. It should remain explicitly framed as intuition unless a more formal roofline derivation is added.
- LayerNorm and softmax reduction rewrites are conceptually correct at the systems level, but formulas are deferred. If formulas enter the final text, they need a math check.
- cuBLAS is currently supported by course cards. Add an NVIDIA cuBLAS documentation card before final API-level publication.
- LightSeq/LightSeq2 are used only as conceptual support now. If the chapter later includes named speedups or implementation claims, add primary paper source cards.

## Chapter 6 Review

### What Works

- FlashAttention is anchored to the primary paper `2205.14135.pdf`, which is the right evidence level.
- The chapter avoids universal speedup claims and keeps benchmark language conditional.
- The online softmax explanation is in prose, which is appropriate until formulas are reviewed.
- FA3/FA4 material is correctly kept as sidebar direction rather than main-line evidence.

### Risks

- The online softmax section needs formula review before publication. The prose invariant is accurate enough for draft, but the final book may need one compact derivation.
- IO-complexity wording is intentionally high-level. If exact asymptotic claims are added, check them against the paper's theorem statements.
- Later FlashAttention generations now have primary source cards for sidebar-level claims. Benchmark or detailed hardware claims still need benchmark conditions in the prose.
- Decode attention is introduced briefly. That is appropriate here, but Chapter 12-13 must define prefill/decode and KV-cache mechanics before relying on it.

## Cross-Chapter Issues

### Evidence Gaps

Add source cards before final clearance:

- NVIDIA CUDA Programming Guide
- NVIDIA cuBLAS documentation
- Programming Massively Parallel Processors, Chapters 5-6, if used for tiling/coalescing exposition
- LightSeq and LightSeq2 papers, if named implementation claims remain
- FlashAttention-2/3/4 primary papers are now covered by source cards; add benchmark-condition cards only if numeric claims enter the text.

### Terminology Checks

- Use `global memory` for GPU device global memory.
- Use `HBM` when specifically discussing high-bandwidth memory in FlashAttention.
- Use `SRAM` for on-chip memory in FlashAttention paper context.
- Use `shared memory` for CUDA programming sections.
- Avoid silently treating `shared memory` and `SRAM` as identical terms; explain the relationship when needed.

### Figure Needs

Part II needs figure specs next:

- Chapter 4: host/device lifecycle, grid/block/thread, SM/warp scheduling, memory hierarchy.
- Chapter 5: memory-bound intuition, naive vs tiled matmul, coalescing/transpose, Transformer operator map, kernel fusion.
- Chapter 6: standard attention materialization, FlashAttention tiling, online softmax rescaling, backward recomputation.

## Clearance

Current status: `review-gated`.

The drafts may be used as the base for Chapter 7. Do not mark Part II `technical-review cleared` until:

- official CUDA/cuBLAS source cards are added;
- Chapter 6 online softmax formulas are checked;
- Part II figure specs are reviewed against captions;
- hardware-specific claims are scanned again after figure captions are written.

Update 2026-07-04:

- NVIDIA CUDA Programming Guide and cuBLAS documentation cards have been added.
- Chapter 6 online softmax formulas have been added and checked for draft use.
- Part II figure specs have been added. Captions still need source-card review before diagram production.

Update 2026-07-04, figure captions:

- Part II figure captions have been reviewed against source cards.
- Captions are cleared for draft diagram production.
- Final rendered artwork still needs terminology review, especially `shared memory` versus `SRAM`.

Owner: Technical Reviewer  
Purpose: Part II technical review after Chapters 4-6 drafts  
Evidence grade: A for review process; individual claims retain their source-card grades  
Open questions: Whether to add runnable CUDA examples before or after Chapter 7  
Handoff: Book Architect for Part II figure specs or Chapter 7 source extraction
