# Technical Review: Chapter 4 — Inside the GPU Programming Model

Date: 2026-07-05  
Reviewer: Codex  
Chapter: `book/chapters/04-gpu-programming-model.md`

## Verdict

Cleared for red-team review.

The chapter is technically coherent at draft level. It introduces CUDA host/device structure, memory lifecycle, grid/block/thread indexing, SMs, warps, SIMT execution, memory hierarchy, and the difference between correctness and performance. The earlier Part II gate on official CUDA evidence is resolved by the added NVIDIA CUDA documentation source cards.

## Checks

### Scope and Boundary

- The chapter correctly teaches programming model vocabulary rather than production GEMM optimization.
- Tiling, coalescing, and Transformer-kernel performance are deferred to Chapter 5.
- FlashAttention is only used as a forward pointer to Chapter 6.

### CUDA Programming Model

- Host/device separation is explained correctly.
- The memory lifecycle uses `cudaMalloc`, `cudaMemcpy`, kernel launch, and `cudaFree` as a minimal conceptual workflow.
- Grid, block, and thread hierarchy is explained before examples.
- Thread-index calculations for vector and matrix examples are correct as explanatory snippets.
- Bounds checks are included for rounded-up launches.

### Hardware Caveats

- SM, warp, and SIMT discussion is qualified as NVIDIA/CUDA-centered.
- Warp size is stated in CUDA context.
- The chapter does not imply all accelerators use CUDA terminology.

### Memory Hierarchy

- Registers, shared memory, global memory, cache, host memory, and interconnect paths are described as part of one performance story.
- The chapter does not equate CUDA shared memory with FlashAttention paper SRAM terminology.
- It avoids latency/bandwidth numbers without hardware conditions.

### Citation and Evidence

- Citation marker escaping has been fixed.
- All cited source-ledger cards exist.
- Official NVIDIA CUDA documentation cards are available for final API-level support.

## Risks Remaining

| Risk | Blocking? | Notes |
|---|---:|---|
| Code snippets are not runnable examples. | No | The chapter states they are minimal explanatory snippets. |
| CUDA-specific vocabulary may need final cross-accelerator caveat pass. | No | Current wording is NVIDIA/CUDA-qualified. |
| No figures yet rendered. | No | Figure specs/captions were reviewed separately for draft production. |

## Required Fixes

None.

## Red-Team Prompts

- Does the chapter make CUDA concepts sound universal across all accelerators?
- Do code snippets look production-ready when they are only explanatory?
- Does memory hierarchy language confuse shared memory, global memory, HBM, and host memory?
- Does the chapter overpromise that exposing threads is enough for performance?

Owner: Technical Reviewer  
Purpose: Chapter 4 technical review  
Evidence grade: A for course lectures and official CUDA documentation cards; no benchmark numbers used  
Assumptions: Review evaluates draft-level correctness, not runnable code packaging  
Open questions: Whether runnable CUDA examples belong in an appendix or examples directory  
Handoff: Red Team reviewer for adversarial critique
