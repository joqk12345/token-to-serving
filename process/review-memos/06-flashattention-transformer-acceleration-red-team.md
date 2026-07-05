# Red Team: Chapter 6 — FlashAttention and Attention Acceleration

Date: 2026-07-05  
Reviewer: Codex  
Chapter: `book/chapters/06-flashattention-transformer-acceleration.md`

## Verdict

Cleared for ready promotion.

The chapter presents FlashAttention as IO-aware exact attention and avoids common overclaims. The online softmax formula is scoped as a one-row explanatory derivation, benchmark claims are conditioned rather than numeric, and later FlashAttention generations are kept as sidebar-level direction.

## Attacks and Outcomes

### Attack 1: The chapter may imply FlashAttention approximates attention.

Outcome: Addressed.

The draft explicitly states that FlashAttention computes exact attention, not sparse or approximate attention, while changing schedule and storage.

### Attack 2: The online softmax formula may look implementation-complete.

Outcome: Addressed.

The draft introduces it as a one-row explanation and states the normalized/unnormalized convention. The separate formula-check memo cleared it for draft use.

### Attack 3: The chapter may overclaim speedups.

Outcome: Addressed.

The draft uses no speedup numbers. It states that exact gains depend on hardware, shape, precision, masking, dropout, baseline, and forward/backward scope.

### Attack 4: SRAM/HBM terminology may conflict with CUDA shared/global memory terminology.

Outcome: Addressed.

The chapter uses HBM/SRAM in the FlashAttention paper context and does not collapse that language into Chapter 4's CUDA shared-memory explanation.

### Attack 5: Decode attention may steal serving-chapter scope.

Outcome: Addressed.

Decode attention is a short bridge only. KV cache, batching, and decode scheduling are left to the serving chapters.

## Required Fixes

None.

## Recommended Non-Blocking Follow-Up

- Final publication may replace the one-row formula with paper notation if the chapter becomes more mathematical.
- Add figure artwork for standard attention materialization, FlashAttention tiling, online softmax rescaling, and backward recomputation.

Owner: Red Team Reviewer  
Purpose: Chapter 6 adversarial review  
Evidence grade: A for reviewed source map and formula-check memo; no benchmark numbers used  
Assumptions: Red-team review evaluates readiness under current IO-aware exact-attention scope  
Open questions: None blocking  
Handoff: Principal author may promote Chapter 6 to ready
