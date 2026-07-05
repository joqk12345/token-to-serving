# Red Team: Chapter 5 — Kernels, Memory, and Transformer Blocks

Date: 2026-07-05  
Reviewer: Codex  
Chapter: `book/chapters/05-kernels-memory-transformer-blocks.md`

## Verdict

Cleared for ready promotion.

The chapter explains performance mechanisms without overclaiming. It keeps simplified arithmetic-intensity reasoning marked as intuition, avoids benchmark numbers, and makes fusion/tiling/mixed precision conditional rather than universal.

## Attacks and Outcomes

### Attack 1: The `0.25 FLOP/B` example may look like a complete roofline model.

Outcome: Addressed.

The draft explicitly says the number is simplified intuition and names omitted factors such as caches, tensor cores, scheduling effects, and library kernels.

### Attack 2: Tiling and fusion may sound always beneficial.

Outcome: Addressed.

The draft states that fusion can increase register use, reduce occupancy, and reduce shape reuse. Tiling is presented as a tradeoff involving coordination and shared-memory capacity.

### Attack 3: LayerNorm rewrite may imply unchecked numerical claims.

Outcome: Addressed.

The draft keeps the LayerNorm point conceptual: algebraic form affects synchronization. It says final formula and numerical details need careful review if expanded.

### Attack 4: cuBLAS mention may hide custom-kernel boundaries.

Outcome: Addressed.

The chapter uses cuBLAS to show that standard dense GEMM often belongs to mature libraries, then explains why Transformer blocks still need custom/fused paths for surrounding operations.

## Required Fixes

None.

## Recommended Non-Blocking Follow-Up

- Add primary LightSeq/LightSeq2 cards only if future revisions include implementation-specific claims or speedups.
- Add figure artwork for naive versus tiled matmul, coalescing/transpose, operator stack, and kernel fusion.

Owner: Red Team Reviewer  
Purpose: Chapter 5 adversarial review  
Evidence grade: A for reviewed source map; no benchmark numbers used  
Assumptions: Red-team review evaluates mechanism-level draft correctness, not production kernel completeness  
Open questions: None blocking  
Handoff: Principal author may promote Chapter 5 to ready
