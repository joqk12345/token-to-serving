---
status: brief
chapter: 6
slug: 06-flashattention-transformer-acceleration
title: FlashAttention and Attention Acceleration
primary_sources:
  - llmsys-21-FlashAttention_tridao2026.4-50476379a6127697ae7fbf974ad28348.pdf
  - 2205.14135.pdf
secondary_sources: []
reader_level: engineer or graduate student who has read Chapters 4-5
technical_depth: intermediate-to-advanced
---

# FlashAttention and Attention Acceleration

## Chapter Thesis

FlashAttention is the cleanest case study for LLM systems co-design: the attention algorithm changes so the GPU memory hierarchy sees less HBM traffic, while preserving exact attention.

## Reader Problem

The reader knows that attention computes `softmax(QK^T)V`, and now knows why memory movement matters. The missing step is understanding how an algorithm can be reorganized around memory hierarchy without changing the mathematical result.

## System Bottleneck

Primary bottlenecks: HBM traffic, materialization of `N x N` attention matrices, softmax normalization across blocks, backward memory footprint, and hardware-specific scheduling.

## Source Map

| Claim | Source card | Evidence grade | Notes |
|---|---|---|---|
| IO-awareness is the organizing principle of FlashAttention. | `dao-2022-flashattention-io-awareness` | A | Primary paper. |
| Standard attention materializes `S` and `P` in HBM. | `dao-2022-standard-attention-materialization` | A | Explain before algorithm. |
| FlashAttention tiles `Q`, `K`, and `V`. | `dao-2022-flashattention-tiling` | A | Main algorithm. |
| Online softmax statistics make blockwise exact attention possible. | `dao-2022-online-softmax` | A | Formula needs review before publication. |
| Backward uses recomputation instead of storing the full attention matrix. | `dao-2022-backward-recomputation` | A | Bridge to memory reuse. |
| The paper analyzes HBM accesses and IO complexity. | `dao-2022-io-complexity` | A | Keep high level unless formulas reviewed. |
| Benchmarks are conditional on specific settings. | `dao-2022-benchmark-context` | A | Avoid universal speedup claims. |
| FlashAttention-2 improves work partitioning and parallelism. | `dao-2023-flashattention-2` | A | Sidebar only unless expanded. |
| FlashAttention-3 targets Hopper asynchrony and low precision. | `shah-2024-flashattention-3` | A | Sidebar only unless expanded. |
| FlashAttention-4 targets Blackwell asymmetric hardware scaling. | `zadouri-2026-flashattention-4` | A | Sidebar only unless expanded. |
| Decode attention has short query length and long KV context. | `llmsys-21-decoding-attention-shape` | A | Bridge to serving chapters. |

## Explanation Arc

1. Reopen from Chapter 5: attention combines GEMM, softmax, masking, and memory movement.
2. Standard attention computes `S = QK^T`, `P = softmax(S)`, and `O = PV`.
3. The problem is not just `O(N^2)` arithmetic; it is materializing and rereading `N x N` intermediates in HBM.
4. FlashAttention blocks the computation so `Q`, `K`, and `V` tiles move through SRAM.
5. Online softmax rescaling preserves exactness across blocks.
6. Backward recomputes attention blocks rather than storing the full attention matrix.
7. The result is an algorithmic change driven by memory hierarchy.
8. Later FlashAttention versions show the pattern continuing as hardware changes.
9. Decode attention has a different shape and will return in inference chapters.

## Required Figures

| Figure ID | Purpose | Form | Source |
|---|---|---|---|
| `fig-06-standard-attention-hbm` | Show standard attention writing `S` and `P` to HBM. | Dataflow diagram | `dao-2022-standard-attention-materialization` |
| `fig-06-flashattention-tiling` | Show Q/K/V tiles moving between HBM and SRAM. | Blocked matrix diagram | `dao-2022-flashattention-tiling` |
| `fig-06-online-softmax` | Show blockwise max/sum update and output rescaling. | Step diagram | `dao-2022-online-softmax` |
| `fig-06-forward-backward-memory` | Compare storing attention matrix versus storing normalization stats and recomputing. | Before/after memory diagram | `dao-2022-backward-recomputation` |
| `fig-06-modern-hardware-sidebar` | Summarize FA3/FA4 as hardware-aware evolution. | Sidebar timeline | `llmsys-21-modern-hardware-attention` |

## Main Sections

### Attention Is a Memory Problem

Show `QK^T`, softmax, and `PV`; then make the `N x N` intermediate explicit.

### Standard Attention Materializes the Wrong Thing

Explain why materializing `S` and `P` costs HBM traffic and memory.

### Tiling Attention

Explain loading blocks into SRAM, computing partial attention contributions, and updating output.

### Online Softmax Makes It Exact

Describe row-wise max and normalization statistics in prose. Include formulas only after technical review.

### Recomputation in Backward

Explain storing normalization statistics and output instead of `N x N` attention probabilities.

### IO-Aware Co-Design

Frame FlashAttention as algorithm plus kernel plus hardware memory hierarchy.

### Beyond FlashAttention v1

Short sidebar: FA2 improves work partitioning, FA3 targets Hopper asynchrony and low precision, and FA4 targets Blackwell asymmetric scaling. Do not include benchmark numbers without benchmark conditions.

### Decode Is Different

Short bridge: training/prefill attention and decode attention have different query/context shapes.

## Technical Checks

- Formula correctness: Online softmax update and rescaling need careful review.
- Complexity / memory accounting: Verify any `O(N^2)` versus linear-memory wording.
- Hardware assumptions: HBM/SRAM terminology is GPU-centered; qualify accordingly.
- Benchmark conditions: Do not print raw speedup claims unless tied to Dao et al. setup or later primary papers.
- Terminology consistency: Use `HBM`, `SRAM`, `attention matrix`, `score matrix`, `probability matrix`, `online softmax`, `recomputation`, and `exact attention` consistently.

## Sidebar Decisions

- FA2/FA3/FA4: sidebar only unless Chapter 6 expands into version-specific treatment.
- Block-sparse FlashAttention: optional sidebar; not needed for main chapter.
- Decode attention: short bridge only; deeper treatment belongs to serving chapters.

## Open Questions

- Verify the online softmax formulas before technical-review clearance.
- Decide whether FA2/FA3/FA4 sidebar should include benchmark conditions or stay qualitative.

## Handoff

Owner: Book Architect  
Purpose: Chapter 6 brief from FlashAttention lecture and primary paper cards  
Evidence grade: A for FlashAttention v1 and FA2/FA3/FA4 primary-paper claims  
Assumptions: Chapter 6 is a case study in IO-aware exact attention, not a full history of all FlashAttention versions  
Handoff: Systems Explainer for draft
