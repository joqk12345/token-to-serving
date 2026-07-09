# Chapter 13 Figure Artwork Review

Date: 2026-07-08

Scope: `book/figures/artwork/ch13/*.svg`

## Verdict

The nine Chapter 13 diagrams are cleared as first-pass book artwork: KV growth, contiguous fragmentation, logical blocks, block-table mapping, PagedAttention access, fixed-block fragmentation, prefix sharing, preemption recovery, and the vLLM engine boundary.

## Checks

| Check | Result | Notes |
| --- | --- | --- |
| XML validity | pass | All nine files pass `xmllint --noout`. |
| Technical overclaim scan | pass | No memory percentage, block-size constant, or recovery-performance ranking was introduced. |
| Terminology | pass | Logical block, physical block, block table, KV cache, PagedAttention, sharing, and preemption match Chapter 13. |
| Mechanism | pass | Allocation, mapping, kernel access, sharing, recovery, and engine ownership remain distinct. |
| Scope boundaries | pass | The page-table analogy is qualified; sharing stops at divergence; swap and recompute remain conditional alternatives. |
| Layout | pass | Quick Look rendered all nine SVGs without parser or thumbnail failures. |
| Accessibility | pass | Each file has `role="img"` plus linked `title` and `desc` elements. |

Owner: Diagram Producer  
Purpose: First-pass Chapter 13 artwork and rendered-artwork review  
Evidence grade: A for source-aligned conceptual diagrams; no benchmark claims  
Assumptions: SVG is the canonical editable format and fixed-size blocks are shown symbolically  
Open questions: Whether publication layout needs a more detailed block-table kernel inset  
Handoff: Chapter 14 diagram production using the same visual system
