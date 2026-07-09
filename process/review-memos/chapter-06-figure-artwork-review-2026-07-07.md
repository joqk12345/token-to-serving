# Chapter 6 Figure Artwork Review

Date: 2026-07-07

Scope: `book/figures/artwork/ch06/*.svg`

## Verdict

The five Chapter 6 diagrams are cleared as first-pass book artwork:

- `fig-06-standard-attention-hbm`
- `fig-06-flashattention-tiling`
- `fig-06-online-softmax`
- `fig-06-forward-backward-memory`
- `fig-06-modern-hardware-sidebar`

The assets complete the Part II figure set using the established 1200 by 760 canvas, type hierarchy, line weights, and semantic palette. Each SVG includes an accessible title and description.

## Checks

| Check | Result | Notes |
| --- | --- | --- |
| XML validity | pass | All five files pass `xmllint --noout`. |
| Technical overclaim scan | pass | No benchmark number, fixed speedup, or universal performance claim was introduced. |
| Terminology | pass | HBM, on-chip SRAM, score matrix, probability matrix, online softmax, and recomputation match Chapter 6. |
| Formula | pass | The online-softmax artwork matches the reviewed one-row formula and explicitly distinguishes normalized `O_old` from unnormalized `O_block`. |
| Mechanism | pass | Materialization, tiling, online rescaling, backward recomputation, and hardware evolution remain distinct. |
| Layout | pass | Quick Look rendered all five SVGs; text and shapes remain inside the declared view box. |
| Accessibility | pass | Each file has `role="img"` plus linked `title` and `desc` elements. |

## Review Notes

- The standard path says materialization is common rather than universal and makes both `N × N` intermediates explicit.
- The tiling figure states that all Q blocks still interact with all K/V blocks, preventing a sparse-attention interpretation.
- The online-softmax figure uses the exact notation reviewed in `chapter-6-online-softmax-formula-check-2026-07-04.md`.
- The backward figure presents recomputation as a storage and memory-traffic tradeoff, not as categorically faster.
- The FA2/FA3/FA4 sidebar is a mechanism timeline without benchmark or speedup claims.

Owner: Diagram Producer  
Purpose: First-pass Chapter 6 artwork and rendered-artwork review  
Evidence grade: A for source-aligned conceptual diagrams; no benchmark claims  
Assumptions: SVG is the canonical editable format and final typography may be adjusted during publication layout  
Open questions: Whether the publication pipeline requires PDF, EPS, or outlined-font exports  
Handoff: Chapter 7 diagram production using the same visual system
