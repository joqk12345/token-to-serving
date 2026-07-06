# Chapter 2 Figure Artwork Review

Date: 2026-07-06

Scope: `book/figures/artwork/ch02/*.svg`

## Verdict

The four Chapter 2 diagrams are cleared as first-pass book artwork:

- `fig-02-token-to-logits`
- `fig-02-qkv-attention`
- `fig-02-causal-mask`
- `fig-02-model-family-map`

The assets continue Chapter 1's 1200 by 760 canvas, type hierarchy, line weights, and semantic palette. Each SVG includes an accessible title and description.

## Checks

| Check | Result | Notes |
| --- | --- | --- |
| XML validity | pass | All four files pass `xmllint --noout`. |
| Technical overclaim scan | pass | No benchmark, speedup, hardware-limit, or universal-performance claim was introduced. |
| Terminology | pass | Token IDs, embeddings, logits, vocabulary, Q/K/V, causal mask, and model-family labels match Chapter 2. |
| Formula | pass | The single-head attention path shows scaled query-key scores, row-wise softmax, and value mixing. |
| Layout | pass | Text and shapes remain within the SVG view box; Quick Look rendering confirms hierarchy, arrows, matrix encoding, and color. |
| Accessibility | pass | Each file has `role="img"` plus linked `title` and `desc` elements. |

## Review Notes

- The token-to-logits figure makes the final vocabulary projection explicit and distinguishes logits from probabilities.
- The Q/K/V figure shows one attention head, avoiding multi-head complexity before the core mechanism is established.
- The causal-mask figure uses query rows and key columns, with future positions masked before softmax.
- The family map compares conditioning patterns and objectives without presenting the families as a quality ranking.
- No benchmark values, model sizes, context limits, latency values, or speedup claims appear in the artwork.

Owner: Diagram Producer  
Purpose: First-pass Chapter 2 artwork and rendered-artwork review  
Evidence grade: A for source-aligned conceptual diagrams; no benchmark claims  
Assumptions: SVG is the canonical editable format and final typography may be adjusted during publication layout  
Open questions: Whether the publication pipeline requires PDF, EPS, or outlined-font exports  
Handoff: Chapter 3 diagram production using the same visual system
