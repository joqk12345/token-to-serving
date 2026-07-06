# Chapter 1 Figure Artwork Review

Date: 2026-07-06

Scope: `book/figures/artwork/ch01/*.svg`

## Verdict

The four Chapter 1 diagrams are cleared as first-pass book artwork:

- `fig-01-visible-vs-hidden-stack`
- `fig-01-token-probability-chain`
- `fig-01-abstraction-levels`
- `fig-01-codesign-loop`

The assets use one 1200 by 760 canvas, one type hierarchy, consistent line weights, and a shared semantic palette. Each SVG includes an accessible title and description.

## Checks

| Check | Result | Notes |
| --- | --- | --- |
| XML validity | pass | All four files pass `xmllint --noout`. |
| Technical overclaim scan | pass | No unsupported speedup, benchmark, or universal-performance claim was introduced. |
| Terminology | pass | `token`, `KV cache`, runtime, kernel, GPU/TPU, memory, communication, and latency match the registries and chapter usage. |
| Formula | pass | The chain-rule figure states conditional token probabilities without adding unsupported numeric examples. |
| Layout | pass | Text and shapes remain within the declared SVG view box; Quick Look rendering confirms type, color, arrows, and hierarchy. |
| Accessibility | pass | Each file has `role="img"` plus linked `title` and `desc` elements. |

## Review Notes

- The token boundaries in `fig-01-token-probability-chain` are explicitly labeled illustrative.
- The abstraction figure distinguishes an engineering abstraction ladder from a deployment stack.
- The co-design loop gives model, algorithm, software, and hardware equal visual weight.
- No benchmark values, hardware limits, latency values, or speedup claims appear in the artwork.
- Final publication layout may still require PDF or print-profile export from the editable SVG sources.

Owner: Diagram Producer  
Purpose: First-pass Chapter 1 artwork and rendered-artwork review  
Evidence grade: A for source-aligned conceptual diagrams; no benchmark claims  
Assumptions: SVG is the canonical editable format and final typography may be adjusted during publication layout  
Open questions: Whether the publication pipeline requires PDF, EPS, or outlined-font exports  
Handoff: Chapter 2 diagram production using the same visual system
