# Chapter 5 Figure Artwork Review

Date: 2026-07-06

Scope: `book/figures/artwork/ch05/*.svg`

## Verdict

The five Chapter 5 diagrams are cleared as first-pass book artwork:

- `fig-05-memory-bound-intuition`
- `fig-05-naive-vs-tiled-matmul`
- `fig-05-coalescing-transpose`
- `fig-05-transformer-operator-map`
- `fig-05-kernel-fusion`

The assets continue the established 1200 by 760 canvas, type hierarchy, line weights, and semantic palette. Each SVG includes an accessible title and description.

## Checks

| Check | Result | Notes |
| --- | --- | --- |
| XML validity | pass | All five files pass `xmllint --noout`. |
| Technical overclaim scan | pass | No unsupported bandwidth, speedup, or universal performance claim was introduced. |
| Terminology | pass | Arithmetic intensity, tiling, shared memory, coalescing, GEMM, reduction, elementwise, and fusion match Chapter 5. |
| Mechanism | pass | Data reuse, access layout, operator categories, and intermediate-memory removal remain distinct. |
| Layout | pass | Text and shapes remain within the SVG view box; rendered thumbnails confirm hierarchy and dataflow. |
| Accessibility | pass | Each file has `role="img"` plus linked `title` and `desc` elements. |

## Review Notes

- The memory-bound comparison is explicitly conceptual and not presented as a formal roofline model.
- The tiled matrix multiplication figure shows synchronization and shared-memory reuse without assigning hardware-specific tile sizes.
- The transpose figure keeps bank conflicts separate from global-memory coalescing.
- The Transformer map classifies mixed operator work rather than duplicating the Chapter 2 architecture figure.
- The fusion example states conditional benefits and preserves `E = A + B + D` from the chapter.

Owner: Diagram Producer  
Purpose: First-pass Chapter 5 artwork and rendered-artwork review  
Evidence grade: A for source-aligned conceptual diagrams; no benchmark claims  
Assumptions: SVG is the canonical editable format and final typography may be adjusted during publication layout  
Open questions: Whether the publication pipeline requires PDF, EPS, or outlined-font exports  
Handoff: Chapter 6 diagram production using the same visual system
