# Chapter 7 Figure Artwork Review

Date: 2026-07-07

Scope: `book/figures/artwork/ch07/*.svg`

## Verdict

The seven Chapter 7 diagrams are cleared as first-pass book artwork:

- `fig-07-python-to-device-program`
- `fig-07-forward-to-backward-graph`
- `fig-07-ir-stack`
- `fig-07-xla-memory-planning`
- `fig-07-tpu-systolic-compile-target`
- `fig-07-pallas-blockspec-grid`
- `fig-07-pallas-pipeline`

The assets use the established 1200 by 760 canvas, type hierarchy, line weights, and semantic palette. Each SVG includes an accessible title and description.

## Checks

| Check | Result | Notes |
| --- | --- | --- |
| XML validity | pass | All seven files pass `xmllint --noout`. |
| Technical overclaim scan | pass | No benchmark number, fixed speedup, or universal compiler-performance claim was introduced. |
| Terminology | pass | JAXpr, StableHLO, HLO, backend IR, MXU, BlockSpec, HBM, and VMEM match Chapter 7. |
| Scope boundaries | pass | JAX/XLA, TPU, and Pallas-specific mechanisms are explicitly labeled rather than generalized to all systems. |
| Mechanism | pass | Program capture, autodiff, IR lowering, memory planning, systolic execution, block mapping, and pipelining remain distinct. |
| Layout | pass | Quick Look rendered all seven SVGs; hierarchy, arrows, timelines, and labels remain inside the declared view box. |
| Accessibility | pass | Each file has `role="img"` plus linked `title` and `desc` elements. |

## Review Notes

- The compiler pipeline separates Python source from scheduled device execution and labels the stack as JAX/XLA-oriented.
- The autodiff figure distinguishes forward edges, reverse-mode gradient flow, and saved-value dependencies.
- The IR stack marks portability-oriented and hardware-specific boundaries without claiming they are universal.
- The XLA memory figure treats fusion and buffer reuse as conditional compiler choices.
- The systolic-array figure names TPU/MXU explicitly.
- The BlockSpec figure prioritizes coordinate-to-slice mapping over API syntax.
- The Pallas timeline presents overlap as a latency-hiding mechanism with buffer and scheduling costs, not a guaranteed speedup.

Owner: Diagram Producer  
Purpose: First-pass Chapter 7 artwork and rendered-artwork review  
Evidence grade: A for source-aligned conceptual diagrams; no benchmark claims  
Assumptions: SVG is the canonical editable format and final typography may be adjusted during publication layout  
Open questions: Whether the publication pipeline requires PDF, EPS, or outlined-font exports  
Handoff: Chapter 8 diagram production using the same visual system
