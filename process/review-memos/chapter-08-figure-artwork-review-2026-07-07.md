# Chapter 8 Figure Artwork Review

Date: 2026-07-07

Scope: `book/figures/artwork/ch08/*.svg`

## Verdict

The five Chapter 8 diagrams are cleared as first-pass book artwork:

- `fig-08-data-parallel-step`
- `fig-08-collective-semantics`
- `fig-08-ring-allreduce`
- `fig-08-naive-vs-overlap-ddp`
- `fig-08-ddp-reducer-hooks`

The assets use the established 1200 by 760 canvas, type hierarchy, line weights, and semantic palette. Each SVG includes an accessible title and description.

## Checks

| Check | Result | Notes |
| --- | --- | --- |
| XML validity | pass | All five files pass `xmllint --noout`. |
| Technical overclaim scan | pass | No bandwidth formula, benchmark number, or perfect-overlap claim was introduced. |
| Terminology | pass | Replica, data shard, all-reduce, reduce-scatter, all-gather, bucket, reducer, and autograd hook match Chapter 8. |
| Semantics | pass | Model parameters remain replicated; collective ownership patterns and sum-versus-training-scale distinction are explicit. |
| Scope boundaries | pass | The ring is labeled logical, and reducer-hook details are labeled PyTorch-specific. |
| Layout | pass | Quick Look rendered all five SVGs; hierarchy, timelines, ownership symbols, and arrows remain inside the view box. |
| Accessibility | pass | Each file has `role="img"` plus linked `title` and `desc` elements. |

## Review Notes

- The data-parallel figure explicitly distinguishes model replication from model sharding.
- The collective table uses symbolic ownership and does not encode communication-volume assumptions.
- The ring figure separates scatter-reduce from all-gather and avoids equating the logical ring with physical topology.
- The overlap timeline keeps a final communication tail visible and states that overlap is conditional.
- The reducer figure traces gradient-ready events through hooks, pending counts, bucket readiness, asynchronous all-reduce, and completion.

Owner: Diagram Producer  
Purpose: First-pass Chapter 8 artwork and rendered-artwork review  
Evidence grade: A for source-aligned conceptual diagrams; no benchmark claims  
Assumptions: SVG is the canonical editable format and final typography may be adjusted during publication layout  
Open questions: Whether ring byte-count formulas will be added later with explicit topology and accounting assumptions  
Handoff: Chapter 9 diagram production using the same visual system
