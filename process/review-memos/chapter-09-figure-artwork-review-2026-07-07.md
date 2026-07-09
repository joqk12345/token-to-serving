# Chapter 9 Figure Artwork Review

Date: 2026-07-07

Scope: `book/figures/artwork/ch09/*.svg`

## Verdict

The seven Chapter 9 diagrams are cleared as first-pass book artwork:

- `fig-09-data-vs-model-parallel`
- `fig-09-naive-pipeline-bubble`
- `fig-09-gpipe-microbatches`
- `fig-09-1f1b-vs-gpipe`
- `fig-09-tensor-parallel-ffn`
- `fig-09-tensor-parallel-attention-heads`
- `fig-09-3d-parallelism`

The assets use the established 1200 by 760 canvas, type hierarchy, line weights, and semantic palette. Each SVG includes an accessible title and description.

## Checks

| Check | Result | Notes |
| --- | --- | --- |
| XML validity | pass | All seven files pass `xmllint --noout`. |
| Technical overclaim scan | pass | No bubble formula, activation-memory number, fixed cluster size, or universal performance claim was introduced. |
| Terminology | pass | Data, pipeline, tensor, micro-batch, GPipe, 1F1B, head, shard, and collective match Chapter 9. |
| Mechanism | pass | Replication, pipeline occupancy, scheduling, tensor partitioning, head partitioning, and parallel-axis composition remain distinct. |
| Scope boundaries | pass | Head-local independence is not generalized to communication-free attention; 3D parallel grouping is labeled illustrative. |
| Layout | pass | Quick Look rendered all seven SVGs; timelines, partitions, arrows, and nested groups remain inside the view box. |
| Accessibility | pass | Each file has `role="img"` plus linked `title` and `desc` elements. |

## Review Notes

- The orientation figure distinguishes gradient synchronization from communication within the model graph.
- The naive schedule exposes idle stage time without attaching a bubble formula.
- The GPipe figure retains warmup, drain, and activation-storage caveats.
- The 1F1B comparison is qualitative and does not assign a universal memory reduction.
- The FFN figure shows column-local hidden work, row-partial output, and a collective assembly boundary.
- The attention figure explicitly preserves the output-projection communication caveat.
- The composed-parallelism figure names the different pressure on tensor, pipeline, and data axes without prescribing topology.

Owner: Diagram Producer  
Purpose: First-pass Chapter 9 artwork and rendered-artwork review  
Evidence grade: A for source-aligned conceptual diagrams; no benchmark claims  
Assumptions: SVG is the canonical editable format and final typography may be adjusted during publication layout  
Open questions: Whether pipeline bubble or activation-memory formulas will later be added with explicit schedule assumptions  
Handoff: Chapter 10 diagram production using the same visual system
