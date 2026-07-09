# Chapter 10 Figure Artwork Review

Date: 2026-07-07

Scope: `book/figures/artwork/ch10/*.svg`

## Verdict

The six Chapter 10 diagrams are cleared as first-pass book artwork:

- `fig-10-ddp-memory-ledger`
- `fig-10-zero-stages`
- `fig-10-zero3-parameter-gather`
- `fig-10-moe-router-experts`
- `fig-10-expert-parallel-alltoall`
- `fig-10-load-balance-skew`

The assets use the established 1200 by 760 canvas, type hierarchy, line weights, and semantic palette. Each SVG includes an accessible title and description.

## Checks

| Check | Result | Notes |
| --- | --- | --- |
| XML validity | pass | All six files pass `xmllint --noout`. |
| Technical overclaim scan | pass | No unconditional byte total, ZeRO performance ranking, MoE quality claim, or speedup number was introduced. |
| Terminology | pass | Optimizer state, shard, gather, router, expert, all-to-all, dispatch, return, and load balancing match Chapter 10. |
| Mechanism | pass | Memory accounting, ZeRO ownership, parameter residency, token routing, expert exchange, and load skew remain distinct. |
| Scope boundaries | pass | Gather communication is visible; both all-to-all directions are shown; uniform routing is not asserted as universally optimal. |
| Layout | pass | Quick Look rendered all six SVGs without parser or thumbnail failures. |
| Accessibility | pass | Each file has `role="img"` plus linked `title` and `desc` elements. |

## Review Notes

- The DDP ledger avoids byte totals and includes persistent, activation, transient, and fragmentation categories.
- The ZeRO-stage table describes ownership structure rather than ranking stages by performance.
- The ZeRO-3 timeline makes gather, compute, release, and repeated residency scheduling explicit.
- The MoE router figure limits itself to conditional expert selection and output reordering.
- The expert-parallel figure preserves both dispatch and return all-to-all paths.
- The load figure frames balancing as avoiding collapse and overload, not enforcing perfect uniformity.

Owner: Diagram Producer  
Purpose: First-pass Chapter 10 artwork and rendered-artwork review  
Evidence grade: A for source-aligned conceptual diagrams; no benchmark claims  
Assumptions: SVG is the canonical editable format and final typography may be adjusted during publication layout  
Open questions: Whether later layout adds the conditional ZeRO memory formulas as a separate boxed derivation  
Handoff: Chapter 11 diagram production using the same visual system
