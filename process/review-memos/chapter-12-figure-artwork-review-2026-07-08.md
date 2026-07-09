# Chapter 12 Figure Artwork Review

Date: 2026-07-08

Scope: `book/figures/artwork/ch12/*.svg`

## Verdict

The seven Chapter 12 diagrams are cleared as first-pass book artwork: `fig-12-prefill-decode`, `fig-12-request-level-batching-problem`, `fig-12-continuous-batching`, `fig-12-scheduler-loop`, `fig-12-selective-batching`, `fig-12-kv-cache-cost`, and `fig-12-prefix-cache-routing`.

## Checks

| Check | Result | Notes |
| --- | --- | --- |
| XML validity | pass | All seven files pass `xmllint --noout`. |
| Technical overclaim scan | pass | No latency value, utilization percentage, byte formula, or universal scheduler-policy claim was introduced. |
| Terminology | pass | Prefill, decode, continuous batching, scheduler, KV cache, and prefix reuse match Chapter 12. |
| Mechanism | pass | Work shape, batch membership, control loop, selective batching, memory growth, and cache-aware routing remain distinct. |
| Scope boundaries | pass | Admission policies are generic; prefix reuse remains conditional on routing, residency, and eviction. |
| Layout | pass | Quick Look rendered all seven SVGs without parser or thumbnail failures. |
| Accessibility | pass | Each file has `role="img"` plus linked `title` and `desc` elements. |

## Review Notes

- Prefill and decode are differentiated by work shape without latency claims.
- Fixed batching exposes tied-up slots; continuous batching visibly changes membership between iterations.
- The scheduler loop stays architecture-level and includes memory update/free.
- Selective batching separates shared operators from request-specific attention and cache state.
- KV growth adds one key/value pair per layer and avoids unsourced byte accounting.
- Prefix routing explicitly states that shared text does not guarantee a cache hit.

Owner: Diagram Producer  
Purpose: First-pass Chapter 12 artwork and rendered-artwork review  
Evidence grade: A for source-aligned conceptual diagrams; no benchmark claims  
Assumptions: SVG is the canonical editable format and final typography may be adjusted during publication layout  
Open questions: Whether formal TTFT and TPOT terminology will be added after a dedicated source-card pass  
Handoff: Chapter 13 diagram production using the same visual system
