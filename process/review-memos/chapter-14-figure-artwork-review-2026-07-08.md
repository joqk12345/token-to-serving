# Chapter 14 Figure Artwork Review

Date: 2026-07-08

Scope: `book/figures/artwork/ch14/*.svg`

## Verdict

The nine Chapter 14 diagrams are cleared as first-pass book artwork: throughput/goodput, TTFT/TPOT, phase interference, prefill-decode disaggregation, placement, KV-aware routing, KV memory hierarchy, external KV management, and the transfer substrate.

## Checks

| Check | Result | Notes |
| --- | --- | --- |
| XML validity | pass | All nine files pass `xmllint --noout`. |
| Technical overclaim scan | pass | No numeric SLO, tier latency, bandwidth, speedup, or universal placement claim was introduced. |
| Terminology | pass | Goodput, TTFT, TPOT, disaggregation, KV routing, memory tiers, and transfer substrate match Chapter 14. |
| Mechanism | pass | SLO accounting, interference, placement, routing, external state, and transfer remain distinct. |
| Scope boundaries | pass | KV transfer is visible; cache locality is one routing signal; topology and protocols are illustrative. |
| Layout | pass | Quick Look rendered all nine SVGs without parser or thumbnail failures. |
| Accessibility | pass | Each file has linked `title` and `desc` metadata. |

Owner: Diagram Producer  
Purpose: First-pass Chapter 14 artwork and rendered-artwork review  
Evidence grade: A for source-aligned conceptual diagrams; no benchmark claims  
Assumptions: SVG is the canonical editable format and placement/topology examples are illustrative  
Open questions: Whether publication layout needs system-specific Dynamo, LMCache, NIXL, or Mooncake insets  
Handoff: Chapter 15 diagram production using the same visual system
