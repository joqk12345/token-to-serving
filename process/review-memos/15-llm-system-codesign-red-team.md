# Red Team: Chapter 15 — Model-Algorithm-System Co-Design

Date: 2026-07-05  
Reviewer: Codex  
Chapter: `book/chapters/15-llm-system-codesign.md`

## Verdict

Cleared for ready promotion.

The chapter succeeds as a synthesis chapter. It stays concrete enough by using long context, large training, cheap adaptation, and goodput-oriented serving as examples. It frames co-design as a decision method rather than a slogan and does not introduce new benchmark claims.

## Attacks and Outcomes

### Attack 1: The chapter may become too abstract.

Outcome: Addressed.

The draft uses concrete synthesis cases from prior chapters: long context, large training, adaptation/serving, and SLO-aware serving. Each case names the bottleneck and the layer where interventions occur.

### Attack 2: Co-design may sound like a generic slogan.

Outcome: Addressed.

The draft gives a bottleneck-ownership checklist and a seven-step engineering checklist. It asks for workload, target, bottleneck, owner layer, intervention, new bottleneck, and evidence.

### Attack 3: The chapter may introduce unsupported new claims while summarizing previous chapters.

Outcome: Addressed.

The draft cites prior source cards for the mechanisms it reuses. It does not add new performance numbers, current product claims, or future predictions.

### Attack 4: Decoder-only dominance may be overstated.

Outcome: Addressed.

The draft uses the careful phrasing from the source card: decoder-only autoregressive models are presented in the course as the most popular architecture choice for modern LLMs.

### Attack 5: The checklist may not enforce evidence discipline.

Outcome: Addressed.

The checklist explicitly asks what source supports the claim and what conditions travel with each number.

### Attack 6: The chapter may ignore negative tradeoffs.

Outcome: Addressed.

The draft repeatedly states that optimizations move bottlenecks and gives examples: larger serving batches increase KV pressure, disaggregation adds KV transfer, quantization requires compatible kernels and introduces numerical risk.

## Required Fixes

None.

## Recommended Non-Blocking Follow-Up

- Add a chapter-to-bottleneck summary table for all 15 chapters.
- Add a co-design loop figure and a bottleneck routing figure.
- Consider a short heterogeneous hardware/kTransformers sidebar if supported by source cards.
- Add exercises or end-of-chapter design prompts if the book later gains exercises.

Owner: Red Team Reviewer  
Purpose: Chapter 15 adversarial review  
Evidence grade: A for reviewed source map and cited prior chapter cards; no new factual claims introduced beyond existing cards  
Assumptions: Red-team review evaluates readiness under current synthesis scope, not final copyediting or figure production  
Open questions: None blocking  
Handoff: Principal author may promote Chapter 15 to ready; next production focus should shift to front-half gated reviews or book-level consistency audit
