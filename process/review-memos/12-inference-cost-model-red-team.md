# Red Team: Chapter 12 — The Cost Model of LLM Inference

Date: 2026-07-05  
Reviewer: Codex  
Chapter: `book/chapters/12-inference-cost-model.md`

## Verdict

Cleared for ready promotion.

The chapter presents inference serving as an online scheduling and memory problem, not a static forward-pass problem. It avoids benchmark numbers, keeps the cost model conceptual, and preserves the boundary to KV-cache/vLLM and disaggregated-serving chapters.

## Attacks and Outcomes

### Attack 1: The conceptual cost equation may look like a numerical formula.

Outcome: Addressed.

The draft explicitly states that the expression is not a numerical formula and should be read as a checklist. It does not attach coefficients or unsupported scaling laws to the terms.

### Attack 2: Continuous batching may sound like it solves variable-length serving completely.

Outcome: Addressed.

The draft says continuous batching improves the system's ability to keep the GPU occupied, but does not remove tradeoffs. It names KV memory contention, prefill/decode interference, fairness, queueing, and user-visible delay.

### Attack 3: KV-cache discussion may leak too much into Chapter 13.

Outcome: Non-blocking.

The chapter introduces KV cache only as serving memory and lists the variables that affect growth. It explicitly avoids byte formulas and leaves PagedAttention/vLLM details to Chapter 13.

### Attack 4: Prefill/decode may not sufficiently distinguish compute and memory pressure.

Outcome: Addressed.

The draft separates prompt-length-driven prefill, sequential decode, batchable cross-request decode iterations, and KV memory growth. It also states that active tokens consume persistent serving memory.

### Attack 5: Framework examples may become a product survey.

Outcome: Addressed.

The draft uses framework examples only to show architectural boundaries between request layers and model execution engines. It explicitly says the point is architectural, not a product recommendation.

### Attack 6: Formal latency metric terminology may be required.

Outcome: Non-blocking.

The draft uses plain-language metric names and explains queueing delay, time to first token, per-token decode time, total response time, and throughput. Formal abbreviations can be added later with source cards, but they are not required for readiness.

### Attack 7: Prefix caching may be overstated.

Outcome: Addressed.

The draft states prefix reuse can reduce effective prefill cost if there is a useful cache hit, but also mentions cache misses and eviction policy. It does not promise universal benefit.

## Required Fixes

None.

## Recommended Non-Blocking Follow-Up

- Add a prefill/decode timeline figure.
- Add a continuous batching timeline with requests entering and leaving between iterations.
- Add a small serving cost checklist box near the chapter end.
- Add formal TTFT/TPOT terminology only after adding source cards for metric definitions.

Owner: Red Team Reviewer  
Purpose: Chapter 12 adversarial review  
Evidence grade: A for reviewed source map; no new factual claims introduced beyond existing cards  
Assumptions: Red-team review evaluates readiness under current claim scope, not final copyediting or figure production  
Open questions: None blocking  
Handoff: Principal author may promote Chapter 12 to ready; next production focus can move to Chapter 13 source extraction
