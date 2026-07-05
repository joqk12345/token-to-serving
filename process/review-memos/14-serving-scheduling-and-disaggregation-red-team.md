# Red Team: Chapter 14 — Scheduling, Caching, and Disaggregated Serving

Date: 2026-07-05  
Reviewer: Codex  
Chapter: `book/chapters/14-serving-scheduling-and-disaggregation.md`

## Verdict

Cleared for ready promotion.

The chapter holds up under adversarial review. It makes goodput under SLOs the central problem, keeps disaggregation conditional on transfer and placement costs, and grounds KV-cache-as-data language in concrete operations rather than marketing claims.

## Attacks and Outcomes

### Attack 1: Disaggregation may sound universally better than colocation.

Outcome: Addressed.

The draft explicitly states that colocation is not always wrong and that disaggregation should be judged by whether SLO benefit exceeds placement and KV-transfer cost for the workload.

### Attack 2: Goodput may be confused with throughput.

Outcome: Addressed.

The draft defines throughput as completed requests per time and goodput as completed requests satisfying SLO per time. It also defines TTFT and TPOT separately.

### Attack 3: KV-aware routing may sound like maximizing cache hit rate only.

Outcome: Addressed.

The routing section balances queueing, cache locality, and SLO risk. It explicitly notes that an overloaded cache-local worker may not be the right route.

### Attack 4: "KV cache as data" may become vague or promotional.

Outcome: Addressed.

The draft grounds the phrase in concrete operations: get/put, share, offload, reuse, compress, and transfer.

### Attack 5: Product/system examples may overclaim current capabilities.

Outcome: Addressed.

Dynamo, LMCache, Mooncake, and NIXL are used as architecture examples tied to lecture cards. The draft avoids exact current API claims, version claims, or product recommendations.

### Attack 6: Transfer hierarchy may understate latency and bandwidth cost.

Outcome: Addressed.

The draft states that moving cache through the hierarchy has a cost and that transfer can be on the serving critical path when routing, disaggregation, or reuse requires it.

### Attack 7: Lack of DistServe paper card may weaken readiness.

Outcome: Non-blocking.

The draft uses only lecture-level DistServe mechanism: placement dimensions, goodput framing, and KV-transfer challenge. It avoids algorithm reproduction and evaluation numbers, so the lecture source is sufficient under current claim scope.

## Required Fixes

None.

## Recommended Non-Blocking Follow-Up

- Add the DistServe paper as a publication-level card before reproducing placement algorithms or evaluation results.
- Add official docs/source-code cards before describing exact Dynamo, LMCache, Mooncake, or NIXL APIs.
- Add figure specs for goodput, PD disaggregation, KV-aware routing, and memory-tier transfer.
- Decide whether kTransformers belongs as a Chapter 14 sidebar or Chapter 15 co-design example.

Owner: Red Team Reviewer  
Purpose: Chapter 14 adversarial review  
Evidence grade: A for reviewed source map; no new factual claims introduced beyond existing cards  
Assumptions: Red-team review evaluates readiness under current claim scope, not final copyediting or figure production  
Open questions: None blocking  
Handoff: Principal author may promote Chapter 14 to ready; next production focus can move to Chapter 15 source extraction
