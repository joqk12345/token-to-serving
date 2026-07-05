# Technical Review: Chapter 14 — Scheduling, Caching, and Disaggregated Serving

Date: 2026-07-05  
Reviewer: Codex  
Chapter: `book/chapters/14-serving-scheduling-and-disaggregation.md`

## Verdict

Cleared for red-team review.

The draft is technically coherent at chapter-draft level. It advances from single-engine KV-cache management to distributed serving, uses goodput under SLOs as the organizing metric, and explains prefill/decode disaggregation, KV-aware routing, external KV-cache management, memory tiers, and transfer substrates without relying on benchmark numbers.

## Checks

### Chapter Boundary

- The chapter correctly follows Chapter 13 by treating KV cache as distributed routing and data-management state.
- It does not re-explain PagedAttention internals.
- It keeps product-specific Dynamo, LMCache, Mooncake, and NIXL details at architectural level.
- It does not drift into Chapter 15 model-system co-design beyond naming heterogeneous/offload ideas as boundary context.

### Metrics and SLOs

- TTFT and TPOT are defined with source support.
- Goodput is correctly distinguished from raw throughput.
- The draft avoids numeric SLO examples and goodput numbers.
- It frames scheduling as SLO-aware rather than purely throughput-maximizing.

### Continuous Batching and Disaggregation

- Continuous batching is described as necessary but insufficient.
- The draft uses the lecture-supported distinction: continuous batching targets utilization/throughput; disaggregation targets goodput under SLOs.
- Prefill/decode disaggregation is not presented as universally superior; the draft states that KV-transfer and placement costs must be weighed.

### Prefill/Decode Mechanism

- Prefill is described as compute-bound and decode as memory-bound, consistent with the source card.
- The draft ties TTFT primarily to prefill/queueing and TPOT to decode cadence/batching/memory bandwidth.
- It avoids model-specific roofline or latency claims.

### Placement and Routing

- DistServe-style placement is correctly summarized as choosing parallelism strategy, instance count, and cluster placement.
- KV-aware routing balances cache locality with load and SLO risk.
- The draft does not imply that maximizing cache hit rate alone is the right routing objective.

### External KV Cache and Memory Hierarchy

- LMCache is used to explain external KV-cache management, not as a product recommendation.
- The draft grounds "KV cache as data" in concrete operations: get/put, share, offload, reuse, compress, transfer.
- HBM/DRAM/SSD/network storage are described as tiers without unsupported capacity/latency numbers.
- NIXL and Mooncake Store are treated as transfer-substrate examples; exact APIs are not claimed beyond lecture-supported abstractions.

### Advanced Cache Techniques

- CacheBlend/selective prefill and KV-cache compression are kept as optional advanced directions.
- No speedup, compression-ratio, cache-hit, or evaluation numbers are used.

## Risks Remaining

| Risk | Blocking? | Notes |
|---|---:|---|
| No DistServe paper card. | No | The draft uses lecture-level mechanism only and avoids algorithm/evaluation detail. |
| No official docs/source cards for Dynamo/LMCache/Mooncake/NIXL. | No | The draft avoids exact current API or feature claims. |
| Goodput discussion is qualitative. | No | Appropriate without experiment-specific SLO cards. |
| Transfer substrates are high-level. | No | Sufficient for chapter scope; API details can be added later. |
| Heterogeneous hardware/kTransformers is mostly omitted. | No | Better saved for Chapter 15 or a sidebar. |

## Required Fixes

None.

## Red-Team Prompts

- Does the chapter imply disaggregation always improves serving, despite KV-transfer cost?
- Does the goodput section sufficiently distinguish throughput, TTFT, TPOT, and SLO satisfaction?
- Does KV-aware routing risk sounding like cache-hit maximization rather than multi-objective routing?
- Does "KV cache as data" become marketing language instead of concrete store/move/reuse/compress operations?
- Does the chapter overclaim current capabilities of Dynamo, LMCache, Mooncake, or NIXL?
- Does the transfer hierarchy omit the latency/bandwidth cost enough to mislead?

Owner: Technical Reviewer  
Purpose: Chapter 14 technical review  
Evidence grade: A for course lecture claims; no benchmark numbers used  
Assumptions: Review evaluates draft-level mechanism correctness, not final copyediting or figure production  
Open questions: Whether to add DistServe paper and official system docs in a later revision  
Handoff: Red Team reviewer for adversarial critique
