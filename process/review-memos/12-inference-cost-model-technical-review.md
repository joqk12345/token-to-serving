# Technical Review: Chapter 12 — The Cost Model of LLM Inference

Date: 2026-07-05  
Reviewer: Codex  
Chapter: `book/chapters/12-inference-cost-model.md`

## Verdict

Cleared for red-team review.

The chapter is technically coherent at draft level. It uses A-grade sources: serving scheduler and serving framework lectures plus the ORCA OSDI paper. It avoids benchmark numbers, keeps the cost model conceptual, and preserves the boundary to Chapter 13's KV-cache/vLLM details.

## Checks

### Chapter Boundary

- The draft correctly positions Chapter 12 as a cost-model chapter.
- PagedAttention/vLLM internals are deferred to Chapter 13.
- Disaggregated serving and deeper cache/routing policy are deferred to Chapter 14.
- Framework examples are used for architecture boundaries rather than recommendations.

### Serving Model

- The chapter distinguishes serving from training by emphasizing online arrivals, variable prompt lengths, variable output lengths, streaming, queueing, and shared prefixes.
- The serving stack separates request/API handling, tokenization, scheduling, model workers, memory management, and detokenization.
- Product/framework claims are kept at course-lecture architecture level.

### Prefill and Decode

- Prefill is correctly described as prompt processing and initial state construction.
- Decode is correctly described as token-by-token generation that reuses prior keys/values.
- The draft states that decode is sequential within a request but batchable across requests.
- It avoids unsupported TTFT/TPOT terminology and leaves formal metric cards for later.

### Batching and Scheduling

- Request-level batching failure is supported by both lecture and ORCA.
- Continuous batching is correctly framed as iteration-level scheduling that updates active requests between decode iterations.
- The draft does not imply continuous batching removes all latency or fairness tradeoffs.
- Selective batching is explained as operation-level batching, with attention/KV handling treated separately.

### Scheduler Overhead

- Scheduler responsibilities are described accurately: receive, stream, check stop, reorder, prepare batches, and allocate memory.
- CPU scheduler overhead is correctly treated as potentially entering the GPU critical path.
- Scheduler/worker overlap is presented qualitatively without performance numbers.

### KV Cache and Prefix Reuse

- KV cache is introduced as serving memory for prior-token keys and values.
- The draft lists factors affecting KV cache growth without using byte formulas.
- RadixAttention and prefix reuse are introduced only as cost-model terms.
- Detailed prefix-tree/KV-cache implementation is deferred to later chapters.

### Conceptual Cost Model

- The final cost model is explicitly a checklist, not a numerical formula.
- Terms include queueing, prefill, decode, KV memory pressure, prefix reuse, and scheduler/runtime overhead.
- No throughput or latency numbers are used.

## Risks Remaining

| Risk | Blocking? | Notes |
|---|---:|---|
| No formal latency metric source card. | No | Draft uses plain-language metric names. |
| KV cache treatment is shallow. | No | Correctly deferred to Chapter 13. |
| Framework examples may age. | No | Draft avoids current product claims. |
| Cost model is qualitative. | No | Appropriate until source cards support formulas. |
| Continuous batching fairness/admission policy is thin. | No | Chapter 14 can deepen scheduling policy. |

## Required Fixes

None.

## Red-Team Prompts

- Does the conceptual cost equation look more precise than intended?
- Does the chapter overstate continuous batching as solving variable-length serving?
- Does the KV-cache section leak too much into Chapter 13?
- Does the prefill/decode explanation sufficiently distinguish compute and memory pressure?
- Does the framework discussion risk becoming a product survey?
- Does the chapter need formal latency metric terminology before ready status?

Owner: Technical Reviewer  
Purpose: Chapter 12 technical review  
Evidence grade: A for course lectures and ORCA paper; no benchmark numbers used  
Assumptions: Review evaluates draft-level correctness; final copy may add formal latency metric source cards later  
Open questions: Whether to add TTFT/TPOT terminology with explicit sources in a later revision  
Handoff: Red Team reviewer for adversarial critique
