---
status: brief
chapter: 12
slug: 12-inference-cost-model
title: The Cost Model of LLM Inference
primary_sources:
  - llmsys-22-llm-serving-scheduler-radixattention-dfa87a4515092525676277a85bc4425d.pdf
  - llmsys-30-serving-c4a70ab21cde01fb60068a256c6e163a.pdf
papers:
  - downloads/llmsystem2026spring/source_pdfs/osdi22-yu.pdf
secondary_sources: []
reader_level: engineer or graduate student who has read Chapters 1-11
technical_depth: intermediate-to-advanced
---

# The Cost Model of LLM Inference

## Chapter Thesis

LLM inference cost is controlled by a sequence of coupled bottlenecks: prompt processing, token-by-token decoding, batch scheduling, KV-cache memory, and user-facing latency. The serving system is not a passive wrapper around a model; it is the runtime that decides which requests run together, which memory remains resident, and which latency metric the user experiences.

## Reader Problem

The reader understands model computation, distributed training, memory partitioning, quantization, and PEFT. The next question is what changes when the model is served to users. Training optimizes a step over known batches. Serving handles online arrivals, variable prompt lengths, variable output lengths, streaming responses, and shared prompt prefixes. The reader needs a cost model before studying vLLM/PagedAttention or disaggregated serving in later chapters.

## System Bottleneck

Primary bottlenecks: prefill compute, decode iteration latency, KV-cache memory, memory bandwidth, batch occupancy, request queueing, scheduler overhead, tokenization/detokenization, prompt-prefix reuse, and user-facing latency.

Secondary bottlenecks: output length variance, batch admission policy, stop conditions, CPU/GPU coordination, cache hit rate, worker routing, model-parallel worker topology, attention backend, and API/server overhead.

## Source Map

| Claim | Source card | Evidence grade | Notes |
|---|---|---|---|
| LLM servers must handle multi-user sessions, variable generation lengths, common prompt prefixes, latency, throughput, and heterogeneous devices. | `llmsys-22-serving-goals` | A | Chapter opening. |
| The serving application stack separates prompt/data preparation from prompt execution/inference. | `llmsys-30-llm-app-stack` | A | Keep high level. |
| Serving frameworks separate request/API concerns from optimized model execution engines. | `llmsys-30-serving-framework-boundary` | A | Avoid product survey. |
| Triton can group requests while an inference engine executes batched model computation. | `llmsys-30-triton-batching-engine` | A | Architecture example only. |
| A request scheduler receives requests, streams outputs, checks stop conditions, reorders requests, prepares batches, and allocates memory. | `llmsys-22-request-scheduler` | A | Core scheduler loop. |
| Autoregressive generation processes a request over multiple model iterations, one output token per iteration. | `yu-2022-orca-autoregressive-iterations` | A | Primary paper support. |
| Request-level batching is inefficient for variable-length generative workloads. | `yu-2022-orca-request-level-limitation` | A | Pair with lecture naive batching slide. |
| Naive batching can wait for the longest generation in a batch. | `llmsys-22-naive-batch-longest` | A | Use as concrete bottleneck. |
| Iteration-level/continuous batching updates batch membership between iterations. | `yu-2022-orca-iteration-level-scheduling` | A | Primary paper support. |
| Continuous batching schedules at token-generation granularity. | `llmsys-22-continuous-batching` | A | Course framing. |
| Selective batching handles attention separately while batching compatible operations. | `yu-2022-orca-selective-batching` | A | Avoid deep attention engine detail. |
| Selective batching can batch non-attention operations while attention uses specialized handling. | `llmsys-22-selective-batching` | A | Course framing. |
| Transformer generation stores previous-token keys and values in GPU memory as KV cache. | `llmsys-22-kv-cache-need` | A | Boundary to Ch. 13. |
| RadixAttention stores KV pointers in a radix tree keyed by prompt prefixes. | `llmsys-22-radixattention-prefix-cache` | A | Mention lightly; Ch. 13 deep dive. |
| Cache-aware scheduling can use matched prefix length or predicted cache hit rates. | `llmsys-22-cache-aware-scheduling` | A | No throughput table. |
| Scheduler CPU work can become overhead; systems can overlap scheduling with GPU worker execution. | `llmsys-22-scheduler-worker-overlap` | A | Runtime overhead. |
| LightLLM-style frameworks can optimize tokenization/inference/detokenization and per-token KV memory/routing. | `llmsys-30-lightllm-async-token-attention` | A | Example only, no current product claims. |

## Explanation Arc

1. Start with serving as online inference: requests arrive continuously and have different prompt/output lengths.
2. Define the two major phases: prefill processes the prompt; decode generates one token per iteration.
3. Explain why autoregressive generation is multi-iteration and why output length variance breaks naive request-level batching.
4. Introduce user-facing latency metrics qualitatively: time to first token, time per output token, total response time, and throughput.
5. Explain the scheduler loop: receive, batch, run, stream, check stop, free/reuse memory.
6. Explain continuous batching as the response to variable-length generation.
7. Explain selective batching as the response to operation-level shape differences.
8. Introduce KV cache as the memory cost of avoiding recomputation during decode.
9. Introduce prompt-prefix reuse and RadixAttention only enough to motivate cache-aware scheduling.
10. Explain that CPU scheduling can become overhead and must be overlapped with GPU execution.
11. Close with a cost model: inference cost is a function of prompt length, generated length, batch policy, KV memory, cache reuse, and scheduler overhead.

## Required Figures

| Figure ID | Purpose | Form | Source |
|---|---|---|---|
| `fig-12-prefill-decode` | Show prompt prefill followed by token-by-token decode. | Timeline/dataflow | `yu-2022-orca-autoregressive-iterations` |
| `fig-12-request-level-batching-problem` | Show short requests waiting for longest generation. | Timeline | `llmsys-22-naive-batch-longest` |
| `fig-12-continuous-batching` | Show batch membership changing between decode iterations. | Timeline | `llmsys-22-continuous-batching`, `yu-2022-orca-iteration-level-scheduling` |
| `fig-12-scheduler-loop` | Show scheduler receive/run/result/free loop. | Loop diagram | `llmsys-22-request-scheduler` |
| `fig-12-selective-batching` | Show non-attention batching with special attention path. | Block diagram | `llmsys-22-selective-batching` |
| `fig-12-kv-cache-cost` | Show per-layer KV cache accumulating with tokens. | Memory growth diagram | `llmsys-22-kv-cache-need` |
| `fig-12-prefix-cache-routing` | Show prompt prefix reuse and cache-aware scheduling. | Prefix tree + queue | `llmsys-22-radixattention-prefix-cache`, `llmsys-22-cache-aware-scheduling` |

## Main Sections

### Serving Is Online, Not a Training Step

Explain online request arrivals, streaming, variable prompt/output lengths, and queueing. Distinguish training batch from serving batch.

### Prefill and Decode

Define prefill and decode. Prefill consumes the input prompt and creates initial KV state; decode generates one token at a time and reuses KV cache.

### Latency Metrics

Introduce time to first token, per-token decode latency, total response time, throughput, and queueing delay. Keep qualitative unless adding explicit metric source cards.

### Why Request-Level Batching Fails

Use variable output lengths and ORCA's request-level limitation. Show short outputs waiting for the longest generation.

### Continuous Batching

Explain iteration-level scheduling: update batch membership between decode iterations. Tie to throughput/latency tradeoff without benchmark numbers.

### Selective Batching

Explain that not all operations batch equally when sequence lengths differ. Non-attention operations may batch cleanly; attention/KV handling can need specialized execution.

### The Scheduler Is Part of the Critical Path

Explain scheduler responsibilities and why CPU-side scheduling, stop checks, memory allocation, and result processing can affect GPU utilization.

### KV Cache as Serving Memory

Explain why Transformer decode stores keys/values for previous tokens. Make it clear that Chapter 13 handles PagedAttention/vLLM in depth.

### Prefix Reuse and Cache-Aware Scheduling

Introduce prompt-prefix sharing, radix tree lookup, and cache-aware routing as cost-model terms.

### Cost Model Summary

Conclude with variables:

```text
request cost ≈ queueing
             + prefill(prompt length)
             + decode(output length, batch policy)
             + KV memory pressure
             + scheduler/runtime overhead
```

State explicitly that this is a conceptual model, not a numerical formula.

## Technical Checks

- Do not quote throughput or latency numbers from lecture tables or ORCA without model/hardware/request distribution setup.
- Do not collapse prefill and decode into one uniform cost.
- Do not imply continuous batching removes all latency tradeoffs.
- Do not move detailed PagedAttention/vLLM design into Chapter 12.
- Do not present product/framework examples as current recommendations.
- Distinguish request-level, iteration-level, and operation-level batching.
- Keep KV cache numeric examples out unless model, precision, layers, hidden dimension, sequence length, and batch assumptions are carried.

## Sidebar Decisions

- ORCA: name in main prose as primary source for iteration-level scheduling; no benchmark numbers.
- Triton/TensorRT-LLM: optional architecture sidebar only.
- LightLLM: optional example for async tokenization/inference/detokenization and token-wise memory management.
- RadixAttention: brief introduction only; detailed prefix/KV-cache mechanisms belong in Chapter 13 or 14.

## Open Questions

- Should the draft introduce formal TTFT/TPOT terminology now, or keep metric names plain-language until a metric source card is added?
- Should Chapter 12 include a simple symbolic cost model or only a conceptual decomposition?
- How much RadixAttention should be included before overlapping Chapter 13?
- Should current framework docs be added if named systems remain in main prose?

## Handoff

Owner: Book Architect  
Purpose: Chapter 12 brief from serving scheduler, ORCA, and serving framework sources  
Evidence grade: A for course lecture claims and ORCA paper; no benchmark numbers used  
Assumptions: Chapter 12 is a cost-model chapter; Chapter 13 handles KV-cache/vLLM internals  
Open questions: Latency metric terminology and RadixAttention depth  
Handoff: Systems Explainer for Chapter 12 draft
