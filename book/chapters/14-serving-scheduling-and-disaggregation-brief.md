---
status: brief
chapter: 14
slug: 14-serving-scheduling-and-disaggregation
title: Scheduling, Caching, and Disaggregated Serving
primary_sources:
  - llmsys-26-dynamo-vikram_mailthody-0ddbeb69382d5c168b6d4636a82185d0.pdf
  - llmsys-27-LMCache_junchenjiang-c21828fe582270cb5e08b4a21a002956.pdf
  - llmsys-28-mooncake-kTransformer-1243bfbecbb0c3610bf95eac030acb2a.pdf
  - llmsys-29-disaggregating_prefill_decode_hao_zhang-c0e55139d20512a2348783423397cc7f.pdf
papers: []
secondary_sources: []
reader_level: engineer or graduate student who has read Chapters 1-13
technical_depth: advanced
---

# Scheduling, Caching, and Disaggregated Serving

## Chapter Thesis

At serving scale, the system stops optimizing a single GPU loop and starts optimizing goodput under SLOs. Continuous batching keeps workers busy, but distributed serving must also route requests to useful KV state, move KV cache across memory tiers, separate prefill from decode when their bottlenecks conflict, and place resources so TTFT and TPOT targets can both be met.

## Reader Problem

The reader now understands inference cost, KV cache, PagedAttention, and vLLM-style single-engine memory management. The next problem is what happens when one engine is no longer the unit of design. Real serving workloads contain many users, multi-turn sessions, agentic calls, long contexts, heterogeneous hardware, and changing traffic. A system that maximizes raw throughput can still miss user-facing latency SLOs. The reader needs a distributed serving model based on scheduling, cache locality, memory transfer, and prefill/decode disaggregation.

## System Bottleneck

Primary bottlenecks: goodput under TTFT/TPOT SLOs, prefill/decode interference, KV-cache locality, routing policy, KV-cache transfer bandwidth, memory-tier placement, worker load balance, prefill/decode resource allocation, cluster placement, and online traffic variation.

Secondary bottlenecks: continuous batching policy, cache hit rate, host memory and SSD offload, remote KV pools, NIXL/Mooncake-style transfer layers, storage plugin interfaces, heterogeneous hardware, fault tolerance, and multi-model serving.

## Source Map

| Claim                                                                                                                                                            | Source card                                | Evidence grade | Notes                                    |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ | -------------- | ---------------------------------------- |
| LLM inference differs from training because it consists of many smaller online jobs with rapid scaling needs.                                                    | `llmsys-26-inference-vs-training`          | A              | Opening scale contrast.                  |
| Large-scale inference must consider variable lengths, multi-turn/agent workloads, heterogeneous hardware, fault tolerance, TCO, goodput, power, and KV hit rate. | `llmsys-26-scale-serving-challenges`       | A              | Scope framing.                           |
| TTFT and TPOT represent different latency constraints.                                                                                                           | `llmsys-29-ttft-tpot-slos`                 | A              | Metric definitions.                      |
| Goodput counts completed requests within SLO criteria, so high throughput can still be low goodput.                                                              | `llmsys-29-goodput-definition`             | A              | Chapter's central metric.                |
| Prefill is compute-bound while decode is memory-bound and needs many batched requests.                                                                           | `llmsys-29-prefill-decode-characteristics` | A              | Mechanism for disaggregation.            |
| Colocating prefill and decode can cause phase interference and coupled parallelism.                                                                              | `llmsys-29-colocation-interference`        | A              | Avoid "always" language.                 |
| Separating prefill and decode can let prefill optimize TTFT and decode optimize TPOT.                                                                            | `llmsys-29-disaggregation-opportunity`     | A              | Core design.                             |
| Disaggregation introduces KV-cache transfer overhead and a harder placement/goodput problem.                                                                     | `llmsys-29-disaggregation-challenges`      | A              | Tradeoff.                                |
| DistServe placement chooses parallelism, instance count, and physical placement.                                                                                 | `llmsys-29-distserve-placement`            | A              | No algorithm reproduction without paper. |
| Continuous batching targets utilization/throughput; disaggregation targets goodput under SLOs.                                                                   | `llmsys-29-cb-vs-disaggregation`           | A              | Bridge from Ch. 12.                      |
| Dynamo is presented as a modular stack with scheduling, disaggregated serving, memory management, and data transfer.                                             | `llmsys-26-dynamo-modular-stack`           | A              | Architecture example.                    |
| Dynamo-style disaggregated serving scales prefill and decode independently.                                                                                      | `llmsys-26-dynamo-disaggregated-serving`   | A              | Pair with DistServe.                     |
| KV-aware routing considers cache hit rate and worker load.                                                                                                       | `llmsys-26-kv-aware-routing`               | A              | No Baseten numbers.                      |
| KV offload can use HBM, host memory, local SSD, and network storage.                                                                                             | `llmsys-26-memory-tiers-kv-offload`        | A              | Memory hierarchy.                        |
| NIXL is presented as a cross-node/cross-memory transfer layer.                                                                                                   | `llmsys-26-nixl-transfer-layer`            | A              | API details need docs.                   |
| KV cache can be treated as reusable serving data.                                                                                                                | `llmsys-27-kv-cache-reuse`                 | A              | Bridge from Ch. 13.                      |
| KV cache can be treated as an AI-native data object, not just an internal tensor.                                                                                | `llmsys-27-kv-cache-ai-native-data`        | A              | Systems framing.                         |
| LMCache separates KV-cache management from inference engines.                                                                                                    | `llmsys-27-lmcache-separated-service`      | A              | External KV manager.                     |
| LMCache-style CPU sharing can reduce extra copies across GPU workers.                                                                                            | `llmsys-27-zero-copy-cpu-sharing`          | A              | Mechanism example only.                  |
| LMCache exposes storage plugin/get-put integration points.                                                                                                       | `llmsys-27-storage-plugin-interface`       | A              | Avoid exact API claims.                  |
| CacheBlend-style selective prefill reuses stored KV while recomputing selected tokens.                                                                           | `llmsys-27-cacheblend-selective-prefill`   | A              | Optional advanced sidebar.               |
| KV-cache compression can reduce stored cache size and transfer volume.                                                                                           | `llmsys-27-kv-cache-compression`           | A              | No ratio numbers.                        |
| Mooncake is presented as a KV-cache-centric disaggregated serving architecture.                                                                                  | `llmsys-28-mooncake-kvcache-centric`       | A              | Architecture example.                    |
| Mooncake also frames P/D disaggregation as avoiding mixed-batch interference.                                                                                    | `llmsys-28-pd-disaggregation-interference` | A              | Reinforces PD claim.                     |
| KV-cache caching creates storage and transfer-bandwidth challenges.                                                                                              | `llmsys-28-kvcache-storage-challenges`     | A              | No token-byte examples.                  |
| Mooncake uses distributed multi-layer KV-cache pools/storage.                                                                                                    | `llmsys-28-distributed-kv-pool`            | A              | No PB-scale claims without setup.        |
| Mooncake Store integrates engines with memory/storage resources through put/get and batch transfer abstractions.                                                 | `llmsys-28-mooncake-store-integration`     | A              | Conceptual interface only.               |

## Explanation Arc

1. Open by reframing the target from throughput to goodput under SLOs.
2. Define TTFT and TPOT as separate constraints.
3. Explain why continuous batching from Chapter 12 is necessary but insufficient.
4. Show why prefill and decode stress different resources and can interfere when colocated.
5. Introduce prefill/decode disaggregation: separate pools, separate SLO optimization, separate parallelism.
6. State the new cost: KV cache must move from prefill workers to decode workers.
7. Explain placement: how many prefill/decode instances, which parallelism, and where to place them.
8. Introduce cache-aware routing: route not only to free capacity but also to useful KV state.
9. Explain KV cache as external data: LMCache-style separation from inference engine.
10. Explain memory hierarchy: HBM, CPU DRAM, local SSD, network storage, remote KV pools.
11. Explain transfer substrate: NIXL/Mooncake-style movement across nodes and memory domains.
12. Close with the design lesson: serving architecture is a joint optimization over SLOs, cache locality, memory hierarchy, transfer bandwidth, and placement.

## Required Figures

| Figure ID                            | Purpose                                                                  | Form                      | Source                                                                           |
| ------------------------------------ | ------------------------------------------------------------------------ | ------------------------- | -------------------------------------------------------------------------------- |
| `fig-14-throughput-vs-goodput`       | Show raw throughput versus requests completed within SLO.                | Timeline/counting diagram | `llmsys-29-goodput-definition`                                                   |
| `fig-14-ttft-tpot`                   | Show first-token latency and inter-token latency on a streamed response. | Token timeline            | `llmsys-29-ttft-tpot-slos`                                                       |
| `fig-14-prefill-decode-interference` | Show colocated prefill/decode creating wasted time.                      | Two-lane timeline         | `llmsys-29-colocation-interference`                                              |
| `fig-14-pd-disaggregation`           | Show prefill pool, KV transfer, and decode pool.                         | Architecture diagram      | `llmsys-29-disaggregation-opportunity`, `llmsys-26-dynamo-disaggregated-serving` |
| `fig-14-placement-problem`           | Show choices for instance count, parallelism, and physical placement.    | Cluster diagram           | `llmsys-29-distserve-placement`                                                  |
| `fig-14-kv-aware-routing`            | Show router choosing workers by KV hit and load.                         | Routing diagram           | `llmsys-26-kv-aware-routing`                                                     |
| `fig-14-kv-memory-hierarchy`         | Show HBM, CPU DRAM, SSD, network storage tiers.                          | Memory hierarchy          | `llmsys-26-memory-tiers-kv-offload`, `llmsys-28-distributed-kv-pool`             |
| `fig-14-external-kv-manager`         | Show inference engines connected to LMCache/remote KV pool.              | Block diagram             | `llmsys-27-lmcache-separated-service`, `llmsys-27-zero-copy-cpu-sharing`         |
| `fig-14-transfer-substrate`          | Show transfer layer between memory domains and nodes.                    | Data-movement diagram     | `llmsys-26-nixl-transfer-layer`, `llmsys-28-mooncake-store-integration`          |

## Main Sections

### From Throughput to Goodput

Define goodput as requests completed within SLO criteria. Explain why a server can complete many requests but still deliver poor UX if TTFT or TPOT targets are missed.

### TTFT and TPOT Split the Latency Objective

Introduce TTFT and TPOT using streamed token output. Tie TTFT to prefill and queueing; tie TPOT to decode cadence, batching, and memory bandwidth.

### Continuous Batching Is Not the End

Review continuous batching as a utilization technique. Explain why it does not by itself solve prefill/decode interference or SLO-specific resource allocation.

### Prefill and Decode Want Different Systems

Explain compute-bound prefill versus memory-bound decode qualitatively. Show how colocation can force one resource/parallelism strategy onto both phases.

### Disaggregated Prefill and Decode

Explain separate prefill and decode pools. Prefill can optimize TTFT; decode can optimize TPOT. State that this is not free because KV cache must cross the boundary.

### Placement and Parallelism

Introduce DistServe-style placement: choose parallelism strategy, instance counts, and physical placement. Keep algorithms high-level unless adding paper-specific cards.

### KV-Aware Routing

Explain why routing must consider cache state. A free worker is not always the cheapest worker if another worker already has useful KV cache.

### KV Cache as External Data

Use LMCache to show the transition from in-engine KV libraries to separated KV-cache management services. Explain get/put, remote pools, and copy avoidance conceptually.

### Memory Tiers and Transfer

Explain HBM, host memory, SSD, and network storage as tiers. Introduce NIXL and Mooncake Store as transfer/integration examples without current API details.

### Advanced Cache Techniques

Briefly mention selective prefill/CacheBlend and KV-cache compression as further examples of treating KV cache as data. Keep them as sidebars unless deeper sources are added.

### Architecture Summary

Close with a combined loop: route request, decide prefill/decode placement, move or reuse KV cache, schedule decode, maintain SLOs, update cache.

## Technical Checks

- Do not say disaggregation always improves performance; state the workload, SLO, placement, and transfer tradeoffs.
- Do not quote goodput, TTFT, TPOT, TPS, throughput, cache-hit, or cost numbers without model, hardware, workload, SLO, and software context.
- Distinguish throughput from goodput.
- Distinguish TTFT from TPOT.
- Keep continuous batching and disaggregation complementary, not contradictory.
- Do not imply KV-aware routing should maximize cache hit rate without considering load and SLO.
- Do not present Dynamo, LMCache, or Mooncake as product recommendations.
- Do not describe exact NIXL/LMCache/Mooncake APIs unless official docs/source cards are added.
- Do not overuse "AI-native data" as marketing language; ground it in store/move/reuse/compress operations.

## Sidebar Decisions

- CacheBlend/selective prefill: optional sidebar.
- KV-cache compression: optional sidebar.
- kTransformers: likely out of scope except as a heterogeneous-serving pointer.
- Dynamo planner/perf configurator: optional sidebar only; avoid benchmark numbers.
- Mooncake evaluation: omit in first draft.
- DistServe evaluation: omit in first draft.

## Open Questions

- Should the brief add the DistServe paper as a paper source before draft?
- Should official docs/source-code cards be added for Dynamo, LMCache, Mooncake, or NIXL?
- How much detail should Chapter 14 include on transfer APIs versus keeping them conceptual?
- Should kTransformers be deferred entirely to Chapter 15's co-design discussion?

## Handoff

Owner: Book Architect  
Purpose: Chapter 14 brief from Dynamo, LMCache, Mooncake, and prefill/decode disaggregation source extraction  
Evidence grade: A for course lecture claims; no benchmark numbers used  
Assumptions: Chapter 14 is about distributed serving mechanisms and SLO-aware scheduling, not product evaluation  
Open questions: Whether to add DistServe paper and official system docs before draft  
Handoff: Systems Explainer for Chapter 14 draft
