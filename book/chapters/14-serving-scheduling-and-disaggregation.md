---
status: ready
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

Chapter 12 built an inference cost model. Chapter 13 showed why KV cache becomes a memory system inside an inference engine. This chapter moves up one level: distributed serving.

At this level, the question is no longer:

```text
How do I keep one model worker busy?
```

It is:

```text
Which request should go to which worker?
Where is the useful KV state?
Should prefill and decode run on the same GPUs?
How much KV cache should move across the network?
Which requests meet their latency SLOs?
```

The shift is from throughput to goodput. A system can complete many requests per second and still be a poor serving system if too many requests miss user-facing latency targets.

The disaggregation lecture defines goodput as completed requests within SLO criteria and contrasts it with raw throughput. [CITE: llmsys-29-goodput-definition] This chapter uses that distinction as the organizing principle.

## Inference at Scale Is Not Training at Scale

Training usually runs a coordinated job. It has a known model, a known parallelism plan, long-running workers, and a steady stream of batches.

Serving is less regular. The Dynamo lecture contrasts training with inference by describing inference as many smaller online jobs with rapid scale-up and scale-down requirements. [CITE: llmsys-26-inference-vs-training]

At scale, the serving system must handle:

```text
variable input lengths
variable output lengths
multi-turn sessions
agentic workflows
heterogeneous hardware
fault tolerance
multiple models
cost and power constraints
KV-cache hit rate
```

The Dynamo lecture lists these as part of the challenge of serving AI inference at scale. [CITE: llmsys-26-scale-serving-challenges]

This is why the serving scheduler becomes a distributed systems component. It is not just selecting the next batch for one GPU. It is making placement, routing, memory, and transfer decisions under SLOs.

## TTFT, TPOT, and Goodput

A streamed LLM response has at least two latency surfaces.

```text
TTFT:
  time to first token
  how long the user waits before output begins

TPOT:
  time per output token
  cadence of streamed tokens after generation starts
```

The disaggregation lecture defines TTFT as time to first token and TPOT as time per output token. [CITE: llmsys-29-ttft-tpot-slos]

These metrics are not interchangeable. A user may tolerate a slower first token for a batch summarization job but notice uneven token streaming in a chat UI. Another application may need the first token quickly but produce only a short output.

Goodput adds the SLO constraint:

```text
throughput:
  completed requests / time

goodput:
  completed requests that satisfy SLO / time
```

The useful serving system is the one that completes enough requests while meeting the relevant TTFT and TPOT constraints.

This reframes scheduling. A scheduler that maximizes tokens per second can still route work poorly if it creates long first-token delays, uneven decode cadence, or cache misses that force repeated prefill.

## Continuous Batching Is Necessary but Not Sufficient

Chapter 12 introduced continuous batching: update batch membership between decode iterations so finished requests can leave and new requests can enter. Continuous batching improves utilization under variable output lengths.

But it does not solve every serving-scale problem. The disaggregation lecture states the distinction directly: continuous batching improves utilization and throughput, while disaggregation targets goodput under SLOs. [CITE: llmsys-29-cb-vs-disaggregation]

Continuous batching still leaves several questions open:

```text
Should a long prefill share a worker with latency-sensitive decode?
Should all requests use the same tensor/pipeline/data parallelism?
Should a request route to the least-loaded worker or the worker with its KV prefix?
When should KV cache move to another worker or memory tier?
```

These questions are outside a single batch loop. They require serving architecture.

## Prefill and Decode Want Different Systems

Prefill and decode are both Transformer inference, but they stress the system differently.

The disaggregation lecture characterizes prefill as compute-bound and decode as memory-bound, with decode needing many batched requests to saturate compute. [CITE: llmsys-29-prefill-decode-characteristics]

The reason is the shape of work:

```text
prefill:
  processes many prompt tokens at once
  uses larger matrix operations
  builds initial KV cache
  strongly affects TTFT

decode:
  advances one generated token per request per iteration
  repeatedly reads parameters and KV cache
  benefits from batching many active requests
  strongly affects TPOT
```

If prefill and decode share one worker pool, the scheduler must decide how to mix them. A long prefill can consume compute resources while decode requests wait. Decode-heavy traffic can delay new prefills and increase first-token latency.

The disaggregation lecture describes this as prefill/decode interference and coupled parallelism strategy when both phases are colocated. [CITE: llmsys-29-colocation-interference]

Colocation is not always wrong. It is simpler and avoids explicit KV transfer between pools. But under workloads with distinct TTFT/TPOT targets and enough traffic to justify specialization, colocation can force one resource plan onto two different phases.

## Disaggregated Prefill and Decode

Prefill/decode disaggregation separates the two phases into different worker pools:

```text
request arrives
  -> prefill worker processes prompt
  -> prefill worker produces KV cache
  -> KV cache moves or becomes visible to decode worker
  -> decode worker streams output tokens
```

The disaggregation lecture describes the opportunity: prefill instances optimize TTFT, decode instances optimize TPOT, and each phase can choose suitable parallelism and resource allocation. [CITE: llmsys-29-disaggregation-opportunity]

Dynamo's lecture presents the same idea as scaling prefill and decode independently on separate GPUs and applying suitable parallelism for each phase. [CITE: llmsys-26-dynamo-disaggregated-serving]

The mechanism is attractive because the serving system can choose different answers for different bottlenecks:

```text
prefill pool:
  provision for prompt bursts and first-token latency
  choose parallelism that helps long prompt processing

decode pool:
  provision for steady token streaming
  choose batching and memory layout that helps TPOT
```

The cost is that KV cache crosses a boundary. A decode worker cannot continue generation unless it can access the keys and values created during prefill.

The disaggregation lecture names this directly: disaggregation introduces KV-cache transmission overhead and makes per-GPU goodput hard to optimize because workload pattern, SLOs, parallelism, resource allocation, and network bandwidth all matter. [CITE: llmsys-29-disaggregation-challenges]

The design question is therefore not "disaggregate or not" in the abstract. It is:

```text
does the SLO benefit from specialization exceed
the placement and KV-transfer cost for this workload?
```

## Placement Is Part of the Algorithm

Once prefill and decode are separate, the system has to decide how many of each to run and where to put them.

DistServe-style placement is framed as choosing:

```text
parallelism strategy for prefill instances
parallelism strategy for decode instances
number of each instance
physical cluster placement
```

The disaggregation lecture describes this as a placement problem for maximizing GPU goodput under workload requirements. [CITE: llmsys-29-distserve-placement]

This is where distributed systems details become load-bearing. If prefill and decode workers are placed far apart in the network topology, KV transfer can become expensive. If they are too tightly colocated, the system may lose some flexibility. If too many GPUs are assigned to prefill, decode may miss TPOT. If too many are assigned to decode, requests may wait too long for their first token.

Placement is not a one-time math exercise. Traffic changes. Prompt lengths change. Output lengths change. Hardware availability changes. A production serving system therefore needs monitoring and adaptation, even if the chapter keeps the control algorithms out of scope.

## KV-Aware Routing

Routing cannot look only at load.

Suppose two workers are available:

```text
worker A:
  low current load
  no useful KV state

worker B:
  moderate current load
  already has the request's shared prefix KV cache
```

The cheapest route may be B, not A, if reusing KV cache avoids expensive prefill or transfer. But if B is overloaded, cache locality may not compensate for queueing delay.

The Dynamo lecture presents KV-aware routing as considering KV hit rate and worker KV load. [CITE: llmsys-26-kv-aware-routing]

The scheduler is balancing at least three terms:

```text
queueing:
  how busy is the worker?

cache locality:
  does the worker or nearby store already have useful KV?

SLO risk:
  will this route preserve TTFT and TPOT?
```

This is the distributed version of the Chapter 13 lesson. KV cache is not just memory usage. It is routing state.

## KV Cache Becomes External Data

Chapter 13 treated KV cache as a paged memory object inside an engine. Chapter 14 treats KV cache as data that may live outside one inference process.

The LMCache lecture says KV cache can be treated as reusable serving data rather than merely internal tensors. [CITE: llmsys-27-kv-cache-ai-native-data] It also states that KV cache avoids repeated computation by storing reusable attention state. [CITE: llmsys-27-kv-cache-reuse]

LMCache is presented as separating KV-cache management from inference engines by running as a separate KV-cache management service. [CITE: llmsys-27-lmcache-separated-service]

The architectural move is:

```text
before:
  inference engine owns KV cache internally

after:
  inference engine can get/put KV cache through an external manager
```

This separation gives the serving system more options:

```text
share cache across workers
offload cache outside GPU memory
reuse cache across sessions or repeated prefixes
compress cache for storage or transfer
integrate with different storage backends
```

The LMCache lecture presents hooks and storage plugins for get/put KV-cache operations. [CITE: llmsys-27-storage-plugin-interface] It also illustrates a multi-process CPU pool and remote KV pool to reduce extra CPU-side copies when sharing cache across GPU workers. [CITE: llmsys-27-zero-copy-cpu-sharing]

The exact APIs are implementation details and can change. The durable concept is the boundary: KV cache is becoming a managed object with a storage and transfer interface.

## Memory Tiers and Transfer

GPU HBM is fast and scarce. KV cache may be too large or too reusable to keep only in HBM.

The Dynamo lecture presents KV offload across HBM, host memory, local SSD, and network storage. [CITE: llmsys-26-memory-tiers-kv-offload]

The hierarchy looks like this:

```text
HBM:
  fastest, smallest, active decode state

CPU DRAM:
  larger, slower, useful for nearby reuse/offload

local SSD:
  larger again, useful for colder cache or spill

network storage / remote pool:
  shared, potentially large, transfer-sensitive
```

Mooncake makes the same design point from a KV-cache-centric architecture. It presents distributed multi-layer KV-cache pools/storage and cache capacity beyond one machine. [CITE: llmsys-28-distributed-kv-pool]

But moving cache through the hierarchy has a cost. Mooncake's lecture states that KV-cache caching creates storage challenges because cache size and transfer bandwidth matter. [CITE: llmsys-28-kvcache-storage-challenges]

This is the core tradeoff:

```text
store more KV cache:
  more reuse opportunities
  less recomputation

move more KV cache:
  more bandwidth demand
  more latency risk
```

The serving system has to decide which KV state is hot enough to keep near the GPU and which state can live in colder tiers.

## Transfer Substrates

Once KV cache moves across workers, memory tiers, and nodes, the transfer mechanism becomes part of serving performance.

The Dynamo lecture presents NIXL as a cross-node and cross-memory transfer layer for buffer lists, with northbound and southbound APIs. [CITE: llmsys-26-nixl-transfer-layer]

Mooncake Store is presented as integrating inference engines with local memory, remote memory, SSD, and third-party storage through object put/get and batch transfer abstractions. [CITE: llmsys-28-mooncake-store-integration]

This chapter does not depend on the exact APIs. The mechanism to remember is:

```text
KV cache movement is not incidental I/O.
It is on the serving critical path when routing,
prefill/decode disaggregation, or cache reuse requires it.
```

A transfer layer has to coordinate memory registration, remote access, batching, and backend differences without making the decode worker wait unnecessarily.

## Mooncake and Dynamo as Architecture Examples

Dynamo is presented as a modular distributed inference stack with scheduling, disaggregated serving, memory management, and data transfer components. [CITE: llmsys-26-dynamo-modular-stack]

Mooncake is presented as a KV-cache-centric disaggregated architecture with cache-aware prefill scheduling, KV-cache pool, KV-cache balancing, and decode scheduling. [CITE: llmsys-28-mooncake-kvcache-centric]

Mooncake's lecture also frames prefill/decode disaggregation as a way to avoid interference in mixed batches and decouple resources and parallelism. [CITE: llmsys-28-pd-disaggregation-interference]

The names are less important than the pattern:

```text
request router
prefill scheduler and workers
KV-cache manager / store
transfer substrate
decode scheduler and workers
SLO-aware planner / controller
```

This is the serving architecture that emerges when KV cache is large, reusable, movable, and latency-sensitive.

## Advanced Cache Techniques

Once KV cache is external data, more transformations become possible.

The LMCache lecture presents CacheBlend-style selective prefill: reuse stored KV cache while recomputing selected tokens to recover interactions that direct cache concatenation would miss. [CITE: llmsys-27-cacheblend-selective-prefill]

It also presents KV-cache compression as a way to store more cache and reduce transfer volume. [CITE: llmsys-27-kv-cache-compression]

These techniques are optional for the first-pass chapter, but they show the direction of travel:

```text
KV cache can be:
  routed to
  shared
  offloaded
  transferred
  selectively recomputed
  compressed
```

That list would make little sense if KV cache were only an internal tensor inside one worker. It makes sense once KV cache is part of the serving data plane.

## The Design Lesson

Chapter 12 showed that inference cost is not one forward pass. Chapter 13 showed that KV cache needs a memory system. Chapter 14 adds the distributed systems layer:

```text
goodput:
  meet SLOs, not just maximize completions

disaggregation:
  specialize prefill and decode, but pay KV-transfer cost

routing:
  balance load, cache locality, and SLO risk

memory hierarchy:
  decide which KV state belongs in HBM, DRAM, SSD, or remote storage

transfer:
  make KV movement a first-class serving operation
```

The common misconception is that serving architecture is just horizontal scaling: add more replicas behind a load balancer. That works only when requests are independent and stateless. LLM serving is not stateless. KV cache creates placement history, cache locality, and transfer costs.

The tradeoff to remember:

```text
distributed serving buys specialization and reuse,
but it turns KV cache into a data-management problem.
```

Owner: Principal Author  
Purpose: Chapter 14 ready draft after source extraction, brief, technical review, and red-team review  
Evidence grade: A for course lecture claims; no benchmark numbers used  
Assumptions: Draft explains mechanisms and tradeoffs without product evaluation or current API guarantees  
Open questions: Whether to add DistServe paper, official system docs, or deeper transfer API cards before technical review  
Handoff: Production can move to Chapter 15 source extraction
