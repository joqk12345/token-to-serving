---
status: ready
chapter: 13
slug: 13-kv-cache-vllm-pagedattention
title: KV Cache, PagedAttention, and vLLM
primary_sources:
  - llmsys-24-vLLM_woosuk_kwon-b6a0750bb310949461ba5a635a1126eb.pdf
papers:
  - https://arxiv.org/abs/2309.06180
secondary_sources: []
reader_level: engineer or graduate student who has read Chapters 1-12
technical_depth: intermediate-to-advanced
---

# KV Cache, PagedAttention, and vLLM

Chapter 12 described inference as an online cost model: prefill, decode, batching, scheduling, and KV-cache memory all interact. This chapter zooms into the memory term.

The important shift is simple:

```text
KV cache is not a temporary activation.
KV cache is persistent serving state.
```

During autoregressive decoding, each active request accumulates key and value tensors that later decode steps need. Those tensors live across model iterations. A serving engine therefore has to decide where they live, how they grow, when they can be shared, and what happens when the GPU runs out of available cache space.

The vLLM lecture states that attention KV cache stores per-token attention key/value state during inference. [CITE: llmsys-24-kv-cache-serving-state] It also frames efficient KV-cache management as crucial for high-throughput LLM serving. [CITE: llmsys-24-kv-cache-memory-management] The original PagedAttention paper makes the same problem central: dynamic KV-cache growth and fragmentation limit useful batching, and vLLM addresses that bottleneck with a new KV-cache memory-management design. [CITE: kwon-2023-pagedattention]

The core idea is to stop treating a request's KV cache as one large contiguous tensor reservation. PagedAttention splits KV cache into fixed-size blocks, maps a request's logical blocks to physical GPU-memory blocks, and teaches the attention kernel how to read through that mapping. [CITE: llmsys-24-pagedattention-definition]

## Why KV Cache Becomes the Serving Memory Problem

Model parameters are large, but they are mostly stable during serving. They are loaded once per worker or sharded across workers, then reused across requests.

KV cache is different. It grows with the live workload:

```text
more active requests      -> more KV state
longer prompts            -> more initial KV state after prefill
longer generations        -> more decode-time KV state
more layers / KV heads    -> more state per token
larger element precision  -> more bytes per state value
```

The vLLM lecture frames inference pressure as increasing with model size, context length, and generated-token count. [CITE: llmsys-24-inference-scaling-pressure] The memory consequence is that a server may have enough arithmetic capacity to continue decoding, but not enough useful KV-cache space to keep more requests resident.

This is why serving cannot be described as:

```text
parameters fit on GPU -> serving is solved
```

A more accurate serving memory ledger is:

```text
serving memory =
  model parameters
  + KV cache for active requests
  + temporary buffers
  + runtime and scheduler state
```

The KV-cache term changes every time requests arrive, finish, or generate another token.

## Why Contiguous Allocation Fails

A straightforward allocator can reserve one contiguous KV-cache region for each request. If the engine knows the request will never exceed a fixed maximum length, it can reserve enough space for that maximum.

That approach matches older static-shape deep learning workloads better than online text generation. In generation, output length is unknown at admission time. Two requests with the same maximum length may finish at very different actual lengths. Different requests may also have different prompt lengths and generation limits.

The vLLM lecture describes previous KV-cache management as preallocating contiguous memory to a request's maximum length, producing internal fragmentation from unknown output length and external fragmentation from non-uniform request maximums. [CITE: llmsys-24-contiguous-preallocation-fragmentation]

The waste has two forms:

```text
internal fragmentation:
  reserved slots inside a request allocation are never filled

external fragmentation:
  free memory exists, but not in the right contiguous shapes
```

For a serving engine, this is not just allocator ugliness. Fragmented KV memory reduces the number of active requests that can fit. Fewer active requests can reduce batching opportunities during decode. Worse batching can reduce throughput or increase queueing delay.

The lecture reports low KV-cache utilization for earlier systems while citing ORCA, but the slide does not carry enough experiment context to use the numeric range as a general book claim. [CITE: llmsys-24-kv-cache-utilization-caution] The safe conclusion is qualitative: contiguous maximum-length reservation is a poor fit for variable-length autoregressive serving.

## KV Blocks

PagedAttention changes the allocation unit. Instead of reserving one contiguous region for an entire request, it divides KV cache into fixed-size KV blocks. The lecture defines a KV block as a fixed-size block of memory that stores KV-cache state. [CITE: llmsys-24-kv-block-definition]

Conceptually:

```text
tokens in sequence order:
  t0 t1 t2 t3 | t4 t5 t6 t7 | t8 t9 ...

logical KV blocks:
  block 0     | block 1     | block 2 ...
```

The block size is an allocator choice. Smaller blocks can reduce unused space at the end of a sequence, but they can increase metadata and management overhead. Larger blocks reduce metadata pressure, but they can waste more space in the final partially filled block.

The useful property is that a request can grow one block at a time:

```text
request starts with prompt blocks
decode fills the current last block
when full, allocate one more physical block
```

This matches generation better than maximum-length reservation. The allocator does not need to know the final output length when the request arrives.

## Logical Blocks and Physical Blocks

Once a request is split into logical blocks, those logical blocks do not have to live in adjacent physical GPU-memory blocks.

PagedAttention introduces a block table. The vLLM lecture shows logical KV blocks mapped to physical KV blocks through that table. [CITE: llmsys-24-block-table-virtualization]

One request might look like this:

```text
logical block:   0   1   2   3
physical block:  7   4   9   1
filled tokens:   B   B   B   2
```

Here `B` means a full block. The request's sequence order is logical. The GPU-memory placement is physical. The block table translates from one to the other.

This is why the operating-system analogy is helpful. The lecture explicitly compares OS pages to KV blocks, and shared pages across processes to shared KV blocks across samples. [CITE: llmsys-24-os-virtual-memory-analogy]

But the analogy has limits:

```text
OS virtual memory:
  hardware-supported address translation
  general-purpose process memory
  page faults and OS-level policies

PagedAttention:
  application-level block table
  specialized KV-cache memory
  attention kernels that understand block indirection
  request-level preemption and recovery
```

The block table is not magic hardware paging. It is a serving-engine data structure that the attention implementation must respect.

## PagedAttention as a Kernel Contract

The allocator change only works if attention can read the new memory layout.

Standard attention wants the previous keys and values for the sequence. If those keys and values are stored in a contiguous tensor, the kernel can use ordinary contiguous indexing. With PagedAttention, the logical sequence is spread across non-contiguous physical KV blocks. The attention computation therefore needs the block table.

The vLLM lecture describes the PagedAttention mechanism as fetching non-contiguous KV blocks using the block table and applying attention on the fly. It also says PagedAttention is implemented as a custom GPU kernel to avoid materializing gathered keys and values. [CITE: llmsys-24-pagedattention-kernel]

The distinction matters. A naive implementation could gather physical blocks into a contiguous temporary buffer, then run ordinary attention. That would spend memory bandwidth and temporary memory to undo the allocator's layout. PagedAttention instead makes the layout part of the attention kernel's contract:

```text
attention input:
  query for current token
  block table for this request
  physical KV-block pool

attention behavior:
  follow logical blocks through the block table
  read physical K/V blocks in sequence order
  compute attention without building a contiguous K/V copy
```

This is the systems tradeoff. PagedAttention reduces allocator fragmentation and enables sharing, but it increases runtime complexity. The kernel has to handle indirection. The scheduler and KV-cache manager have to maintain block tables correctly. The engine has to coordinate memory allocation with request admission and decode iterations.

## Fragmentation After Paging

With fixed-size blocks, the main remaining internal waste is at the last block of a sequence. The lecture states that internal fragmentation only happens at the last block and that the number of wasted tokens per sequence is less than the block size. [CITE: llmsys-24-pagedattention-fragmentation]

Under that model:

```text
full logical blocks:
  no unused token slots

final logical block:
  may be partially filled
```

This does not mean PagedAttention eliminates all memory cost. It means the allocator has changed the shape of the waste. Instead of reserving every request up to a maximum length, the engine pays at block granularity as the request grows.

The practical design question becomes:

```text
block too small:
  more metadata, more allocation activity, possible transfer overheads

block too large:
  more final-block waste
```

A draft of this chapter should not pick a universal block size. The right value depends on implementation, model layout, attention backend, memory allocator, and workload.

## Sharing KV Blocks

Paged memory also makes sharing natural.

Suppose a server runs several continuations from the same prompt:

```text
prompt:
  "The future of cloud computing is ..."

sample A:
  prompt + continuation A

sample B:
  prompt + continuation B

sample C:
  prompt + continuation C
```

The prompt prefix has the same KV state for all three samples. Without sharing, each sequence may store duplicate prefix KV cache. With block-table indirection, several logical sequences can point to the same physical prefix blocks until their continuations diverge.

The vLLM lecture uses parallel sampling to show KV blocks shared across sequences. [CITE: llmsys-24-kv-block-sharing]

The mechanism is easier to see as references:

```text
physical blocks:
  P0 P1 P2 P3 P4 P5 ...

sample A logical blocks:
  P0 P1 P2 A3 A4 ...

sample B logical blocks:
  P0 P1 P2 B3 B4 ...

sample C logical blocks:
  P0 P1 P2 C3 C4 ...
```

The shared prefix blocks are stored once. The divergent suffix blocks are separate.

This is the same design pattern as the rest of PagedAttention: logical sequence identity is separated from physical memory ownership. The scheduler and cache manager can exploit that separation when prompts or decoding branches share a prefix.

## Memory Pressure, Preemption, and Recovery

Paged allocation does not make GPU memory infinite. At some point, the physical KV-block pool may be full.

When a new block is needed and no physical block is available, the serving engine has to choose a policy:

```text
wait:
  leave the request queued or stalled

preempt:
  free KV blocks from another request

reject or shed:
  apply admission control outside the model worker
```

The vLLM lecture focuses on request preemption and recovery. Its stated goal is to free some requests' KV cache so other requests can run first. It presents two recovery options: swapping KV cache to CPU memory and later swapping it back, or deleting KV cache and recomputing it later. [CITE: llmsys-24-preemption-recovery]

Swapping treats KV cache as data to move:

```text
GPU KV blocks -> CPU memory
CPU memory    -> GPU KV blocks when resumed
```

Recomputation treats KV cache as derived state:

```text
delete KV blocks
later rerun prompt/prefix computation
recreate KV cache
resume decode
```

Neither option is free. Swapping consumes host-device bandwidth and creates transfer scheduling work. Recomputation consumes GPU compute and can affect latency. The right policy depends on request length, model cost, interconnect, memory pressure, and scheduling goals.

The important point for this chapter is that KV-cache management becomes part of request scheduling. The scheduler is not only choosing which tokens to compute. It is choosing which requests deserve scarce persistent memory.

## vLLM Around PagedAttention

PagedAttention is the central mechanism in this chapter, but vLLM is an inference engine, not just an allocator.

The lecture presents vLLM as exposing both an offline Python `LLM` interface and an OpenAI-compatible server interface. [CITE: llmsys-24-vllm-api-surface] That API surface is not the main lesson, but it marks the boundary between user-facing serving and the engine internals.

Inside the engine, the lecture groups vLLM optimizations into four areas:

```text
minimizing CPU overheads
efficient GPU kernels
model parallelism
efficient memory management and caching
```

[CITE: llmsys-24-vllm-optimization-areas]

PagedAttention sits in the fourth category, but it interacts with the other three:

```text
CPU overhead:
  scheduler and KV-cache manager update block tables and batch metadata

GPU kernels:
  attention kernels must read non-contiguous blocks through indirection

model parallelism:
  KV cache may be partitioned or coordinated across devices

memory and caching:
  block allocation, sharing, preemption, and prefix reuse affect residency
```

The lecture also describes serving parallelism choices, including data, tensor, expert, context, pipeline parallelism, and prefill/decode disaggregation. [CITE: llmsys-24-vllm-parallelism-options]

Those topics matter, but they are not the center of this chapter. The boundary is:

```text
Chapter 13:
  how one engine manages KV cache as paged memory

Chapter 14:
  how serving systems schedule, cache, route, and disaggregate work across engines
```

## The Design Lesson

PagedAttention is useful to read as a memory-system design, not merely as an attention variant.

It changes four contracts at once:

```text
allocator contract:
  allocate fixed-size physical KV blocks on demand

sequence contract:
  represent each request as logical blocks

kernel contract:
  compute attention through block-table indirection

scheduler contract:
  account for KV-block ownership, sharing, and preemption
```

This is why vLLM belongs in a systems book. The improvement is not from one isolated trick. It comes from aligning the allocator, attention kernel, scheduler, and request lifecycle around the actual shape of autoregressive serving.

The common misconception is that inference serving is mostly about making `forward()` faster. Chapter 12 showed that serving cost includes scheduling and latency. This chapter shows that serving cost also includes memory address translation, fragmentation control, and cache residency.

The tradeoff to remember:

```text
PagedAttention exchanges simple contiguous KV tensors
for a paged KV-cache memory system.

That exchange can reduce fragmentation and enable sharing,
but it requires specialized kernels and careful runtime bookkeeping.
```

Owner: Principal Author  
Purpose: Chapter 13 ready draft after source extraction, brief, technical review, and red-team review  
Evidence grade: A for course lecture claims and original PagedAttention paper; no benchmark numbers used  
Assumptions: This draft focuses on single-engine KV-cache memory management and leaves distributed serving/disaggregation to Chapter 14  
Open questions: Whether to add a KV-cache byte formula, block-size waste derivation, or copy-on-write detail after a narrower source pass  
Handoff: Production can move to Chapter 14 source extraction
