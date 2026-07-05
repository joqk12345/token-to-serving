---
status: brief
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

## Chapter Thesis

Serving throughput depends on treating KV cache as a first-class memory system. PagedAttention improves the memory allocator and attention kernel contract by splitting KV cache into fixed-size blocks, mapping logical sequence blocks to physical GPU-memory blocks, sharing blocks when sequences share prefixes, and recovering from memory pressure through request-level preemption.

## Reader Problem

Chapter 12 gave the reader the inference cost model: prefill, decode, batching, scheduler overhead, and KV-cache memory. The next problem is sharper: if every active request grows persistent KV state during decode, a serving engine can run out of useful GPU memory before it runs out of arithmetic. The reader needs to understand why ordinary contiguous tensor allocation is a poor fit for variable-length generation, and how vLLM/PagedAttention recasts attention cache as a paged memory object.

## System Bottleneck

Primary bottlenecks: GPU KV-cache capacity, memory fragmentation, block allocation granularity, attention-kernel indirection, prefix sharing, request preemption, recomputation or swapping cost, scheduler/KV-cache-manager coordination, and CPU overhead in a high-token-rate decode loop.

Secondary bottlenecks: block size, cache hit rate for shared prefixes, maximum sequence length, output length uncertainty, model-parallel layout, attention backend, CUDA graph constraints, kernel fusion, and multi-GPU communication.

## Source Map

| Claim | Source card | Evidence grade | Notes |
|---|---|---|---|
| Inference pressure increases with model size, context length, and generated-token count. | `llmsys-24-inference-scaling-pressure` | A | Qualitative motivation only. |
| KV cache stores per-token key/value state across Transformer layers during inference. | `llmsys-24-kv-cache-serving-state` | A | Reconnect to Ch. 12. |
| Efficient KV-cache management is central to high-throughput serving. | `llmsys-24-kv-cache-memory-management` | A | Avoid unsourced memory breakdowns. |
| Contiguous preallocation to maximum request length can create internal and external fragmentation. | `llmsys-24-contiguous-preallocation-fragmentation` | A | Core problem statement. |
| The lecture reports low KV-cache utilization in earlier systems, citing ORCA. | `llmsys-24-kv-cache-utilization-caution` | A | Do not use the numeric range as a general claim without original experiment conditions. |
| PagedAttention applies application-level paging and virtualization to attention KV cache. | `llmsys-24-pagedattention-definition` | A | Central mechanism. |
| A KV block is a fixed-size memory block for KV-cache state. | `llmsys-24-kv-block-definition` | A | Define before block table. |
| Logical KV blocks map to physical KV blocks through a block table. | `llmsys-24-block-table-virtualization` | A | Main data structure. |
| PagedAttention fetches non-contiguous KV blocks through the block table and avoids materializing gathered K/V tensors. | `llmsys-24-pagedattention-kernel` | A | Kernel/runtime contract. |
| Internal fragmentation is limited to a final partially filled block under the block allocator model. | `llmsys-24-pagedattention-fragmentation` | A | State conditions; derive only with block-size assumptions. |
| KV blocks can be shared across related sequences with shared prefixes. | `llmsys-24-kv-block-sharing` | A | Use parallel sampling/shared prompt as example. |
| When physical KV blocks are exhausted, the system can preempt requests through swapping or recomputation. | `llmsys-24-preemption-recovery` | A | No latency numbers. |
| OS virtual memory is a useful but limited analogy for PagedAttention. | `llmsys-24-os-virtual-memory-analogy` | A | Explicitly state differences. |
| vLLM exposes offline and online serving APIs. | `llmsys-24-vllm-api-surface` | A | Mention only as system boundary. |
| vLLM optimizes CPU overhead, GPU kernels, model parallelism, and memory/caching. | `llmsys-24-vllm-optimization-areas` | A | Keep PagedAttention central. |
| Serving parallelism choices trade communication, load balance, and memory. | `llmsys-24-vllm-parallelism-options` | A | Defer detailed disaggregation to Ch. 14. |
| The original PagedAttention paper frames dynamic KV-cache growth and fragmentation as the serving bottleneck addressed by vLLM. | `kwon-2023-pagedattention` | A | Publication-level support; avoid benchmark numbers without experiment cards. |

## Explanation Arc

1. Open from the Chapter 12 cost model: decode keeps active requests alive and grows persistent KV-cache state.
2. Show why KV cache is not just "activation memory": it survives across decode steps and must be indexed by request and position.
3. Explain why maximum-length contiguous allocation wastes memory under unknown output length and heterogeneous request lengths.
4. Introduce the allocator design problem: the system wants dynamic growth, compact physical memory use, and fast attention reads.
5. Define KV blocks as fixed-size units and distinguish logical sequence order from physical GPU-memory placement.
6. Introduce the block table as the request-local address translation layer.
7. Explain PagedAttention: attention reads K/V through block-table indirection rather than requiring a contiguous cache tensor.
8. Explain the tradeoff: reduced fragmentation and more flexible sharing, but more complex attention kernels and memory indirection.
9. Explain block sharing for common prefixes, parallel sampling, and related continuations.
10. Explain what happens under KV-block exhaustion: request-level preemption, swapping, or recomputation.
11. Place PagedAttention inside vLLM's larger engine: API layer, scheduler/KV-cache manager, GPU kernels, and model parallelism.
12. Close with the design lesson: serving engines are memory managers and schedulers as much as model executors.

## Required Figures

| Figure ID | Purpose | Form | Source |
|---|---|---|---|
| `fig-13-kv-cache-growth` | Show KV state accumulating per generated token across layers. | Memory growth diagram | `llmsys-24-kv-cache-serving-state` |
| `fig-13-contiguous-fragmentation` | Show internal and external fragmentation from max-length contiguous reservations. | Memory strip | `llmsys-24-contiguous-preallocation-fragmentation` |
| `fig-13-kv-blocks` | Show a sequence split into fixed-size logical KV blocks. | Block diagram | `llmsys-24-kv-block-definition` |
| `fig-13-block-table` | Show logical blocks mapped to non-contiguous physical blocks. | Address-translation diagram | `llmsys-24-block-table-virtualization` |
| `fig-13-pagedattention-kernel` | Show attention reading non-contiguous blocks through block-table indirection. | Dataflow | `llmsys-24-pagedattention-kernel` |
| `fig-13-fragmentation-boundary` | Show waste limited to the final partially filled block. | Before/after memory diagram | `llmsys-24-pagedattention-fragmentation` |
| `fig-13-prefix-sharing` | Show multiple samples sharing prompt-prefix KV blocks before divergence. | Tree/block reference diagram | `llmsys-24-kv-block-sharing` |
| `fig-13-preemption-recovery` | Compare swap-to-CPU and recompute recovery paths. | Two-lane flow | `llmsys-24-preemption-recovery` |
| `fig-13-vllm-engine-boundary` | Place PagedAttention inside vLLM's API, scheduler, KV manager, worker, and kernels. | Architecture block diagram | `llmsys-24-vllm-optimization-areas` |

## Main Sections

### KV Cache Becomes a Memory System

Reintroduce KV cache from Chapter 12. Emphasize that the cache is persistent request state, not a temporary activation that disappears after a layer or batch step.

### Why Contiguous Allocation Fails

Explain maximum-length reservation, unknown output length, internal fragmentation, and external fragmentation. Keep numeric utilization claims out of main prose unless experiment conditions are supplied.

### KV Blocks

Define fixed-size KV blocks. Explain allocation granularity, final-block waste, and why the example block size in the lecture is illustrative rather than prescriptive.

### Logical Blocks and Physical Blocks

Introduce the block table. Use the virtual-memory analogy carefully: logical sequence order is stable while physical GPU blocks can be non-contiguous.

### PagedAttention as a Kernel Contract

Explain that the attention kernel must read K/V through block-table indirection. State the benefit and cost: no materialized gathered K/V buffer, but specialized kernels and indirection complexity.

### Fragmentation and Sharing

Explain why block allocation reduces fragmentation and why shared prefix blocks reduce duplicate memory for parallel samples or common prompts.

### Memory Pressure, Preemption, and Recovery

Explain what happens when the physical block pool is exhausted. Compare swapping and recomputation at the mechanism level; do not quote latency numbers.

### vLLM Beyond PagedAttention

Place PagedAttention inside the vLLM engine. Mention offline/online APIs, scheduler/KV-cache manager, CPU-overhead minimization, kernels, and model parallelism as surrounding system components.

### Boundary to Distributed Serving

Name data/tensor/expert/context/pipeline parallelism and prefill/decode disaggregation only as boundary markers. Detailed routing, LMCache, Dynamo, Mooncake, and disaggregated serving belong in Chapter 14.

## Technical Checks

- Do not say PagedAttention eliminates all memory waste; say it reduces fragmentation under a fixed-size block allocator and may still waste space in final partially filled blocks.
- Do not use the slide's KV utilization range as a general claim without original ORCA experiment conditions.
- Do not quote PagedAttention/vLLM speedups without model, GPU, workload trace, decoding algorithm, baseline, batch/request mix, and implementation version.
- Do not imply that a block table is identical to a hardware page table; state the analogy and differences.
- Do not hide the kernel cost: block-table indirection requires attention kernels that understand non-contiguous KV layout.
- Do not turn the chapter into a vLLM product manual or API tutorial.
- Keep disaggregated prefill/decode and multi-system caching for Chapter 14.
- If adding formulas for KV-cache bytes, define layers, KV heads, head dimension, tokens, batch/request count, dtype bytes, and sharding.

## Sidebar Decisions

- OS virtual memory analogy: main prose, with a small "where the analogy breaks" box.
- Original vLLM paper: use for publication-level support, but no benchmark table in first draft.
- vLLM APIs: one short paragraph only.
- Piecewise CUDA graphs / kernel backend abstractions: optional sidebar only if the draft needs to show vLLM is more than PagedAttention.
- Hybrid memory allocator: defer unless Chapter 13 later expands into modern vLLM internals.

## Open Questions

- Should the first draft include a symbolic KV-cache byte formula, or defer until a dedicated formula card is created?
- Should block size be discussed qualitatively only, or should a boxed derivation show the final-block waste bound?
- Should copy-on-write for shared KV blocks be explained now using the original paper, or kept as a later refinement?
- How much vLLM engine architecture should appear before overlapping Chapter 14?

## Handoff

Owner: Book Architect  
Purpose: Chapter 13 brief from vLLM/PagedAttention source extraction  
Evidence grade: A for course lecture claims and original PagedAttention paper; no benchmark numbers used  
Assumptions: Chapter 13 focuses on single-engine KV-cache memory management; distributed serving remains Chapter 14  
Open questions: KV-cache byte formula, block-size depth, and copy-on-write detail  
Handoff: Systems Explainer for Chapter 13 draft
