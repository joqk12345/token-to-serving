---
status: brief
chapter: 10
slug: 10-zero-moe-and-memory
title: ZeRO, MoE, and Training Memory
primary_sources:
  - llmsys-17-MoE-3aa3125f9ccdd4bb7109ef077fbe9260.pdf
  - llmsys-18-zero-20eb6c8d8c1e7092e1b922abf03d8cdd.pdf
papers:
  - https://arxiv.org/abs/1910.02054
  - https://arxiv.org/abs/2101.03961
  - https://arxiv.org/abs/2006.16668
  - https://arxiv.org/abs/2201.05596
  - https://arxiv.org/abs/2401.06066
secondary_sources: []
reader_level: engineer or graduate student who has read Chapters 1-9
technical_depth: intermediate-to-advanced
---

# ZeRO, MoE, and Training Memory

## Chapter Thesis

Large-scale training is constrained by what each device must keep resident at each point in the step. ZeRO attacks replicated training state: optimizer states, gradients, and parameters. MoE attacks dense activation of model capacity: many parameters exist, but each token activates only a subset. Both techniques trade memory relief for communication, scheduling, and load-balancing complexity.

## Reader Problem

The reader now understands data parallelism and model parallelism. The remaining training-systems question is why neither technique alone solves memory at LLM scale. DDP replicates too much state. Pipeline and tensor parallelism split computation, but they do not by themselves remove all redundant optimizer, gradient, and parameter storage. MoE adds a different twist: it can increase total parameter capacity without activating every parameter for every token, but the router turns token placement into a systems problem.

## System Bottleneck

Primary bottlenecks: per-device memory capacity, optimizer-state replication, gradient storage, parameter residency, activation memory, temporary buffers, memory fragmentation, parameter transfer, expert all-to-all communication, router load balance, and memory bandwidth.

Secondary bottlenecks: precision assumptions, optimizer choice, worker count, micro-batch shape, sequence length, expert capacity, token skew, topology, communication scheduling, and kernel efficiency.

## Source Map

| Claim | Source card | Evidence grade | Notes |
|---|---|---|---|
| Mixed-precision DDP memory includes parameters, gradients, optimizer states, activations, temporary buffers, and fragmentation. | `llmsys-18-ddp-memory-accounting` | A | Use as opening memory ledger. |
| ZeRO partitions optimizer states, gradients, and parameters to remove DDP redundancy. | `llmsys-18-zero-key-idea` | A | Pair with original ZeRO paper. |
| Original ZeRO targets memory redundancy while preserving useful data/model-parallel execution properties. | `rajbhandari-2019-zero` | A | Avoid benchmark numbers. |
| ZeRO stage 1 partitions optimizer states. | `llmsys-18-zero-stage1-optimizer` | A | Explain first because optimizer states dominate Adam memory. |
| ZeRO stage 2 partitions gradients and retains only owned gradient shards. | `llmsys-18-zero-stage2-gradients` | A | Distinguish temporary buffers from retained shards. |
| ZeRO stage 3 partitions parameters and communicates parameter shards when needed. | `llmsys-18-zero-stage3-parameters` | A | Communication tradeoff must travel with claim. |
| Lecture memory formulas use `N`, `M`, and `K` assumptions. | `llmsys-18-zero-memory-formulas` | A | Include only as a boxed derivation with assumptions. |
| ZeRO-3 has additional parameter-transfer cost. | `llmsys-18-zero-communication-cost` | A | Avoid fixed multiplier unless conditions are stated. |
| ZeRO systems also manage activation checkpointing, buffers, fragmentation, and communication reduction. | `llmsys-18-zero-other-memory-optimizations` | A | Use as final ZeRO broadening. |
| Transformer MoE replaces a dense FFN with multiple expert FFNs and a router. | `llmsys-17-moe-ffn-router` | A | Define MoE mechanism. |
| Switch-style MoE routes tokens to selected experts. | `llmsys-17-switch-top1-routing` | A | Pair with Switch paper. |
| Switch Transformer frames sparse expert selection with complexity, communication, and instability concerns. | `fedus-2021-switch-transformer` | A | Avoid speed claims. |
| Expert parallelism places experts on different devices and requires all-to-all communication. | `llmsys-17-expert-parallelism` | A | Core MoE system bottleneck. |
| GShard combines conditional computation and automatic sharding for sparse MoE Transformers. | `lepikhin-2020-gshard` | A | Use for expert-parallel/autosharding context. |
| MoE training needs load balancing to avoid routing collapse. | `llmsys-17-moe-load-balancing` | A | Avoid formula unless notation is checked. |
| MoE inference depends on model size, activated experts, memory bandwidth, grouping, communication, and kernels. | `llmsys-17-moe-inference-bottlenecks` | A | Keep as preview; serving depth belongs later. |
| Expert parallelism creates all-to-all communication and may use optimized/hierarchical communication schedules. | `llmsys-17-moe-alltoall-optimization` | A | No topology-free latency law. |
| DeepSpeed-MoE treats MoE as an end-to-end training/inference systems problem. | `rajbhandari-2022-deepspeed-moe` | A | Avoid paper's performance numbers. |
| Shared-routed expert designs combine shared and routed experts. | `llmsys-17-shared-routed-experts` | A | Tie to DeepSeek-style design only at high level. |
| DeepSeekMoE proposes fine-grained experts and shared experts for specialization. | `dai-2024-deepseekmoe` | A | Optional sidebar. |

## Explanation Arc

1. Start from the memory ledger: DDP replicates parameters, gradients, optimizer state, activations, buffers, and fragmented allocations.
2. Separate two questions: "What state is duplicated?" and "Which computation is activated for each token?"
3. Explain ZeRO as state partitioning while preserving data-parallel training structure.
4. Walk through ZeRO-1, ZeRO-2, and ZeRO-3 as progressively partitioning optimizer states, gradients, and parameters.
5. Explain the memory formulas only with explicit `N`, `M`, `K`, and precision assumptions.
6. Make the ZeRO tradeoff explicit: less resident memory, more communication/scheduling for later stages.
7. Transition to MoE: instead of making every token use every FFN parameter, route tokens to selected expert FFNs.
8. Explain router, top-k/top-1 selection, expert FFNs, and shared/routed expert variants.
9. Explain expert parallelism: experts on different devices, non-expert layers replicated or handled by other parallelism axes.
10. Show all-to-all as the MoE systems bottleneck: tokens must be dispatched to expert owners and outputs returned.
11. Explain load balancing as both a learning and systems issue.
12. Close by comparing ZeRO and MoE: one partitions state; the other sparsifies activated compute; both require communication-aware runtime design.

## Required Figures

| Figure ID | Purpose | Form | Source |
|---|---|---|---|
| `fig-10-ddp-memory-ledger` | Show what a DDP worker stores. | Stacked memory ledger | `llmsys-18-ddp-memory-accounting` |
| `fig-10-zero-stages` | Compare ZeRO-1/2/3 partitioned state. | Three-row state partition diagram | `llmsys-18-zero-key-idea` |
| `fig-10-zero3-parameter-gather` | Show parameter shard availability during forward/backward. | Timeline/dataflow | `llmsys-18-zero-stage3-parameters` |
| `fig-10-moe-router-experts` | Show token routing from FFN input to selected experts. | Block diagram | `llmsys-17-moe-ffn-router` |
| `fig-10-expert-parallel-alltoall` | Show token dispatch and return across expert devices. | All-to-all diagram | `llmsys-17-expert-parallelism` |
| `fig-10-load-balance-skew` | Show router collapse versus balanced expert use. | Histogram or token flow | `llmsys-17-moe-load-balancing` |

## Main Sections

### The Memory Ledger After Model Parallelism

Reopen from Chapter 9: model parallelism partitions model computation, but DDP-style state replication remains a separate memory problem. Introduce the ledger of parameters, gradients, optimizer states, activations, temporary buffers, and fragmentation.

### ZeRO: Remove Redundant State

Define ZeRO as staged partitioning of data-parallel state. Emphasize that ZeRO is not the same as pipeline/tensor model parallelism: it changes which training state is resident on each data-parallel worker.

### ZeRO-1: Partition Optimizer States

Explain why Adam optimizer state can dominate memory. ZeRO-1 partitions optimizer states while parameters still exist for forward/backward.

### ZeRO-2: Partition Gradients

Explain that workers compute gradients from local data but do not need to retain full gradient copies. Gradients are reduced toward owner partitions.

### ZeRO-3: Partition Parameters

Explain parameter sharding and parameter availability. The key tradeoff: lower resident memory but parameter communication during forward/backward.

### The Formula Box

Optional boxed derivation using lecture variables:

```text
N = parameter count
M = optimizer bytes per parameter
K = data-parallel workers
```

Only include formulas if the prose states what is excluded: activations, temporary buffers, fragmentation, and implementation-specific overheads.

### MoE: Store Capacity, Activate Sparsely

Define MoE as replacing a dense FFN with multiple expert FFNs and a router. Explain that the total parameter count can be large while per-token activated compute is smaller than using all experts.

### Routing Is a Systems Problem

Explain top-k/top-1 routing, token-to-expert assignment, and why skew matters. The router affects model quality, communication load, and device utilization.

### Expert Parallelism and All-to-All

Explain expert placement across devices. Non-expert layers may be replicated or handled by tensor/data parallelism; expert layers require token exchange. All-to-all is the core communication pattern.

### Load Balancing

Explain routing collapse and why auxiliary/device-level balancing exists. Keep formulas out of the main draft unless notation is verified against the original papers.

### Inference Preview

Briefly state that MoE inference is also memory-bandwidth and routing sensitive. Do not turn Chapter 10 into a serving chapter; Part V handles serving architecture.

### What to Remember

ZeRO reduces replicated state. MoE reduces dense activation. Both make bigger training runs possible only by moving work into communication, scheduling, and runtime memory management.

## Technical Checks

- Memory accounting: Any formula must state `N`, `M`, `K`, precision, and exclusions.
- ZeRO stage semantics: Do not imply ZeRO-1/2/3 are just compression; they are partitioning strategies.
- Communication: Do not claim ZeRO-3 is free; parameter transfer is the tradeoff.
- MoE sparsity: Do not claim total parameters equal activated parameters.
- MoE performance: No speed, latency, throughput, or cost claims without model/hardware/batch/sequence/precision/source context.
- Router/load balance: Avoid implying routing is only a model-quality issue; it also controls system load.
- Chapter boundary: Keep detailed serving, vLLM, and KV-cache scheduling for Part V.

## Sidebar Decisions

- DeepSeekMoE: optional sidebar on fine-grained and shared experts; avoid DeepSeek-V3 numeric architecture claims unless separately sourced.
- ZeRO++: mention only as a pointer if needed; do not explain quantized communication deeply without original source extraction.
- CPU/NVMe offload: leave out or one cautionary sentence unless adding source cards.
- Memory formulas: include as a box only if the draft can carry assumptions.

## Open Questions

- Should the main draft include the ZeRO memory formulas, or leave them for a boxed derivation?
- Should DeepSeek-style shared/fine-grained experts be a sidebar or part of the main MoE arc?
- Should Chapter 10 mention PyTorch FSDP as a modern API analogue, or avoid it until official docs/source cards are added?
- How much inference material belongs here before overlapping with Part V?

## Handoff

Owner: Book Architect  
Purpose: Chapter 10 brief from ZeRO and MoE source extraction  
Evidence grade: A for course lecture claims and primary papers; no benchmark numbers used  
Assumptions: Chapter 10 explains training memory/state partitioning and sparse expert activation; serving depth remains Part V  
Open questions: Formula inclusion and DeepSeek/FSDP sidebar scope  
Handoff: Systems Explainer for Chapter 10 draft
