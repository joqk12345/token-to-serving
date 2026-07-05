---
status: brief
chapter: 8
slug: 08-distributed-training-ddp
title: Distributed Training and Data Parallelism
primary_sources:
  - llmsys-14-distributed-training-b27c3d1dc185e680c6f5cc924e9ec9d7.pdf
  - llmsys-15-ddp-165dbe3873fac21eb8b339e64bcfee28.pdf
secondary_sources: []
reader_level: engineer or graduate student who has read Chapters 1-7
technical_depth: intermediate
---

# Distributed Training and Data Parallelism

## Chapter Thesis

Data parallelism scales training only when gradient synchronization is engineered as a communication schedule: all-reduce semantics, ring phases, bucketing, autograd hooks, and overlap determine whether extra GPUs reduce step time or just wait on the network.

## Reader Problem

The reader knows how a single-device training step becomes compiled tensor work. The missing step is what changes when the same model is replicated across many GPUs. The naive answer is “average gradients,” but the systems problem is when, how often, through which collective, and overlapped with which computation.

## System Bottleneck

Primary bottlenecks: communication bandwidth, communication latency, synchronization barriers, gradient bucket scheduling, backward/communication overlap, and straggler sensitivity.

Secondary bottlenecks: process setup, rank/device mapping, NCCL stream behavior, bucket size tuning, and optimizer-state consistency.

## Source Map

| Claim                                                                                                                                | Source card                               | Evidence grade | Notes                                               |
| ------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------- | -------------- | --------------------------------------------------- |
| NCCL provides multi-GPU communication primitives.                                                                                    | `llmsys-14-nccl-communication`            | A              | Official NCCL docs needed for exact API guarantees. |
| Core collectives include broadcast, reduce, reduce-scatter, all-gather, and all-reduce.                                              | `llmsys-14-collective-primitives`         | A              | Define before DDP.                                  |
| All-reduce reduces across ranks and makes the result available to all ranks; it can be decomposed as reduce-scatter plus all-gather. | `llmsys-14-allreduce-semantics`           | A              | Central communication primitive.                    |
| Data-parallel training partitions data, computes local gradients, all-reduces average gradients, and updates parameters locally.     | `llmsys-14-data-parallel-allreduce`       | A              | Main algorithmic structure.                         |
| Ring all-reduce uses scatter-reduce and all-gather phases over chunks.                                                               | `llmsys-14-ring-allreduce-phases`         | A              | Mechanism only; no speedup claims.                  |
| Parameter-server and all-reduce data parallelism differ in where parameter updates and synchronization occur.                        | `llmsys-14-parameter-server-vs-allreduce` | A              | Conceptual contrast only.                           |
| DDP replicates models across nodes/GPUs, runs forward/backward independently, averages gradients, and runs local optimizers.         | `llmsys-15-ddp-replica-gradient-average`  | A              | Bridge from all-reduce to PyTorch DDP.              |
| DDP design goals include non-intrusive local-script reuse and interceptive internal optimization.                                    | `llmsys-15-ddp-design-goals`              | A              | Add Li et al. paper before draft if emphasized.     |
| Naive DDP would all-reduce after the entire backward pass.                                                                           | `llmsys-15-ddp-naive-allreduce`           | A              | Opening bottleneck.                                 |
| DDP groups gradients into buckets instead of synchronizing every parameter separately.                                               | `llmsys-15-ddp-bucketing`                 | A              | Main scheduling mechanism.                          |
| DDP overlaps backward computation with asynchronous all-reduce when buckets are ready.                                               | `llmsys-15-ddp-overlap`                   | A              | Performance claim must stay conditional.            |
| DDP uses autograd hooks/reducer logic to notice ready gradients and trigger bucket all-reduce.                                       | `llmsys-15-ddp-autograd-hooks`            | A              | Framework-internals connection.                     |

## Explanation Arc

1. Concrete problem: one GPU trains too slowly; replicate the model across GPUs and split the batch.
2. Minimal data-parallel step: each worker computes gradients on its data shard.
3. Correctness requirement: workers must agree on the gradient update, usually by averaging gradients before local optimizer steps.
4. Communication primitive: define all-reduce and decompose it into reduce-scatter plus all-gather.
5. Communication schedule: explain ring all-reduce as chunks moving around a ring.
6. Baseline DDP: all-reduce all gradients after backward; simple but leaves communication on the critical path.
7. DDP optimization: use buckets and autograd hooks so ready gradients launch asynchronous all-reduce during backward.
8. Tradeoff: bucket size, parameter order, model shape, and network determine how much overlap is available.
9. Boundary to later chapters: data parallelism replicates model state; model parallelism and ZeRO change what is replicated.

## Required Figures

| Figure ID                     | Purpose                                                                          | Form                     | Source                                                             |
| ----------------------------- | -------------------------------------------------------------------------------- | ------------------------ | ------------------------------------------------------------------ |
| `fig-08-data-parallel-step`   | Show replicas, data shards, local gradients, all-reduce, local optimizer update. | Step diagram             | `llmsys-14-data-parallel-allreduce`                                |
| `fig-08-collective-semantics` | Compare broadcast, reduce, all-reduce, reduce-scatter, all-gather.               | Collective table/diagram | `llmsys-14-collective-primitives`, `llmsys-14-allreduce-semantics` |
| `fig-08-ring-allreduce`       | Show scatter-reduce and all-gather phases over chunks.                           | Ring diagram             | `llmsys-14-ring-allreduce-phases`                                  |
| `fig-08-naive-vs-overlap-ddp` | Compare post-backward all-reduce with bucketed overlap.                          | Timeline                 | `llmsys-15-ddp-naive-allreduce`, `llmsys-15-ddp-overlap`           |
| `fig-08-ddp-reducer-hooks`    | Show autograd hooks, bucket pending counts, and async all-reduce trigger.        | Control-flow diagram     | `llmsys-15-ddp-autograd-hooks`                                     |

## Main Sections

### Replicating the Model Is the Easy Part

Introduce data parallelism as model replicas plus data shards. Make the systems bottleneck explicit: every worker must synchronize gradients before applying an equivalent update.

### All-Reduce Is the Central Collective

Define broadcast, reduce, reduce-scatter, all-gather, and all-reduce. Keep the notation simple: each rank starts with a gradient vector; after all-reduce, each rank has the averaged result.

### Ring All-Reduce Is a Schedule

Explain chunking, scatter-reduce, and all-gather. Avoid speedup or bandwidth claims unless official/NCCL sources and hardware conditions are added.

### Parameter Server Versus All-Reduce

Use as a brief contrast: central update/distribution versus decentralized collective gradient synchronization. Do not turn this into a survey.

### The Naive DDP Critical Path

Show why waiting until all backward computation completes before communication creates a large communication tail.

### Gradient Buckets

Explain why per-parameter synchronization is too fine-grained, while one giant all-reduce is too late. Buckets are a scheduling compromise.

### Autograd Hooks and Overlap

Show how framework internals observe gradient readiness and trigger communication. This connects Chapter 7 framework internals to distributed systems.

### What Controls Scaling

Discuss model size, gradient size, bucket order, bucket size, network bandwidth/latency, batch size, and stragglers. No universal scaling number.

### Boundary to Model Parallelism

Close by saying data parallelism replicates the model; when the model, optimizer state, or activations no longer fit or communication dominates, later chapters partition model and state.

## Technical Checks

- Formula correctness: Define average gradient carefully: `g = (1/W) * Σ_i g_i` for `W` workers, and distinguish sum from average depending on framework scaling.
- Complexity / memory accounting: If discussing communication volume, include message size, number of workers, topology, and whether counting per-rank or aggregate bytes.
- Hardware assumptions: NCCL/ring behavior depends on interconnect topology and implementation.
- Benchmark conditions: Do not include speedup/scaling efficiency without model, batch size, GPU count, interconnect, precision, software version, and baseline.
- Terminology consistency: Use `rank`, `worker`, `process`, `replica`, `bucket`, `all-reduce`, `reduce-scatter`, `all-gather`, `gradient averaging`, and `overlap` consistently.

## Sidebar Decisions

- PyTorch DDP API snippet: optional; keep illustrative, not API documentation.
- NCCL official docs: add before draft if exact API semantics matter.
- Li et al. VLDB 2020 paper: add before draft if DDP design goals or implementation history become central.
- Stragglers/fault tolerance: mention only briefly; reliability belongs later unless the spine changes.

## Open Questions

- Should Chapter 8 add official NCCL and PyTorch DDP documentation cards before draft?
- Should the Li et al. VLDB 2020 DDP paper be extracted before draft?
- Should the chapter include communication-volume formulas for ring all-reduce, or keep ring all-reduce qualitative until technical review?
- How much overlap detail belongs here versus a later systems-performance appendix?

## Handoff

Owner: Book Architect  
Purpose: Chapter 8 brief from distributed-training and DDP source extraction  
Evidence grade: A for course lecture claims; official NCCL/PyTorch docs and Li et al. VLDB 2020 recommended before API-level or publication-level DDP claims  
Assumptions: Chapter 8 focuses on synchronous data parallelism and DDP, leaving model/state partitioning to Chapters 9–10  
Open questions: Add official NCCL/PyTorch/DDP paper cards before draft if exact API or implementation-history claims are used  
Handoff: Systems Explainer for Chapter 8 draft
