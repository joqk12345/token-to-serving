---
status: brief
chapter: 9
slug: 09-model-parallelism
title: Model Parallelism
primary_sources:
  - llmsys-16-model-parallel-83b41612547620ee0e172caa1ee448ed.pdf
secondary_sources: []
official_docs:
  - https://docs.pytorch.org/docs/2.12/distributed.pipelining.html
papers:
  - https://arxiv.org/abs/1811.06965
  - https://arxiv.org/abs/1806.03377
  - https://arxiv.org/abs/2104.04473
reader_level: engineer or graduate student who has read Chapters 1-8
technical_depth: intermediate-to-advanced
---

# Model Parallelism

## Chapter Thesis

Data parallelism splits batches while replicating the model; model parallelism changes the model execution itself. Pipeline parallelism partitions layers across devices, while tensor parallelism partitions individual matrix operations. Both are scheduling and communication problems, not just memory-saving tricks.

## Reader Problem

The reader understands DDP: every worker has a full model replica and synchronizes gradients. The missing step is what to do when the model, activations, or per-step memory footprint no longer fit on one device, or when data parallelism alone does not expose enough useful compute. The reader needs to distinguish layer-wise partitioning from tensor-wise partitioning and understand the costs each introduces.

## System Bottleneck

Primary bottlenecks: device memory capacity, pipeline bubbles, activation storage, partition-boundary activation/gradient communication, tensor-parallel collectives, load balance, and schedule complexity.

Secondary bottlenecks: stage placement, micro-batch count, recomputation, pipeline flush behavior, tensor-parallel degree, server topology, and composition with data parallelism.

## Source Map

| Claim | Source card | Evidence grade | Notes |
|---|---|---|---|
| Model parallelism is motivated by models exceeding single-GPU memory. | `llmsys-16-model-parallel-motivation` | A | Avoid exact model-size examples unless separately sourced. |
| Pipeline parallelism partitions model layers across devices. | `llmsys-16-layerwise-pipeline` | A | Basic definition. |
| Naive pipeline schedules can leave most devices idle and fail to overlap communication/computation. | `llmsys-16-naive-pipeline-idle` | A | Use as opening bottleneck. |
| GPipe-style micro-batching fills the pipeline with smaller micro-batches. | `llmsys-16-gpipe-microbatching` | A | Add GPipe paper before detailed algorithm/performance claims. |
| GPipe partitions sequential layers across accelerators and uses batch-splitting pipeline parallelism. | `huang-2018-gpipe` | A | No benchmark numbers without setup. |
| Pipeline parallelism has bubble overhead, boundary communication, and activation-memory pressure. | `llmsys-16-pipeline-costs` | A | Formulas require review. |
| Gradient checkpointing trades recomputation for lower activation memory. | `llmsys-16-gradient-checkpointing` | A | Add Chen et al. before asymptotic claims. |
| 1F1B starts backward earlier to reduce in-flight activation storage. | `llmsys-16-one-f-one-b` | A | Add PipeDream paper before implementation detail. |
| PipeDream schedules forward/backward passes across partitioned layers to keep devices productive. | `harlap-2018-pipedream` | A | Variants need careful wording. |
| Chunking/interleaving pipeline stages can reduce pipeline inefficiency. | `llmsys-16-pipeline-chunking` | A | Megatron paper needed for named system detail. |
| Megatron-LM composes tensor, pipeline, and data parallelism and discusses interleaved pipeline scheduling. | `narayanan-2021-megatron-lm` | A | Avoid throughput claims without setup. |
| PyTorch pipeline APIs use pipeline stages and schedules. | `llmsys-16-pytorch-pipelining` | A | Official docs needed before API-level draft claims. |
| PyTorch pipeline docs describe `PipelineStage`, GPipe/1F1B schedules, and alpha-status caveat. | `official-pytorch-pipeline-parallelism` | A | Preserve API caveat. |
| Tensor parallelism splits matrix computation across devices. | `llmsys-16-tensor-parallel-matmul` | A | Main tensor-parallel definition. |
| FFN tensor parallelism can split first and second projections in complementary ways. | `llmsys-16-tensor-parallel-ffn` | A | Notation and collectives need review. |
| Self-attention tensor parallelism can split heads/columns for head-local computation. | `llmsys-16-tensor-parallel-attention` | A | Do not claim the entire attention block has no communication. |
| Embedding tensor parallelism has distinct input/output communication behavior. | `llmsys-16-tensor-parallel-embeddings` | A | High-risk; use cautiously or sidebar. |
| Tensor, pipeline, and data parallelism can be combined. | `llmsys-16-parallelism-composition` | A | Qualitative guidance only. |

## Explanation Arc

1. Start from Chapter 8's boundary: data parallelism replicates the model; model parallelism partitions it.
2. Define model parallelism as partitioning forward/backward/update computation.
3. Pipeline parallelism: split layers across devices.
4. Show naive pipeline schedule and bubble/idle problem.
5. Add micro-batching to fill the pipeline.
6. Add activation-memory pressure and recomputation/checkpointing tradeoff.
7. Add 1F1B and interleaving as schedule improvements.
8. Switch to tensor parallelism: split one large matrix operation across devices.
9. Use Transformer FFN and attention as concrete tensor-parallel cases.
10. Explain composition: tensor parallel within a server, pipeline across stages, data parallel across replicas.
11. Close with tradeoff: model parallelism solves fit/compute exposure at the cost of communication and scheduling complexity.

## Required Figures

| Figure ID | Purpose | Form | Source |
|---|---|---|---|
| `fig-09-data-vs-model-parallel` | Contrast data-parallel replication with model partitioning. | Side-by-side architecture diagram | `llmsys-16-model-parallel-motivation` |
| `fig-09-naive-pipeline-bubble` | Show idle devices in naive layer-wise pipeline. | Timeline | `llmsys-16-naive-pipeline-idle` |
| `fig-09-gpipe-microbatches` | Show micro-batches filling pipeline stages. | Timeline | `llmsys-16-gpipe-microbatching` |
| `fig-09-1f1b-vs-gpipe` | Compare all-forward-then-backward with 1F1B. | Timeline | `llmsys-16-one-f-one-b` |
| `fig-09-tensor-parallel-ffn` | Show column/row split of FFN projections. | Matrix partition diagram | `llmsys-16-tensor-parallel-ffn` |
| `fig-09-tensor-parallel-attention-heads` | Show attention head partitioning. | Block diagram | `llmsys-16-tensor-parallel-attention` |
| `fig-09-3d-parallelism` | Show tensor × pipeline × data parallel axes. | 3D grid or nested groups | `llmsys-16-parallelism-composition` |

## Main Sections

### Data Parallelism Replicates; Model Parallelism Partitions

Reopen from Chapter 8: DDP keeps a full replica on every worker. Model parallelism begins when full replication is not viable or not sufficient.

### Pipeline Parallelism: Split the Layers

Explain layer-wise partitioning, activation send/receive, backward gradient send/receive, and per-stage memory responsibilities.

### The Bubble Problem

Show naive pipeline idle time. Make the bottleneck concrete: multiple devices exist, but the schedule exposes only one stage at a time.

### Micro-Batching and GPipe

Explain how smaller micro-batches fill the pipeline. Avoid GPipe performance claims until GPipe paper is extracted.

### Activation Memory and Checkpointing

Explain why in-flight micro-batches require activation storage and why recomputation trades compute for memory.

### 1F1B and Interleaving

Explain backward starting earlier, fewer in-flight activations, and chunked/interleaved stages.

### Tensor Parallelism: Split the Matrix

Explain row/column partitioning at the matrix-multiply level. Use the communication consequence as the main idea: some partitions need all-gather, others need reduction.

### Transformer FFN Tensor Parallelism

Show why splitting the expansion projection and contraction projection in complementary ways avoids materializing the full intermediate on one GPU.

### Attention Head Parallelism

Explain head-wise partitioning carefully. Head-local attention does not mean the entire block is communication-free.

### Combining Parallelism Axes

Introduce tensor parallel groups, pipeline stages, and data-parallel replicas as composable but topology-sensitive axes.

## Technical Checks

- Formula correctness: If including bubble overhead or memory formulas, verify notation and assumptions from GPipe/PipeDream papers.
- Complexity / memory accounting: Distinguish parameter memory, activation memory, gradient memory, and optimizer state.
- Hardware assumptions: Tensor-parallel guidance depends on within-node bandwidth and topology.
- Benchmark conditions: Do not quote GPipe/PipeDream/Megatron throughput without hardware, model, micro-batches, partitions, precision, and baseline.
- Terminology consistency: Use `pipeline stage`, `micro-batch`, `bubble`, `1F1B`, `tensor parallelism`, `column parallel`, `row parallel`, `pipeline parallelism`, and `data parallelism` consistently.
- Chapter boundary: Do not move ZeRO, optimizer-state partitioning, or MoE into Chapter 9 except as a transition.

## Sidebar Decisions

- GPipe: main mechanism allowed; paper card needed before exact formula/performance.
- PipeDream/1F1B: schedule concept allowed; paper card needed for detailed variants.
- Megatron-LM: use only as later source if explaining interleaving or tensor-parallel Transformer implementation deeply.
- PyTorch pipeline API: keep as optional sidebar unless official docs are added.
- Embedding parallelism: optional advanced sidebar due to higher risk of implementation-specific details.

## Open Questions

- Should Chapter 9 include pipeline bubble formulas, or keep them qualitative until technical review?
- How much tensor-parallel FFN math should be included before becoming too implementation-specific?
- Should embedding parallelism be included in the main draft or left as a sidebar?

## Handoff

Owner: Book Architect  
Purpose: Chapter 9 brief from model-parallel source extraction  
Evidence grade: A for course lecture claims, GPipe/PipeDream/Megatron papers, and PyTorch pipeline docs  
Assumptions: Chapter 9 covers pipeline and tensor parallelism; ZeRO/MoE remain Chapter 10  
Open questions: Use named systems without benchmark numbers unless all experimental conditions are carried  
Handoff: Systems Explainer for Chapter 9 draft
