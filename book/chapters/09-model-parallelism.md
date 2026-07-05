---
status: ready
chapter: 9
slug: 09-model-parallelism
title: Model Parallelism
primary_sources:
  - llmsys-16-model-parallel-83b41612547620ee0e172caa1ee448ed.pdf
official_docs:
  - https://docs.pytorch.org/docs/2.12/distributed.pipelining.html
papers:
  - https://arxiv.org/abs/1811.06965
  - https://arxiv.org/abs/1806.03377
  - https://arxiv.org/abs/2104.04473
secondary_sources: []
reader_level: engineer or graduate student who has read Chapters 1-8
technical_depth: intermediate-to-advanced
---

# Model Parallelism

Chapter 8 stopped at a hard boundary:

```text
data parallelism splits data, not the model
```

That boundary is useful while every worker can hold a full model replica, its gradients, its activations, and whatever optimizer state the training loop needs. Once that assumption fails, adding more data-parallel workers does not solve the local memory problem. Every worker is still trying to host the whole model.

Model parallelism changes the object being split. Instead of copying the whole model onto every device, it partitions model computation across devices. The course lecture frames model-parallel training as partitioning the forward pass, backward pass, and update computation across multiple workers, motivated by models that no longer fit in one GPU's memory. [CITE: llmsys-16-model-parallel-motivation]

There are two main ideas in this chapter:

```text
pipeline parallelism: split layers or blocks across devices
tensor parallelism: split individual tensor operations across devices
```

Both ideas solve a memory or compute-exposure problem by creating a scheduling and communication problem. The design question is not just "where do the weights fit?" It is:

```text
which device owns which computation,
what tensors cross device boundaries,
and when can each device do useful work?
```

## Data Parallelism Replicates; Model Parallelism Partitions

In synchronous data parallelism, every worker stores the same parameters `θ`. Each worker sees a different local batch, computes gradients, participates in gradient synchronization, and applies an equivalent local update.

Model parallelism breaks that symmetry. Device 0 may own early layers, device 1 may own middle layers, and device 2 may own later layers. Or several devices may jointly compute one large matrix multiplication inside a Transformer block.

The difference is visible in what each device stores:

```text
data parallel:
  GPU 0: full model, batch shard 0
  GPU 1: full model, batch shard 1
  GPU 2: full model, batch shard 2

model parallel:
  GPU 0: part of model
  GPU 1: part of model
  GPU 2: part of model
```

This does not remove distributed training communication. It moves the communication to different edges. In data parallelism, the main synchronization object is usually gradients. In model parallelism, the system must also move activations, activation gradients, partial matrix results, or reduced tensor-parallel outputs at specific points in the forward and backward graphs.

## Pipeline Parallelism Splits the Layers

Pipeline parallelism partitions the model into stages. A stage is a contiguous or otherwise assigned subset of the model that runs on a device or device group. The lecture introduces pipeline parallelism through layer-wise partitioning: different layers execute on different GPUs. [CITE: llmsys-16-layerwise-pipeline]

A simple four-stage pipeline might look like this:

```text
input
  -> stage 0: embedding + early Transformer blocks
  -> stage 1: next Transformer blocks
  -> stage 2: next Transformer blocks
  -> stage 3: final blocks + output head
  -> loss
```

During the forward pass, stage 0 computes activations and sends boundary activations to stage 1. Stage 1 computes its part and sends activations onward. During the backward pass, gradients flow in the reverse direction. Each stage computes parameter gradients for the layers it owns.

The memory benefit is direct: no single stage needs to store every model parameter. But the cost is also direct: each stage depends on tensors produced by neighboring stages. Pipeline parallelism therefore introduces point-to-point activation and gradient communication at partition boundaries.

## The Naive Pipeline Wastes Devices

The first attempt at pipeline parallelism is often too literal:

```text
run the whole batch through stage 0
then stage 1
then stage 2
then stage 3
```

That schedule partitions the model but does not keep the devices busy. While stage 0 is running, stages 1-3 are idle. While stage 1 is running, stages 0, 2, and 3 are idle. The lecture calls out this problem: in a naive pipeline, all but one GPU can be idle at a given moment. [CITE: llmsys-16-naive-pipeline-idle]

The issue is not that layer partitioning is wrong. The issue is that a batch is too large a unit of scheduling. A pipeline needs several independent work items in flight so one stage can process item `k+1` while the next stage processes item `k`.

This is the same systems pattern that appeared in Chapter 8. DDP improves when communication can overlap with backward computation. Pipeline parallelism improves when stages can overlap different micro-batches.

## Micro-Batching Fills the Pipeline

GPipe-style pipeline parallelism divides a mini-batch into smaller micro-batches. [CITE: llmsys-16-gpipe-microbatching] The GPipe paper describes partitioning a network across accelerators and using batch-splitting pipeline parallelism to improve utilization. [CITE: huang-2018-gpipe]

The scheduling idea is easier to see as a timeline:

```text
time ->

stage 0:  mb0  mb1  mb2  mb3
stage 1:       mb0  mb1  mb2  mb3
stage 2:            mb0  mb1  mb2  mb3
stage 3:                 mb0  mb1  mb2  mb3
```

The beginning and end still contain bubbles. At the start, later stages wait for the first micro-batch to arrive. At the end, earlier stages run out of new micro-batches while later stages drain the pipeline. The lecture identifies bubble overhead as a core pipeline cost. [CITE: llmsys-16-pipeline-costs]

This chapter intentionally does not state a bubble formula. Such formulas depend on stage count, micro-batch count, schedule, whether forward and backward are both included, and how the timeline accounts for warmup and drain. The safe lesson is qualitative:

```text
more micro-batches can improve pipeline occupancy,
but they also affect activation memory, launch overhead, and scheduling complexity
```

Micro-batching is therefore not a free knob. It changes how much work is available to fill the pipeline and how many intermediate tensors may be in flight.

## Activation Memory Becomes a Schedule Problem

Backpropagation needs forward activations. In a pipeline, several micro-batches may be in flight before their backward passes run. That means stages may need to keep activations for multiple micro-batches.

The lecture lists activation memory as one of the costs of pipeline parallelism and introduces gradient checkpointing/rematerialization as a way to reduce activation storage. [CITE: llmsys-16-pipeline-costs] [CITE: llmsys-16-gradient-checkpointing]

Checkpointing trades memory for computation. Instead of storing every intermediate activation, the system stores selected checkpoints and recomputes omitted activations during backward.

```text
without checkpointing:
  store many forward activations
  backward can reuse them directly

with checkpointing:
  store fewer activations
  recompute missing activations during backward
```

The tradeoff is not abstract. It changes the pipeline schedule. Recompute work consumes GPU time that might otherwise run forward or backward computation for another micro-batch. The right choice depends on memory pressure, stage balance, micro-batch count, and compute headroom.

## 1F1B Starts Backward Earlier

One way to reduce in-flight activation pressure is to start backward as soon as useful backward work is available. The lecture describes 1F1B as a schedule that starts backward as soon as possible, reducing activation memory relative to schedules that run many forwards before backward. [CITE: llmsys-16-one-f-one-b]

The name means "one forward, one backward" in the steady state. A simplified view is:

```text
warm up pipeline with forward micro-batches
then alternate forward and backward work where dependencies allow
drain remaining backward work
```

PipeDream is a representative system in this design space. The PipeDream paper describes pipelining forward and backward passes across model partitions to keep devices productive. [CITE: harlap-2018-pipedream]

The important distinction is between partitioning and scheduling:

```text
partitioning decides which stage owns which layers
scheduling decides when each stage runs forward or backward for each micro-batch
```

1F1B changes the schedule. It does not eliminate boundary communication, and it does not remove the need to balance stage compute. A slow stage can still throttle the pipeline.

## Interleaving and Chunking Reduce Imbalance

A pipeline stage does not have to be one contiguous chunk assigned once to one device. The lecture discusses chunking or interleaving stages as a way to improve pipeline behavior. [CITE: llmsys-16-pipeline-chunking]

Megatron-LM is a useful source for this broader composition. Its paper discusses combining tensor, pipeline, and data parallelism and includes interleaved pipeline scheduling as part of its distributed training design. [CITE: narayanan-2021-megatron-lm]

Interleaving can help when a simple stage assignment leaves bubbles or imbalance, but it also makes the schedule harder to reason about. More chunks mean more dependencies, more sends and receives, and more opportunities for implementation details to matter.

The robust rule is:

```text
pipeline efficiency is controlled by stage balance, pipeline occupancy, communication, and schedule
```

Layer partitioning alone is only the placement decision.

## Pipeline APIs Expose the Runtime Contract

PyTorch's pipeline parallelism documentation describes a runtime built around pipeline stages and schedules. The docs note that `torch.distributed.pipelining` is alpha, and describe `PipelineStage` as managing buffers and communication operations for a stage. They also list schedules such as GPipe, 1F1B, interleaved 1F1B, and looped BFS. [CITE: official-pytorch-pipeline-parallelism]

The API details may change, but the runtime contract is stable enough to learn from:

```text
a stage needs static expectations about inputs and outputs
a schedule decides which micro-batch work runs next
the runtime manages sends, receives, and buffers at stage boundaries
```

This is why pipeline parallelism belongs in a systems book. It is not just a model refactor. It is an execution protocol.

## Tensor Parallelism Splits the Matrix

Pipeline parallelism splits the model by layers or blocks. Tensor parallelism splits computation inside an operation.

The lecture introduces tensor parallelism as splitting matrix computation across GPUs. [CITE: llmsys-16-tensor-parallel-matmul] For a linear layer:

```text
Y = X A
```

one can split the weight matrix by columns:

```text
A = [A0 A1]

Y0 = X A0
Y1 = X A1
Y  = [Y0 Y1]
```

Each device computes a slice of the output features. If the next operation can consume those slices independently, communication can be delayed. If the next operation needs the full `Y` on every device, an all-gather-like communication step is needed.

Alternatively, split by rows:

```text
A = [A0
     A1]

X = [X0 X1]

Y = X0 A0 + X1 A1
```

Now devices compute partial sums that must be reduced. The communication pattern depends on the partition. That is the central tensor-parallel lesson:

```text
the matrix split determines the collective
```

## Transformer FFN Tensor Parallelism

A Transformer feed-forward network has two linear projections with a nonlinearity between them. In simplified form:

```text
H = activation(X A)
Z = H B
```

where `A` expands the hidden dimension and `B` projects back down.

The lecture describes splitting the first projection over columns and the second projection over rows. [CITE: llmsys-16-tensor-parallel-ffn]

With two partitions:

```text
A = [A0 A1]

H0 = activation(X A0)
H1 = activation(X A1)
```

The nonlinearity is elementwise, so each partition can apply it locally to its slice. Then the second projection can be split so each partition produces a partial contribution:

```text
Z = H0 B0 + H1 B1
```

The final output requires combining those partial contributions. In practice, that combination is a reduction across the tensor-parallel group.

The mechanism matters because it avoids materializing the full expanded intermediate on one device. The expanded hidden dimension is distributed across devices while the elementwise activation remains local.

## Attention Head Parallelism

Self-attention has a natural partition boundary: heads. Multi-head attention computes several attention heads and then combines their outputs.

The lecture describes splitting attention weights over columns or heads and notes that the head-local computation does not require all-reduce. [CITE: llmsys-16-tensor-parallel-attention]

That statement needs a boundary. It is safe for the per-head attention work:

```text
head i:
  Q_i, K_i, V_i
  attention_i = softmax(Q_i K_i^T) V_i
```

Different heads can be computed on different devices because one head's score matrix does not depend on another head's score matrix.

It is not safe to claim that the entire attention block is communication-free. After head outputs are produced, the model usually combines them through an output projection. Depending on how that projection is partitioned and what the following layer expects, the system may need a reduction, all-gather, or another layout transition.

The useful rule is:

```text
head-local attention is parallel-friendly;
block-level tensor layout still has to match the next operation
```

## Embeddings Are a Special Case

Embedding layers can also be partitioned, but they are easy to explain imprecisely. The lecture distinguishes communication behavior for input and output embeddings, including cases that require all-reduce or all-gather and cases where output embedding work can be fused with cross-entropy to reduce communication. [CITE: llmsys-16-tensor-parallel-embeddings]

For this chapter, the main point is limited:

```text
not every tensor-parallel layer has the same communication pattern
```

Embedding tables, output logits, and loss computation interact with vocabulary partitioning. Those details are implementation-sensitive enough that they should be treated as an advanced design point, not a universal recipe.

## Combining Tensor, Pipeline, and Data Parallelism

Large training runs often combine several parallelism axes:

```text
tensor parallelism: split operations inside a layer
pipeline parallelism: split layers or blocks into stages
data parallelism: replicate those partitions over batch shards
```

The lecture presents tensor, pipeline, and data parallelism as composable, with tensor parallelism often used within a server and pipeline parallelism extending across servers, then data parallelism scaling to additional replicas. [CITE: llmsys-16-parallelism-composition]

That is guidance, not a law. The correct placement depends on hardware topology, interconnect bandwidth, model shape, batch size, sequence length, precision, and framework/runtime support.

A practical mental model is:

```text
tensor parallelism wants fast collectives
pipeline parallelism wants balanced stages and enough micro-batches
data parallelism wants enough local compute to amortize gradient synchronization
```

The axes interact. Increasing tensor-parallel degree can reduce per-device parameter memory but increase collective communication. Increasing pipeline stages can make a model fit but increase bubbles or boundary traffic. Increasing data-parallel replicas can improve throughput only if synchronization remains manageable.

## What to Remember

Model parallelism is not one technique. It is a family of partitioning and scheduling choices.

Pipeline parallelism asks:

```text
which layers belong to which stage,
how do micro-batches move through the stages,
and how much idle time remains?
```

Tensor parallelism asks:

```text
which tensor dimension is split,
which partial results are local,
and which collective reconstructs the needed value?
```

The common misconception is:

```text
model parallelism just makes big models fit
```

Fitting is only the first constraint. The system also has to keep devices busy, store or recompute activations, move tensors across partition boundaries, and compose with data parallelism. At LLM scale, the model is not merely a neural network. It is a distributed program whose partitioning decisions determine memory pressure, communication, and schedule efficiency.

Owner: Principal Author
Purpose: Chapter 9 ready draft after source extraction, brief, technical review, and red-team review
Evidence grade: A for course lecture claims, GPipe/PipeDream/Megatron papers, and PyTorch pipeline docs; no benchmark numbers used
Assumptions: Chapter 9 covers pipeline and tensor parallelism; ZeRO, optimizer-state partitioning, and MoE remain Chapter 10
Open questions: Add formulas for pipeline bubbles or activation memory only if a later revision carries explicit schedule assumptions
Handoff: Production can move to Chapter 10 source extraction
