---
status: ready
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

Chapter 9 partitioned model computation. Pipeline parallelism split layers. Tensor parallelism split matrix operations. Those techniques answer one question:

```text
How can several devices jointly execute one model?
```

This chapter starts from a different question:

```text
What must each device keep in memory while training?
```

That question matters because distributed training does not only store weights. A worker may also hold gradients, optimizer state, activations, temporary communication buffers, and fragmented allocations. At LLM scale, the memory ledger is as important as the compute graph.

ZeRO and MoE attack different entries in that ledger.

```text
ZeRO: reduce redundant training state across data-parallel workers
MoE: increase parameter capacity while activating only selected experts per token
```

Both are systems techniques. Neither makes memory pressure disappear. ZeRO trades resident memory for partitioning and communication. MoE trades dense activation for routing, all-to-all communication, and load balancing.

## The Memory Ledger After Model Parallelism

Data parallelism is attractive because every worker runs a normal model program on a local batch shard. The price is replication. Chapter 8 focused on gradient synchronization. Now look at what each worker stores.

The ZeRO lecture describes mixed-precision DDP memory as including FP16 parameters, FP16 gradients, FP32 optimizer-related states, activations, temporary buffers, and fragmentation. [CITE: llmsys-18-ddp-memory-accounting]

For Adam-style training under the lecture's mixed-precision framing, the important categories are:

```text
parameters
gradients
optimizer states
activations
temporary buffers
fragmentation / allocator overhead
```

Parameters are only the first line item. Adam keeps state such as momentum and variance. Mixed-precision training may also maintain higher-precision copies of values used by the optimizer. Activations are needed for backward. Communication libraries and framework reducers may flatten or bucket tensors into temporary buffers. Allocators may leave memory unusable because of fragmentation.

This is why "the model has N parameters" is not enough to decide whether training fits. The fit question is:

```text
resident memory per worker =
  parameters
  + gradients
  + optimizer state
  + activations
  + temporary buffers
  + fragmentation
  + implementation-specific overhead
```

Model parallelism can reduce how much of the model computation sits on one device. It does not automatically remove every replicated data-parallel state. ZeRO targets that redundancy directly.

## ZeRO: Remove Redundant State

ZeRO stands for Zero Redundancy Optimizer. The lecture states the key idea as eliminating data redundancy in DDP by partitioning optimizer states, gradients, and parameters across stages: ZeRO-1, ZeRO-2, and ZeRO-3. [CITE: llmsys-18-zero-key-idea]

The original ZeRO paper frames the same goal as eliminating memory redundancies while retaining useful computational granularity and communication properties. [CITE: rajbhandari-2019-zero]

The core idea is simple:

```text
DDP: every worker stores the same training state
ZeRO: workers share responsibility for pieces of that state
```

ZeRO does not mean the model no longer trains data-parallel. Workers still process different data shards. The difference is that not every worker must permanently store every optimizer state, gradient, and parameter tensor.

The stages are easiest to understand as progressively removing replicated state:

```text
ZeRO-1: partition optimizer states
ZeRO-2: partition optimizer states + gradients
ZeRO-3: partition optimizer states + gradients + parameters
```

Each stage reduces a different memory category. Later stages save more resident memory but require more careful communication and scheduling.

## ZeRO-1 Partitions Optimizer States

Optimizer state can dominate training memory. For Adam-style optimizers, each parameter can have associated state such as momentum and variance. In mixed-precision setups, optimizer-side state may be stored at higher precision than the model copy used for forward and backward.

ZeRO-1 partitions optimizer states across `K` workers. The lecture describes stage 1 as partitioning optimizer states into `K` parts, with each GPU processing one partition, while FP16 parameters are still used for forward and backward. [CITE: llmsys-18-zero-stage1-optimizer]

Conceptually:

```text
DDP:
  worker 0: all optimizer state
  worker 1: all optimizer state
  worker 2: all optimizer state
  worker 3: all optimizer state

ZeRO-1:
  worker 0: optimizer state shard 0
  worker 1: optimizer state shard 1
  worker 2: optimizer state shard 2
  worker 3: optimizer state shard 3
```

Forward and backward still need model parameters. Gradients still have to be computed. But the optimizer update responsibility is partitioned: each worker owns the optimizer state for a shard of the parameters.

The design point is that optimizer state is large, persistent, and redundant under ordinary DDP. It is a natural first target.

## ZeRO-2 Partitions Gradients

ZeRO-2 extends the idea to gradients. The lecture states that each GPU computes all parameter gradients for its data partition, but stores only one partition of gradients instead of all gradients. Gradients outside a worker's responsibility are passed to the responsible GPU. [CITE: llmsys-18-zero-stage2-gradients]

This distinction matters:

```text
computed temporarily does not mean retained persistently
```

During backward, a worker may produce gradient values for many parameters. But after the relevant reduction/transfer, it only needs to keep the shard it owns.

In a four-worker example:

```text
gradient shard 0 -> worker 0 owns final shard
gradient shard 1 -> worker 1 owns final shard
gradient shard 2 -> worker 2 owns final shard
gradient shard 3 -> worker 3 owns final shard
```

Other workers may hold temporary buffers while computing or reducing those gradients, but they do not keep the full gradient set after ownership is resolved.

ZeRO-2 is therefore not just "smaller gradients." It is a change in lifetime and ownership:

```text
temporary local gradient computation
  -> reduce or send to owner
  -> release unowned gradient storage
```

That lifetime management is the systems mechanism.

## ZeRO-3 Partitions Parameters

ZeRO-3 partitions parameters themselves. The lecture describes partitioning parameters into `K` parts and communicating parameter partitions during forward and backward. [CITE: llmsys-18-zero-stage3-parameters]

This is a stronger memory reduction because parameters are no longer fully resident on every worker. But the model computation still needs parameter values at the moment a layer runs. That means the runtime must make parameter shards available when needed and release unowned shards when they are no longer needed.

A simplified view:

```text
before computing a layer group:
  gather or broadcast needed parameter shard

compute forward or backward for that layer group

after the computation:
  keep owned shard
  release temporary unowned shards when safe
```

The important tradeoff is explicit in the lecture: ZeRO-3 reduces memory at the cost of additional parameter transfer. [CITE: llmsys-18-zero-communication-cost]

That tradeoff is not a footnote. It is the whole design problem. ZeRO-3 turns parameter residency into a schedule:

```text
which parameter shard is needed next?
which worker owns it?
when should it be communicated?
when can temporary copies be freed?
```

The memory saving is only useful if the communication can be scheduled without dominating the step.

## A Conditional Memory Formula

The lecture gives a compact symbolic memory accounting for ZeRO stages. This box is useful, but only under its assumptions. [CITE: llmsys-18-zero-memory-formulas]

Assume:

```text
N = number of model parameters
M = optimizer-state bytes per parameter
K = number of data-parallel workers
```

Use the lecture's simplified mixed-precision-style state accounting, where the formula tracks model-state categories and excludes activations, temporary buffers, fragmentation, and implementation-specific overhead.

Under those assumptions:

```text
Original DDP: 4N + M*N
ZeRO-1:       4N + M*N/K
ZeRO-2:       2N + (2+M)*N/K
ZeRO-3:       (4+M)*N/K
```

The formula should not be read as total training memory. It does not include activation memory, sequence-length effects, micro-batch size, framework buffers, allocator behavior, or communication staging.

What the formula does show is the direction of the stages:

```text
ZeRO-1 divides optimizer state
ZeRO-2 also divides gradient state
ZeRO-3 also divides parameter state
```

The reliable takeaway is not a universal byte count. It is the accounting habit: identify each resident state category, then ask whether it is replicated, partitioned, temporary, or recomputed.

## Beyond the Three Stages

The lecture also lists other memory optimizations: partitioned activation checkpointing, constant-size buffers, memory defragmentation, memory reuse, and communication reduction techniques. [CITE: llmsys-18-zero-other-memory-optimizations]

This matters because real training memory is not just long-lived tensors.

Activations can dominate for long sequences or large micro-batches. Checkpointing trades recomputation for lower activation storage, as Chapter 9 introduced. Communication buffers can spike memory during reductions or parameter transfers. Fragmentation can make nominally free memory unusable for a large allocation.

So ZeRO should be understood as part of a runtime memory manager:

```text
partition long-lived state
control temporary buffers
reduce activation residency
avoid fragmentation
schedule communication around computation
```

That is why ZeRO belongs after DDP and model parallelism. It is not another way to split layers. It is a way to split the training state that DDP would otherwise replicate.

## MoE: Store Capacity, Activate Sparsely

ZeRO reduces memory by partitioning training state. Mixture-of-Experts changes a different axis: which parameters are activated for each token.

The MoE lecture defines Transformer MoE as replacing a single dense FFN with multiple expert FFNs and a router or gating network that selects one or more experts. [CITE: llmsys-17-moe-ffn-router]

In a dense Transformer block, every token passes through the same FFN parameters:

```text
token hidden state -> FFN -> output
```

In an MoE Transformer block:

```text
token hidden state -> router
                   -> selected expert FFN(s)
                   -> combined output
```

The system may store many expert FFNs, but each token activates only a subset. Switch-style MoE is a simple version: the lecture describes one token being passed through one selected FFN. [CITE: llmsys-17-switch-top1-routing] The Switch Transformer paper frames sparse expert selection as choosing different parameters for incoming examples, while also emphasizing complexity, communication cost, and training instability as practical barriers. [CITE: fedus-2021-switch-transformer]

The key distinction is:

```text
total parameters: all experts plus shared model components
activated parameters per token: selected expert path plus shared components
```

These are not the same number. That is why MoE can increase model capacity without making every token pay for every expert. It is also why parameter count alone is misleading for MoE cost.

## Routing Is a Systems Problem

The router is part of the model, but it also behaves like a scheduler. It decides where tokens go.

For each token, the router scores experts and chooses one or more expert paths. If many tokens choose the same expert, that expert's device becomes overloaded while others may sit underused. If the router spreads tokens more evenly, the system can use expert devices more effectively.

This is not only a throughput issue. The lecture notes that MoE training uses load-balancing losses to avoid routing collapse to experts and to balance computation across experts or devices. [CITE: llmsys-17-moe-load-balancing]

A useful way to think about the router:

```text
model role:
  choose expert computation for token representation

systems role:
  assign token work to expert devices
```

The same routing decision influences gradient flow, expert specialization, device utilization, communication volume, and queueing at expert owners.

This is why MoE is not just "a sparse FFN." It is conditional computation with a runtime placement problem.

## Expert Parallelism and All-to-All

MoE becomes a distributed-systems problem when experts are placed on different devices. The lecture describes expert parallelism as splitting experts across devices while replicating non-expert components, and it states that expert parallelism requires all-to-all communication. [CITE: llmsys-17-expert-parallelism]

The flow is:

```text
1. each device starts with a batch of token hidden states
2. router chooses expert assignments
3. tokens are dispatched to devices that own selected experts
4. expert FFNs run
5. expert outputs return to the devices/layers that need them
```

The dispatch and return are the communication bottleneck. Unlike DDP all-reduce, where every worker combines corresponding gradient tensors, MoE expert parallelism moves token representations according to routing decisions. The communication pattern depends on the batch's token-to-expert assignment.

GShard is one important source for this style of system. The GShard paper combines conditional computation and automatic sharding for large sparse MoE Transformers. [CITE: lepikhin-2020-gshard]

The main lesson is:

```text
expert parallelism converts sparse activation into token exchange
```

Sparse compute helps only if token exchange, expert load, and kernel execution are scheduled efficiently.

## All-to-All Is Not a Detail

The MoE lecture states that expert parallelism creates all-to-all communication and discusses optimized all-to-all patterns such as hierarchical or parallelism-coordinated communication schedules. [CITE: llmsys-17-moe-alltoall-optimization]

That claim needs conditions. All-to-all cost depends on:

- number of expert-parallel devices;
- token count per batch or micro-batch;
- hidden dimension;
- routing fanout;
- token skew across experts;
- interconnect topology;
- implementation of the collective or point-to-point exchange;
- overlap with computation.

This chapter therefore avoids topology-free latency claims. The safe statement is:

```text
MoE communication is data-dependent:
the router determines which token representations move where
```

This makes MoE sensitive to batch shape and routing skew in a way that dense FFNs are not.

## Shared and Fine-Grained Experts

Modern MoE designs do not have to use only one set of routed experts. The lecture describes shared-routed expert designs, where an always-used shared expert captures common computation and routed experts handle token-specific computation. [CITE: llmsys-17-shared-routed-experts]

DeepSeekMoE is an example of this design direction. The paper proposes fine-grained expert segmentation and shared experts to improve expert specialization. [CITE: dai-2024-deepseekmoe]

At this chapter's level, the mechanism is:

```text
shared expert:
  always active path for common capacity

routed experts:
  selected paths for token-specific capacity

fine-grained experts:
  smaller expert units allow more flexible combinations
```

This is not a license to quote model-quality or compute-reduction numbers without experiment conditions. The useful systems point is that MoE architecture choices change routing granularity, expert load, and communication patterns.

## MoE Inference Preview

The lecture notes that MoE inference performance depends on overall model size, number of activated experts, memory bandwidth, token grouping, communication scheduling, and MoE kernels. [CITE: llmsys-17-moe-inference-bottlenecks]

DeepSpeed-MoE frames MoE as an end-to-end training and inference systems problem, including the difficulty of serving sparse expert models efficiently. [CITE: rajbhandari-2022-deepspeed-moe]

This chapter will not turn into a serving chapter. Part V covers serving, scheduling, KV cache, and inference memory in detail. The reason to mention inference here is to prevent a misconception:

```text
sparse activation reduces some compute,
but MoE still has memory bandwidth, routing, communication, and kernel costs
```

Training and inference both have to move token representations through expert paths. The exact bottleneck changes with workload and hardware, but the systems shape remains.

## ZeRO and MoE Solve Different Problems

ZeRO and MoE are often discussed together because both are used in large-model training stacks. But they solve different memory problems.

ZeRO asks:

```text
Which training state is redundantly stored on every data-parallel worker?
Can optimizer states, gradients, or parameters be sharded instead?
```

MoE asks:

```text
Does every token need to use the same dense FFN parameters?
Can the model store more expert capacity while activating only selected experts?
```

The bottlenecks differ:

```text
ZeRO bottleneck:
  parameter/gradient/optimizer-state residency
  plus communication to make shards available

MoE bottleneck:
  router decisions, expert load, token exchange,
  all-to-all communication, and memory bandwidth
```

They can also compose with the techniques from earlier chapters. A system may use tensor parallelism inside layers, pipeline parallelism across blocks, data parallelism across replicas, ZeRO to shard training state, and expert parallelism for MoE layers. That combination is powerful, but it is no longer "just training a Transformer." It is a distributed runtime with several interacting schedules.

## What to Remember

The central object in this chapter is not a layer or a kernel. It is a memory ledger.

For ZeRO:

```text
identify replicated state
partition ownership
communicate shards when needed
release temporary copies when safe
```

For MoE:

```text
store many experts
route each token to selected experts
move token representations to expert owners
balance load so sparse compute remains useful
```

The misconception to discard is:

```text
larger models are only a compute-scaling problem
```

At LLM scale, training is also a residency problem. The system must decide what lives on each device, what is temporary, what is partitioned, what is recomputed, and what must cross the network. ZeRO and MoE are two different answers to that residency problem: one partitions training state; the other sparsifies activated model capacity.

Owner: Principal Author
Purpose: Chapter 10 ready draft after source extraction, brief, technical review, and red-team review
Evidence grade: A for course lecture claims and primary papers; no benchmark numbers used
Assumptions: ZeRO formulas are presented only under the lecture's simplified `N`, `M`, `K` model-state accounting and exclude activations, buffers, fragmentation, and implementation overhead
Open questions: Whether to add PyTorch FSDP or ZeRO++ source cards in a later revision
Handoff: Production can move to Chapter 11 source extraction
