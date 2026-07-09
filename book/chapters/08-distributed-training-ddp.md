---
status: ready
chapter: 8
slug: 08-distributed-training-ddp
title: Distributed Training and Data Parallelism
primary_sources:
  - llmsys-14-distributed-training-b27c3d1dc185e680c6f5cc924e9ec9d7.pdf
  - llmsys-15-ddp-165dbe3873fac21eb8b339e64bcfee28.pdf
official_docs:
  - https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
  - https://docs.pytorch.org/docs/2.12/generated/torch.nn.parallel.DistributedDataParallel.html
papers:
  - https://arxiv.org/abs/2006.15704
secondary_sources: []
reader_level: engineer or graduate student who has read Chapters 1-7
technical_depth: intermediate
---

# Distributed Training and Data Parallelism

Chapter 7 ended with a single training step becoming an executable program. That is still a one-replica view of training. Large models and large datasets quickly force a wider question:

```text
How do several devices cooperate on one training step?
```

The simplest answer is data parallelism. Copy the model to several workers. Split the batch. Let every worker run forward and backward on its shard. Then combine the gradients so every replica applies the same update.

That description is correct, but it hides the systems problem.

The expensive part is not making copies of the model. The expensive part is synchronizing gradients at the end of every step without turning the network into the critical path. Data parallelism scales only when communication is scheduled as carefully as computation.

This chapter is about that schedule.

## Replicating the Model Is the Easy Part

Suppose there are `W` workers. Each worker has a replica of the same model parameters `θ`. A global batch is split into `W` local batches:

![Data parallel training runs model replicas on different data shards, computes local gradients, aggregates them with all-reduce, and applies the synchronized update on each worker.](../figures/artwork/ch08/fig-08-data-parallel-step.svg)

```text
B = B_0 ∪ B_1 ∪ ... ∪ B_{W-1}
```

Worker `i` computes a local loss and local gradient:

```text
g_i = ∇θ L(θ; B_i)
```

For synchronous data parallel training, the workers need a shared gradient before the optimizer step. The usual averaged gradient is:

```text
g = (1/W) * Σ_i g_i
```

Then every worker applies the same optimizer update locally:

```text
θ ← optimizer_update(θ, g)
```

The course lecture on distributed training presents this pattern as data partitioning, local gradient computation, all-reduce to compute an average gradient, and local parameter updates. [CITE: llmsys-14-data-parallel-allreduce]

This is why data parallelism is attractive. Each worker performs the same model computation as single-device training, just on different data. The model code does not need to be partitioned across layers or tensors.

But the gradient vector is large. For dense training, each worker must communicate information proportional to the parameter gradients it owns. If communication happens after all backward computation finishes, step time becomes:

```text
step time ≈ forward + backward + gradient synchronization + optimizer
```

The synchronization term is now on the critical path.

## All-Reduce Is the Central Collective

Distributed training relies on collective communication. NCCL is a common GPU communication library; the lecture introduces NCCL as providing inter-GPU communication APIs, including collective and point-to-point primitives. [CITE: llmsys-14-nccl-communication] NVIDIA's NCCL documentation lists collective operations including all-reduce, broadcast, reduce, all-gather, and reduce-scatter. [CITE: official-nvidia-nccl-collectives]

![Broadcast, reduce, all-reduce, reduce-scatter, and all-gather differ in which workers own full tensors or shards before and after communication.](../figures/artwork/ch08/fig-08-collective-semantics.svg)

The key collective for synchronous data parallelism is all-reduce.

In a reduce operation, several ranks contribute values and one rank receives the reduced result. In a broadcast, one rank sends a value to all ranks. All-reduce combines those ideas: every rank contributes, the values are reduced, and every rank receives the result. The lecture defines all-reduce this way and also presents all-reduce as reduce-scatter followed by all-gather. [CITE: llmsys-14-allreduce-semantics]

For gradients, each worker begins with:

```text
rank 0: g_0
rank 1: g_1
rank 2: g_2
rank 3: g_3
```

After all-reduce with summation, every worker has:

```text
g_sum = g_0 + g_1 + g_2 + g_3
```

If the framework averages gradients, it divides by `W`:

```text
g = g_sum / W
```

That average matters. PyTorch's DDP documentation warns that gradient magnitude depends on whether loss is summed or averaged locally. [CITE: official-pytorch-ddp-docs] The chapter should therefore avoid saying "sum" and "average" interchangeably. The communication primitive may sum; the training algorithm often wants an average or an equivalent scaling.

## Ring All-Reduce Is a Schedule

An all-reduce call looks like one operation at the API level. Underneath, it is a communication schedule.

![Ring all-reduce circulates tensor chunks through scatter-reduce and all-gather phases so each worker receives the reduced result.](../figures/artwork/ch08/fig-08-ring-allreduce.svg)

The distributed training lecture explains ring all-reduce through two phases: scatter-reduce and all-gather. [CITE: llmsys-14-ring-allreduce-phases]

Imagine each gradient vector is split into chunks:

```text
g_i = [g_i0, g_i1, g_i2, g_i3]
```

In the scatter-reduce phase, workers send chunks around a ring. Each chunk is reduced as it moves, so after enough steps each worker owns one fully reduced chunk. In the all-gather phase, those reduced chunks move around the ring so every worker obtains the full reduced gradient.

The mechanism matters more than the exact diagram:

```text
all-reduce is not magic;
it is chunk movement plus reduction plus redistribution
```

This chapter deliberately avoids ring all-reduce bandwidth formulas. They are useful, but easy to misuse without stating message size, number of ranks, topology, algorithm variant, per-rank versus aggregate accounting, and whether links are full-duplex. The reliable claim here is qualitative: all-reduce performance is controlled by communication volume, latency, topology, and implementation.

## Parameter Server Versus All-Reduce

A parameter-server design centralizes part of the update. Workers send gradients to a server, the server updates parameters, and workers receive new parameters. The lecture contrasts that with all-reduce data parallelism, where workers synchronize gradients and then update locally. [CITE: llmsys-14-parameter-server-vs-allreduce]

This is not a universal judgment that all-reduce is always better. Parameter-server systems, sharded optimizers, and hybrid designs can be reasonable under different constraints.

The distinction for this chapter is simpler:

```text
parameter server: central update/distribution path
all-reduce data parallelism: collective gradient synchronization, local update
```

PyTorch DDP follows the all-reduce data-parallel pattern.

## The Naive DDP Critical Path

PyTorch `DistributedDataParallel` implements module-level data parallelism with gradient synchronization across model replicas. The official docs define it as data parallelism based on `torch.distributed` and note that the user is responsible for splitting inputs across participating GPUs. [CITE: official-pytorch-ddp-docs]

![Naive DDP waits until backward computation finishes before reducing gradients, while bucketed DDP can start asynchronous all-reduce as buckets become ready.](../figures/artwork/ch08/fig-08-naive-vs-overlap-ddp.svg)

The DDP lecture describes the standard structure: replicas run forward and backward independently, gradients are averaged across nodes, and optimizers run locally with identical updates. [CITE: llmsys-15-ddp-replica-gradient-average]

A naive implementation waits until the entire backward pass completes, then all-reduces all gradients. [CITE: llmsys-15-ddp-naive-allreduce]

The timeline looks like:

```text
forward
backward layer N
backward layer N-1
...
backward layer 1
all-reduce all gradients
optimizer step
```

That is correct, but it creates a communication tail. The network sits mostly idle during backward, then the GPU waits for communication before the optimizer can run.

The key DDP idea is to start communication before backward is over.

## Gradient Buckets

DDP cannot all-reduce every parameter as soon as its gradient appears. That would create many tiny communication operations, and latency would dominate.

It also should not wait for one giant gradient buffer at the end if it can avoid it.

The compromise is bucketing. DDP groups parameter gradients into buckets. The lecture notes that bucket size can be configured with `bucket_cap_mb`, and that bucket assignment is determined at construction time based on bucket size and parameter sizes. [CITE: llmsys-15-ddp-bucketing] The official PyTorch DDP signature also exposes `bucket_cap_mb`. [CITE: official-pytorch-ddp-docs]

A bucket becomes ready when all gradients assigned to it are ready. Then DDP can launch all-reduce for that bucket while backward continues computing other gradients.

This changes the timeline:

```text
backward late layers
bucket A ready -> async all-reduce A
backward middle layers while A communicates
bucket B ready -> async all-reduce B
backward early layers while A/B communicate
finish remaining communication
optimizer step
```

The communication has not disappeared. It has been moved under computation where possible.

## Autograd Hooks Make Communication Timely

Chapter 7 described frameworks as systems that can intercept computation. DDP is a concrete example.

![PyTorch DDP uses autograd hooks and reducer buckets to trigger asynchronous all-reduce when a bucket's gradients are ready.](../figures/artwork/ch08/fig-08-ddp-reducer-hooks.svg)

The DDP lecture shows reducer pseudocode built around autograd hooks. When a parameter's gradient has accumulated, an `autograd_hook` marks the variable ready. Buckets track pending gradients. When a bucket's pending count reaches zero, DDP marks the bucket ready and launches communication for that bucket. [CITE: llmsys-15-ddp-autograd-hooks]

In simplified form:

```text
on gradient ready(parameter):
  bucket = bucket_for(parameter)
  bucket.pending -= 1
  if bucket.pending == 0:
    all_reduce(bucket.gradients)
```

This is the "interceptive" side of DDP's design. The lecture cites DDP design goals as non-intrusive for user training scripts and interceptive enough for the implementation to trigger internal algorithms promptly. [CITE: llmsys-15-ddp-design-goals]

The Li et al. paper frames the same systems difficulty: data parallelism is conceptually straightforward, but dependencies between computation and communication make efficient training non-trivial. It identifies bucketing gradients and overlapping computation with communication as DDP acceleration techniques. [CITE: li-2020-pytorch-ddp]

## Overlap Is Conditional

It is tempting to summarize DDP as "communication is hidden by backward." That is too strong.

Overlap depends on several conditions:

- the order in which gradients become ready;
- how parameters are assigned to buckets;
- bucket size;
- gradient tensor sizes;
- network bandwidth and latency;
- GPU compute time per layer;
- whether communication contends with computation for resources;
- whether all workers reach bucket readiness at similar times.

The lecture states that DDP overlaps backward computation with all-reduce when buckets are ready. [CITE: llmsys-15-ddp-overlap] That is the safe claim. It can reduce exposed communication time when communication for earlier buckets runs concurrently with remaining backward computation. It does not guarantee perfect hiding.

A useful mental model is:

```text
exposed communication = communication not covered by useful computation
```

DDP bucketing tries to reduce exposed communication. The last bucket, stragglers, small models, small batches, or slow interconnects can still leave communication on the critical path.

## What Controls Scaling

Data parallelism increases the amount of compute available per step. It also increases synchronization work.

Scaling depends on at least:

- model parameter size, which drives gradient communication volume;
- local batch size, which affects compute per worker;
- number of workers;
- interconnect topology and bandwidth;
- collective implementation;
- precision and gradient dtype;
- bucket size and bucket readiness order;
- optimizer semantics and gradient scaling;
- stragglers.

This chapter does not quote a universal scaling number. The Li et al. paper reports evaluation results under specific PyTorch and hardware settings, but those numbers should be used only with their experimental conditions. [CITE: li-2020-pytorch-ddp]

The robust lesson is:

```text
data parallelism is a race between extra local compute and extra synchronization
```

When local computation is large relative to communication, scaling can be favorable. When communication dominates, adding workers can reduce hardware efficiency or even increase step time.

## Boundary to Model and State Parallelism

Data parallelism replicates model parameters on every worker. It also usually replicates optimizer state unless another technique is used. That is acceptable while the model and optimizer state fit per device.

When the model does not fit, Chapter 9 will partition the model itself. When optimizer state and gradients dominate memory, Chapter 10 will introduce ZeRO-style partitioning. Those techniques change what is replicated.

So Chapter 8's boundary is:

```text
data parallelism splits data, not the model
```

That boundary is why DDP is often the first distributed training idea to learn. It preserves the single-model program and adds a communication schedule around gradients. But it is not the whole distributed-training stack.

## What to Remember

The core data-parallel step is simple:

```text
replicate model
split batch
compute local gradients
all-reduce gradients
apply equivalent local updates
```

The systems work is in the middle:

```text
when do gradients become ready?
how are they bucketed?
which collective synchronizes them?
how much communication is exposed on the critical path?
```

DDP is important because it connects framework internals to communication scheduling. Autograd hooks tell the reducer when gradients are ready. Buckets make communication coarse enough to be efficient but early enough to overlap. All-reduce makes every replica agree on the update.

The misconception to discard is:

```text
distributed data parallelism is just running the same script on more GPUs
```

Running the script is the easy part. Synchronizing the gradients without wasting the step is the system.

Owner: Principal Author  
Purpose: Chapter 8 ready draft after source extraction, brief, draft, technical review, and red-team review  
Evidence grade: A for course lecture claims, NVIDIA NCCL docs, PyTorch DDP docs, and Li et al. DDP paper; no benchmark numbers used  
Assumptions: Chapter 8 focuses on synchronous data parallelism and DDP, leaving model/state partitioning to Chapters 9–10  
Open questions: Add ring all-reduce byte-count formulas only if a later revision can carry topology/message-size/full-duplex/per-rank assumptions  
Handoff: Production can move to Chapter 9 source extraction
