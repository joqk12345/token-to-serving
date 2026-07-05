# Source Note: llmsys-14 Distributed Training

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-14-distributed-training-b27c3d1dc185e680c6f5cc924e9ec9d7.pdf`

## Scope

This source introduces distributed training with emphasis on multi-GPU communication, NCCL collectives, data-parallel training through gradient all-reduce, ring all-reduce, and the contrast between parameter-server and all-reduce data parallelism.

## Key Claims

- NCCL provides inter-GPU communication APIs, including collective operations and point-to-point send/receive primitives, across interconnects such as PCIe, NVLink, InfiniBand, and IP sockets.
- Collective primitives include broadcast, reduce, reduce-scatter, all-gather, and all-reduce.
- All-reduce can be understood as a reduce operation followed by broadcast; the lecture also presents all-reduce as reduce-scatter followed by all-gather.
- Data-parallel training partitions the data across workers, computes local gradients independently, uses all-reduce to compute average gradients, and then updates parameters locally on each worker.
- Ring all-reduce implements gradient synchronization through scatter-reduce and all-gather phases.
- Compared with a parameter-server pattern, all-reduce data parallelism avoids a central server update/distribution step; each local worker updates parameters after synchronized gradients.

## Chapter 8 Use

- Open the chapter with the bottleneck: data parallelism gives more compute only if gradient synchronization does not dominate the step.
- Use NCCL primitives to define the communication vocabulary before DDP.
- Explain all-reduce as the central collective for synchronous data-parallel gradient averaging.
- Use ring all-reduce to show why communication algorithms are data-movement schedules, not just API calls.
- Contrast all-reduce data parallelism with parameter server only as a systems tradeoff, not a full distributed-ML history.

## Do Not Use As

- A benchmark source for scaling efficiency or speedup.
- A complete specification of NCCL internals.
- A complete treatment of model parallelism, pipeline parallelism, ZeRO, or MoE.

## Candidate Source Cards

- `llmsys-14-nccl-communication`
- `llmsys-14-collective-primitives`
- `llmsys-14-allreduce-semantics`
- `llmsys-14-data-parallel-allreduce`
- `llmsys-14-ring-allreduce-phases`
- `llmsys-14-parameter-server-vs-allreduce`

Owner: Technical Researcher  
Purpose: Chapter 8 source extraction  
Evidence grade: A for course framing; NCCL official docs/source needed for exact API guarantees or implementation details  
Assumptions: Chapter 8 focuses on synchronous data parallelism and DDP, not all distributed-training strategies  
Open questions: Whether to add NCCL official documentation cards before draft for exact collective semantics  
Handoff: Book Architect for Chapter 8 brief
