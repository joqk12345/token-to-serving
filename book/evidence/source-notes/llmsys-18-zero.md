# Source Note: llmsys-18 ZeRO

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-18-zero-20eb6c8d8c1e7092e1b922abf03d8cdd.pdf`

## Scope

This source covers memory consumption in mixed-precision distributed data-parallel training, ZeRO stages 1-3, optimizer-state partitioning, gradient partitioning, parameter partitioning, activation checkpointing, constant-size buffers, memory defragmentation, ZeRO++ communication reduction, and composition with pipeline/tensor parallelism.

## Key Claims

- DDP has good compute/communication efficiency but poor memory efficiency because each device stores a copy of model-related state.
- Mixed-precision DDP memory includes FP16 parameters, FP16 gradients, FP32 optimizer-related state, activations, temporary buffers, and fragmentation.
- ZeRO reduces DDP memory redundancy by partitioning optimizer states, gradients, and parameters across data-parallel workers.
- ZeRO stage 1 partitions optimizer states.
- ZeRO stage 2 partitions gradients: each GPU computes gradients for its data shard but stores only the gradient partition it owns, sending other gradients to responsible GPUs.
- ZeRO stage 3 partitions parameters and gathers/broadcasts parameter partitions when needed for forward/backward computation.
- The lecture gives symbolic memory accounting for ZeRO stages using parameter count `N`, optimizer bytes per parameter `M`, and `K` GPUs.
- ZeRO stage 3 trades lower resident memory for additional parameter transfer.
- Other memory optimizations include partitioned activation checkpointing, constant-size buffers, memory defragmentation, and communication reduction techniques.

## Chapter 10 Use

- Open the ZeRO half from a memory ledger, not from an algorithm list.
- Reuse Chapter 8's DDP boundary: data parallelism replicates model state.
- Explain ZeRO stages as removing three classes of redundancy: optimizer states, gradients, and parameters.
- Use formulas only with the lecture's assumptions: `N` parameters, optimizer state size `M` bytes per parameter, `K` workers, and mixed-precision-style accounting.
- Treat ZeRO-3 communication as a tradeoff, not a free memory reduction.
- Leave exact ZeRO++ and offload performance claims out of scope unless original papers are added.

## Do Not Use As

- A source for universal memory savings independent of optimizer, precision, or worker count.
- A complete implementation reference for DeepSpeed or PyTorch FSDP.
- A benchmark source without original experimental setup.

## Candidate Source Cards

- `llmsys-18-ddp-memory-accounting`
- `llmsys-18-zero-key-idea`
- `llmsys-18-zero-stage1-optimizer`
- `llmsys-18-zero-stage2-gradients`
- `llmsys-18-zero-stage3-parameters`
- `llmsys-18-zero-memory-formulas`
- `llmsys-18-zero-communication-cost`
- `llmsys-18-zero-other-memory-optimizations`

Owner: Technical Researcher  
Purpose: Chapter 10 ZeRO source extraction  
Evidence grade: A for course framing; original ZeRO paper needed for publication-level algorithm claims  
Assumptions: Chapter 10 treats ZeRO as data-parallel state partitioning, distinct from Chapter 9 model parallelism  
Open questions: Whether to include the lecture's memory formulas in the main chapter or keep them in a boxed derivation  
Handoff: Book Architect for Chapter 10 brief
