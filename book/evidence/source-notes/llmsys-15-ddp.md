# Source Note: llmsys-15 Distributed Data Parallel Training

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-15-ddp-165dbe3873fac21eb8b339e64bcfee28.pdf`

## Scope

This source focuses on distributed data parallel training, especially PyTorch DDP design goals, process setup, local model replicas, gradient synchronization, gradient bucketing, asynchronous all-reduce, autograd hooks, and overlap between backward computation and communication.

## Key Claims

- Distributed data parallel training creates model replicas across multiple nodes/GPUs; each replica performs forward and backward independently, gradients are averaged across processes, and optimizers run locally with identical updates.
- PyTorch DDP design goals include being non-intrusive to local training scripts and interceptive enough to trigger internal algorithms promptly.
- A naïve DDP implementation would all-reduce gradients after the entire backward pass finishes.
- DDP improves on that by overlapping gradient computation and communication.
- Synchronizing per parameter would create too much communication overhead; DDP groups gradients into buckets.
- Bucket size is configurable through `bucket_cap_mb`; bucket assignment is determined at construction time based on bucket size and parameter sizes.
- When all gradients in a bucket are ready, the reducer launches an asynchronous all-reduce for that bucket.
- DDP uses autograd hooks to notice when parameter gradients have accumulated and to trigger bucket readiness.

## Chapter 8 Use

- Use DDP as the concrete system design case after explaining all-reduce.
- Explain why the problem is not just “average gradients,” but “average gradients without leaving communication idle at the end of backward.”
- Use gradient buckets as the main scheduling mechanism.
- Use autograd hooks to connect framework internals to distributed communication.
- Keep PyTorch API snippets illustrative; exact API behavior should be checked against official docs if draft becomes API-specific.

## Do Not Use As

- A current PyTorch API reference.
- A performance benchmark source.
- A source for every DDP optimization in modern PyTorch.

## Candidate Source Cards

- `llmsys-15-ddp-replica-gradient-average`
- `llmsys-15-ddp-design-goals`
- `llmsys-15-ddp-naive-allreduce`
- `llmsys-15-ddp-bucketing`
- `llmsys-15-ddp-overlap`
- `llmsys-15-ddp-autograd-hooks`

Owner: Technical Researcher  
Purpose: Chapter 8 source extraction  
Evidence grade: A for course framing; PyTorch official docs and Li et al. VLDB 2020 paper needed for API/publication-level details  
Assumptions: Chapter 8 uses PyTorch DDP as the main data-parallel implementation case study  
Open questions: Add the PyTorch Distributed VLDB 2020 paper card before draft if DDP design goals become central  
Handoff: Book Architect for Chapter 8 brief
