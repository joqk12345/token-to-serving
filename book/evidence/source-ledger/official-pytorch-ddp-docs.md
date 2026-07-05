# Source Card: official-pytorch-ddp-docs

Source ID: official-pytorch-ddp-docs  
Title: DistributedDataParallel — PyTorch 2.12 documentation  
Author/issuer: PyTorch documentation  
Date: Accessed 2026-07-05  
Source type: official-docs  
File path / URL: `https://docs.pytorch.org/docs/2.12/generated/torch.nn.parallel.DistributedDataParallel.html`  
Pages / slides / sections: `torch.nn.parallel.DistributedDataParallel` reference  
Claim supported: PyTorch `DistributedDataParallel` implements module-level data parallelism by synchronizing gradients across model replicas; DDP exposes arguments including `process_group` and `bucket_cap_mb`.  
Exact quote: "synchronizing gradients"  
Paraphrase: The official PyTorch reference defines DDP as data parallelism based on `torch.distributed`, with gradient synchronization across replicas and constructor options relevant to process groups and bucket sizing.  
Evidence grade: A  
Technical sensitivity: implementation  
Conditions:
  model:
  hardware: PyTorch supported distributed backends; NCCL recommended for GPUs in docs
  batch size:
  sequence length:
  precision:
  software version: PyTorch 2.12 documentation
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use for API-level claims; avoid unqualified speed claims from docs without benchmark context.
