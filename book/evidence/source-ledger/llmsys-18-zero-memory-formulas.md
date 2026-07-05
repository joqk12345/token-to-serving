# Source Card: llmsys-18-zero-memory-formulas

Source ID: llmsys-18-zero-memory-formulas  
Title: Memory Optimization in Distributed Training  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2025  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-18-zero-20eb6c8d8c1e7092e1b922abf03d8cdd.pdf`  
Pages / slides / sections: Slide 64  
Claim supported: The lecture gives symbolic per-device memory formulas for original DDP and ZeRO stages using `N` parameters, optimizer bytes per parameter `M`, and `K` workers.  
Exact quote: "ZeRO-3: (4+M)*N / K"  
Paraphrase: Under the lecture's simplified accounting, successive ZeRO stages divide more categories of model state by the data-parallel worker count.  
Evidence grade: A  
Technical sensitivity: formula | memory accounting  
Conditions:
  model: LLM with N parameters
  hardware: K GPU workers
  batch size:
  sequence length:
  precision: lecture formula assumptions
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: If used in prose, state assumptions and exclude activations/fragments unless included separately.
