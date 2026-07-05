# Source Card: llmsys-18-ddp-memory-accounting

Source ID: llmsys-18-ddp-memory-accounting  
Title: Memory Optimization in Distributed Training  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2025  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-18-zero-20eb6c8d8c1e7092e1b922abf03d8cdd.pdf`  
Pages / slides / sections: Slides 6-13  
Claim supported: Mixed-precision DDP memory includes parameters, gradients, optimizer states, activations, temporary buffers, and fragmentation.  
Exact quote: "FP16 parameters, FP16 Gradients, FP32 Optimizer States"  
Paraphrase: DDP's memory problem is a ledger of replicated model-related states, not just parameter tensors.  
Evidence grade: A  
Technical sensitivity: memory accounting  
Conditions:
  model: mixed-precision Transformer training
  hardware:
  batch size:
  sequence length:
  precision: FP16 parameters/gradients and FP32 optimizer-related states in lecture framing
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Numeric examples require carrying model and precision assumptions.
