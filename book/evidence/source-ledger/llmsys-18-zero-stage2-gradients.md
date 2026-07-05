# Source Card: llmsys-18-zero-stage2-gradients

Source ID: llmsys-18-zero-stage2-gradients  
Title: Memory Optimization in Distributed Training  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2025  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-18-zero-20eb6c8d8c1e7092e1b922abf03d8cdd.pdf`  
Pages / slides / sections: Slides 42-50  
Claim supported: ZeRO stage 2 partitions gradients so each GPU computes gradients for its data partition but keeps only the gradient partition it owns.  
Exact quote: "only stores one partition of gradients instead of all gradients"  
Paraphrase: Stage 2 reduces persistent gradient memory by reducing and retaining gradient shards on their responsible workers.  
Evidence grade: A  
Technical sensitivity: memory partitioning | communication  
Conditions:
  model: data-parallel training
  hardware: K GPUs in lecture examples
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Be careful to distinguish temporary gradient buffers from retained gradient shards.
