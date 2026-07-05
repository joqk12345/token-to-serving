# Source Card: llmsys-18-zero-stage1-optimizer

Source ID: llmsys-18-zero-stage1-optimizer  
Title: Memory Optimization in Distributed Training  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2025  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-18-zero-20eb6c8d8c1e7092e1b922abf03d8cdd.pdf`  
Pages / slides / sections: Slides 18-41  
Claim supported: ZeRO stage 1 partitions optimizer states across `K` GPU workers while each worker still has FP16 parameters for forward/backward.  
Exact quote: "Partition the optimizer states to K parts"  
Paraphrase: Stage 1 removes replicated optimizer-state storage but leaves parameters and gradients broadly in the DDP pattern.  
Evidence grade: A  
Technical sensitivity: memory partitioning  
Conditions:
  model: data-parallel training
  hardware: K GPUs in lecture examples
  batch size:
  sequence length:
  precision: lecture mixed-precision framing
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use as staged mechanism, not as a complete implementation trace.
