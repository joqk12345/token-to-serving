# Source Card: llmsys-18-zero-key-idea

Source ID: llmsys-18-zero-key-idea  
Title: Memory Optimization in Distributed Training  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2025  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-18-zero-20eb6c8d8c1e7092e1b922abf03d8cdd.pdf`  
Pages / slides / sections: Slides 16-17, 75  
Claim supported: ZeRO reduces DDP memory redundancy by partitioning optimizer states, gradients, and parameters.  
Exact quote: "partitioning the optimizer states (zero-1), gradients (zero-2), parameters (zero-3)"  
Paraphrase: ZeRO keeps data-parallel training semantics while removing duplicated state across data-parallel workers in stages.  
Evidence grade: A  
Technical sensitivity: distributed training | memory  
Conditions:
  model: data-parallel distributed training
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Do not claim zero extra communication for all stages.
