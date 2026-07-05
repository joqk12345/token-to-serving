# Source Card: llmsys-12-xla-attention-fusion

Source ID: llmsys-12-xla-attention-fusion  
Title: Introduction to JAX / XLA / TPU  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Google slide source  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-12-Introduction_to_JAX_XLA_TPU-f0450caf9e7e6707c009f7f77997a2be.pdf`  
Pages/slides: attention fusion and softmax data path sections  
Claim supported: XLA can reduce memory traffic in attention-like computations by fusing operations and avoiding materialization of large intermediate matrices when the optimized HLO permits it.  
Exact quote: "never writing the massive"  
Paraphrase: The source describes attention/softmax fusion where logits or exponential intermediates are consumed through on-chip paths instead of being written out as large temporary matrices.  
Evidence grade: A  
Technical risk: High; exact behavior is workload, shape, precision, backend, and compiler-version dependent.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use as a conditional compiler example; do not generalize to all attention programs.
