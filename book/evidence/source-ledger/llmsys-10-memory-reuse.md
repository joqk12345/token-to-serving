# Source Card: llmsys-10-memory-reuse

Source ID: llmsys-10-memory-reuse  
Title: Accelerating Transformer Training and Inference  
Author/issuer: CMU 11868/11968 LLM Systems course staff  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-10-transformer-acc-5ba466406bf7296f86cd244ad0405867.pdf`  
Pages/slides: memory management for self-attention backward and inference details  
Claim supported: Transformer training and inference can reduce memory allocation by reusing tensor storage once values are no longer needed.  
Exact quote: "Reuse Memory"; "No need to keep intermediate results and gradients during infernece"  
Paraphrase: The source uses self-attention backward and inference implementation details to show memory reuse as an optimization technique.  
Evidence grade: A  
Technical risk: Medium; exact liveness depends on implementation and autograd/runtime semantics.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Use for liveness and memory-planning discussion.
