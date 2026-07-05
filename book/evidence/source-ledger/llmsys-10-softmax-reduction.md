# Source Card: llmsys-10-softmax-reduction

Source ID: llmsys-10-softmax-reduction  
Title: Accelerating Transformer Training and Inference  
Author/issuer: CMU 11868/11968 LLM Systems course staff  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-10-transformer-acc-5ba466406bf7296f86cd244ad0405867.pdf`  
Pages/slides: rewrite reduction Softmax forward sections  
Claim supported: Softmax is a row-wise normalization that requires reductions, and optimized kernels tune parameters to shape.  
Exact quote: "Two thread synchronizations"; "Parameters ... are shape dependent"  
Paraphrase: The source presents max and sum-exp reductions as synchronization-heavy parts of softmax.  
Evidence grade: A  
Technical risk: Low for concept; implementation details are shape- and architecture-dependent.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Use for non-GEMM operator discussion.
