# Source Card: llmsys-10-fused-embedding

Source ID: llmsys-10-fused-embedding  
Title: Accelerating Transformer Training and Inference  
Author/issuer: CMU 11868/11968 LLM Systems course staff  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-10-transformer-acc-5ba466406bf7296f86cd244ad0405867.pdf`  
Pages/slides: fused embedding forward and backward sections  
Claim supported: Embedding lookup, positional embedding, scaling, dropout, gradient aggregation, and related steps can be fused to reduce kernel launches and intermediate IO.  
Exact quote: "5 cuda kernel launches"; "1 cuda kernel launch"; "Less IO for intermediate results"  
Paraphrase: The source presents fused embedding forward/backward operators as an example of Transformer-specific fusion.  
Evidence grade: A  
Technical risk: Medium; use conceptually unless LightSeq source/paper is reviewed.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Use as optional concrete example.
