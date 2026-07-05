# Source Card: llmsys-16-tensor-parallel-embeddings

Source ID: llmsys-16-tensor-parallel-embeddings  
Title: Model Parallel Training  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Lei Li  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-16-model-parallel-83b41612547620ee0e172caa1ee448ed.pdf`  
Pages/slides: Tensor Parallelism - Embeddings section  
Claim supported: Input and output embeddings have distinct tensor-parallel communication patterns; input embeddings may require all-reduce, while output embeddings can fuse with cross-entropy to reduce communication before all-gather.  
Exact quote: "all-reduce is required"  
Paraphrase: The lecture distinguishes input embedding and output embedding partitioning and notes cross-entropy fusion can reduce output communication.  
Evidence grade: A  
Technical risk: High; draft needs careful wording and likely Megatron/source-code support for exact implementation.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Optional; use cautiously.
