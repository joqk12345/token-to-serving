# Source Card: llmsys-14-collective-primitives

Source ID: llmsys-14-collective-primitives  
Title: Distributed Training  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Lei Li  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-14-distributed-training-b27c3d1dc185e680c6f5cc924e9ec9d7.pdf`  
Pages/slides: NCCL primitives section  
Claim supported: Distributed training communication uses collectives such as broadcast, reduce, reduce-scatter, all-gather, and all-reduce.  
Exact quote: "AllReduce"  
Paraphrase: The source lists the core NCCL collective primitives used later to explain gradient synchronization.  
Evidence grade: A  
Technical risk: Low.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use before explaining data-parallel all-reduce.
