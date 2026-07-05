# Source Card: llmsys-14-ring-allreduce-phases

Source ID: llmsys-14-ring-allreduce-phases  
Title: Distributed Training  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Lei Li  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-14-distributed-training-b27c3d1dc185e680c6f5cc924e9ec9d7.pdf`  
Pages/slides: Ring AllReduce and implementation sections  
Claim supported: Ring all-reduce can be implemented as scatter-reduce followed by all-gather, moving chunks around a ring of workers.  
Exact quote: "Scatter-reduce"  
Paraphrase: The source walks through ring all-reduce diagrams and pseudocode for scatter-reduce and all-gather phases.  
Evidence grade: A  
Technical risk: Medium; performance claims require topology, message size, implementation, and hardware conditions.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use for mechanism, not speedup.
