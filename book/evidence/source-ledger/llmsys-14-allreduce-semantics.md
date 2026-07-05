# Source Card: llmsys-14-allreduce-semantics

Source ID: llmsys-14-allreduce-semantics  
Title: Distributed Training  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Lei Li  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-14-distributed-training-b27c3d1dc185e680c6f5cc924e9ec9d7.pdf`  
Pages/slides: AllReduce, ReduceScatter, AllGather sections  
Claim supported: All-reduce computes a reduction across devices and makes the result available to every rank; it can also be described as reduce-scatter followed by all-gather.  
Exact quote: "AllReduce = ReduceScatter & AllGather"  
Paraphrase: The lecture defines all-reduce as reduction plus distribution of the result and decomposes it into reduce-scatter and all-gather phases.  
Evidence grade: A  
Technical risk: Low for concept; implementation details vary.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Central mathematical/communication primitive card.
