# Source Card: llmsys-14-parameter-server-vs-allreduce

Source ID: llmsys-14-parameter-server-vs-allreduce  
Title: Distributed Training  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Lei Li  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-14-distributed-training-b27c3d1dc185e680c6f5cc924e9ec9d7.pdf`  
Pages/slides: Parameter Server vs AllReduce Data Parallel section  
Claim supported: Parameter-server training synchronizes parameters and local gradients through a server, while all-reduce data parallelism has local workers update parameters after gradient synchronization.  
Exact quote: "needs to synchronize twice"  
Paraphrase: The lecture contrasts central server update/distribution with local updates after all-reduce gradient averaging.  
Evidence grade: A  
Technical risk: Medium; current production systems may use variants and hybrids.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use as a conceptual contrast only.
