# Source Card: llmsys-14-data-parallel-allreduce

Source ID: llmsys-14-data-parallel-allreduce  
Title: Distributed Training  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Lei Li  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-14-distributed-training-b27c3d1dc185e680c6f5cc924e9ec9d7.pdf`  
Pages/slides: Data Parallel Training section  
Claim supported: In data-parallel training, workers process data partitions independently, compute local gradients, all-reduce to average gradients, and then update parameters locally.  
Exact quote: "AllReduce (compute average grad)"  
Paraphrase: The lecture shows data partitioning across workers, local gradient computation, all-reduce gradient averaging, and local parameter updates.  
Evidence grade: A  
Technical risk: Medium; optimizer equivalence depends on synchronization and numerical details.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Main Chapter 8 systems pattern.
