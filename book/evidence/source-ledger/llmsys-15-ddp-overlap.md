# Source Card: llmsys-15-ddp-overlap

Source ID: llmsys-15-ddp-overlap  
Title: Distributed Data Parallel Training  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Lei Li  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-15-ddp-165dbe3873fac21eb8b339e64bcfee28.pdf`  
Pages/slides: Implementing Distributed Data Parallel and DDP Reduces Latency sections  
Claim supported: DDP overlaps backward computation with all-reduce communication by asynchronously all-reducing a bucket when its gradients are ready.  
Exact quote: "overlap gradient computation and synchronization"  
Paraphrase: The source states that once a bucket's gradients are ready, DDP kicks off asynchronous all-reduce, overlapping communication with remaining backward computation.  
Evidence grade: A  
Technical risk: Medium; performance impact depends on model, bucket order, network, hardware, and implementation.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Main Chapter 8 system-design claim.
