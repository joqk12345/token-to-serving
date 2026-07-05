# Source Card: llmsys-15-ddp-naive-allreduce

Source ID: llmsys-15-ddp-naive-allreduce  
Title: Distributed Data Parallel Training  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Lei Li  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-15-ddp-165dbe3873fac21eb8b339e64bcfee28.pdf`  
Pages/slides: How to Implement Distributed Data Parallel section  
Claim supported: A naïve DDP implementation would synchronize gradients with all-reduce after the entire backward pass finishes.  
Exact quote: "after the entire backward pass finishes"  
Paraphrase: The lecture presents post-backward gradient synchronization as the baseline implementation to improve upon.  
Evidence grade: A  
Technical risk: Low.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use as the concrete bottleneck setup.
