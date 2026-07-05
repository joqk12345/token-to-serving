# Source Card: llmsys-15-ddp-autograd-hooks

Source ID: llmsys-15-ddp-autograd-hooks  
Title: Distributed Data Parallel Training  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Lei Li  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-15-ddp-165dbe3873fac21eb8b339e64bcfee28.pdf`  
Pages/slides: DDP Implementation section  
Claim supported: DDP uses autograd hooks to mark gradients ready, decrement bucket pending counts, and trigger all-reduce when a bucket is ready.  
Exact quote: "autograd_hook"  
Paraphrase: The lecture shows reducer pseudocode where `autograd_hook` marks variables ready, buckets track pending gradients, and ready buckets launch communication hooks.  
Evidence grade: A  
Technical risk: Medium; source-code review needed for current implementation details.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Connects framework internals to communication scheduling.
