# Source Card: llmsys-15-ddp-replica-gradient-average

Source ID: llmsys-15-ddp-replica-gradient-average  
Title: Distributed Data Parallel Training  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Lei Li  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-15-ddp-165dbe3873fac21eb8b339e64bcfee28.pdf`  
Pages/slides: Distributed Data Parallel section  
Claim supported: DDP creates model replicas across nodes/GPUs; each replica runs forward and backward independently, average gradients are gathered across nodes, and optimizers run locally with identical updates.  
Exact quote: "Gather average gradients across nodes"  
Paraphrase: The lecture describes DDP as replicated local training plus synchronized gradient averaging and local optimizer steps.  
Evidence grade: A  
Technical risk: Medium; exact equivalence depends on synchronous execution and optimizer/state handling.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Main DDP concept card.
