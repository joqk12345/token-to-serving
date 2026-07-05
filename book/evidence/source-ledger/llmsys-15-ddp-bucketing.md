# Source Card: llmsys-15-ddp-bucketing

Source ID: llmsys-15-ddp-bucketing  
Title: Distributed Data Parallel Training  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Lei Li  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-15-ddp-165dbe3873fac21eb8b339e64bcfee28.pdf`  
Pages/slides: Gradient Bucketing section  
Claim supported: DDP groups gradients into buckets; bucket size can be configured through `bucket_cap_mb`, and bucket assignment is determined at construction time from bucket size and parameter sizes.  
Exact quote: "bucket_cap_mb"  
Paraphrase: The lecture explains that DDP does not all-reduce every parameter separately; it groups gradients into buckets based on configured size limits and parameter sizes.  
Evidence grade: A  
Technical risk: Medium; current PyTorch behavior should be verified against official docs/source for API-level claims.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Central DDP scheduling card.
