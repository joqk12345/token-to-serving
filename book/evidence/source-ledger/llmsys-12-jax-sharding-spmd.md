# Source Card: llmsys-12-jax-sharding-spmd

Source ID: llmsys-12-jax-sharding-spmd  
Title: Introduction to JAX / XLA / TPU  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Google slide source  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-12-Introduction_to_JAX_XLA_TPU-f0450caf9e7e6707c009f7f77997a2be.pdf`  
Pages/slides: sharding, collective injection, and unified execution model sections  
Claim supported: JAX sharding annotations can be lowered into a global SPMD program where XLA inserts collectives such as all-gather to satisfy partitioned matrix operations.  
Exact quote: "Collective Injection"  
Paraphrase: The source shows sharding metadata preserved through StableHLO and optimized HLO that includes bandwidth-oriented all-gather operations.  
Evidence grade: A  
Technical risk: Medium; detailed distributed semantics should be deferred to later chapters or checked against JAX/XLA docs.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Bridge card from Chapter 7 to Chapters 8–10.
