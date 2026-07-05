# Source Card: llmsys-12-xla-optimization-passes

Source ID: llmsys-12-xla-optimization-passes  
Title: Introduction to JAX / XLA / TPU  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Google slide source  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-12-Introduction_to_JAX_XLA_TPU-f0450caf9e7e6707c009f7f77997a2be.pdf`  
Pages/slides: HLO optimization passes section  
Claim supported: XLA optimization includes graph simplification, fusion, layout and tiling decisions, buffer/copy insertion, and memory-space assignment.  
Exact quote: "Buffer & Copy Insertion"  
Paraphrase: The source lists compiler passes that turn high-level tensor operations into a scheduled, memory-aware executable plan.  
Evidence grade: A  
Technical risk: Medium; individual pass behavior should be checked against XLA docs/source for detailed claims.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use to explain why compiler choices affect memory traffic.
