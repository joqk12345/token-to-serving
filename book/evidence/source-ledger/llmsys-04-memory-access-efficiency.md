# Source Card: llmsys-04-memory-access-efficiency

Source ID: llmsys-04-memory-access-efficiency  
Title: GPU Acceleration  
Author/issuer: CMU 11868/11968 LLM Systems course staff  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-04-gpu-acceleration-48ffa5768ba62c54138f0a71ca2b68b8.pdf`  
Pages/slides: memory access efficiency section  
Claim supported: GPU kernels may be bounded by memory load/store bandwidth, not peak arithmetic throughput.  
Exact quote: "may also be bounded by memory load/store"  
Paraphrase: The source introduces compute-to-global-memory-access ratio as a way to reason about whether a kernel is memory-bound.  
Evidence grade: A  
Technical risk: Low for concept; avoid printing hardware-specific numbers as universal.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Use to open Chapter 5.
