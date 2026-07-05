# Source Card: llmsys-03-thread-indexing

Source ID: llmsys-03-thread-indexing  
Title: GPU Programming 2  
Author/issuer: CMU 11868/11968 LLM Systems course staff  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-03-gpu-programming2-b82b6ffdf554494747d00ce7ac606c3b.pdf`  
Pages/slides: device runtime variables and indexing examples  
Claim supported: CUDA exposes built-in variables for grid dimensions, block indices, block dimensions, and thread indices.  
Exact quote: "gridDim"; "blockIdx"; "blockDim"; "threadIdx"  
Paraphrase: The source shows how each GPU thread computes which element of the input or output it owns.  
Evidence grade: A  
Technical risk: Low.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Central for explaining why CUDA code uses index arithmetic.
