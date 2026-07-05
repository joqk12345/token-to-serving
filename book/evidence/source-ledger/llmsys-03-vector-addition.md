# Source Card: llmsys-03-vector-addition

Source ID: llmsys-03-vector-addition  
Title: GPU Programming 2  
Author/issuer: CMU 11868/11968 LLM Systems course staff  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-03-gpu-programming2-b82b6ffdf554494747d00ce7ac606c3b.pdf`  
Pages/slides: vector addition sections  
Claim supported: Vector addition is a minimal CUDA example where each thread computes one output element using its thread and block indices.  
Exact quote: "VecAddKernel"; "int i = blockDim.x * blockIdx.x + threadIdx.x"  
Paraphrase: The source demonstrates mapping a one-dimensional data-parallel operation onto CUDA threads.  
Evidence grade: A  
Technical risk: Low.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Use as the primary concrete code example for Chapter 4.
