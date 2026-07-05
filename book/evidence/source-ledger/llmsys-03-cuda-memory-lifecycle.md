# Source Card: llmsys-03-cuda-memory-lifecycle

Source ID: llmsys-03-cuda-memory-lifecycle  
Title: GPU Programming 2  
Author/issuer: CMU 11868/11968 LLM Systems course staff  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-03-gpu-programming2-b82b6ffdf554494747d00ce7ac606c3b.pdf`  
Pages/slides: CUDA operations and memory management sections  
Claim supported: A basic CUDA program allocates device memory, copies host data to the device, launches a kernel, copies results back, and frees device memory.  
Exact quote: "cudaMalloc"; "cudaMemcpy"; "cudaFree"  
Paraphrase: The source lays out the standard CUDA memory and execution lifecycle from CPU orchestration to GPU computation.  
Evidence grade: A  
Technical risk: Low for introductory CUDA; API details should be checked against official docs before final code publication.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Use for minimal CUDA lifecycle in Chapter 4.
