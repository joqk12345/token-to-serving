# Source Card: llmsys-04-tiling-shared-memory

Source ID: llmsys-04-tiling-shared-memory  
Title: GPU Acceleration  
Author/issuer: CMU 11868/11968 LLM Systems course staff  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-04-gpu-acceleration-48ffa5768ba62c54138f0a71ca2b68b8.pdf`  
Pages/slides: tiling for matrix multiplication section  
Claim supported: Tiled matrix multiplication loads input tiles into shared memory, synchronizes threads, computes partial sums, and repeats for later tiles.  
Exact quote: "load the first tile of each input matrix to shared memory"; "__syncthreads()"  
Paraphrase: The source presents tiling as a reuse strategy that reduces global-memory traffic during matrix multiplication.  
Evidence grade: A  
Technical risk: Low for conceptual tiling; production GEMM is more complex.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Core Chapter 5 source.
