# Source Card: llmsys-04-cublas

Source ID: llmsys-04-cublas  
Title: GPU Acceleration  
Author/issuer: CMU 11868/11968 LLM Systems course staff  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-04-gpu-acceleration-48ffa5768ba62c54138f0a71ca2b68b8.pdf`  
Pages/slides: cuBLAS section  
Claim supported: cuBLAS provides optimized BLAS operations including vector dot product, matrix-vector product, and matrix-matrix multiplication.  
Exact quote: "CUDA Basic Linear Algebra Subroutine library"; "Matrix matrix multiplication"  
Paraphrase: The source frames cuBLAS as the standard library path for optimized dense linear algebra on CUDA.  
Evidence grade: A  
Technical risk: Low for concept; exact API reference should use NVIDIA docs.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Use to explain when not to write custom GEMM.
