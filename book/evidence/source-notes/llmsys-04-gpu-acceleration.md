# Source Note: llmsys-04 GPU Acceleration

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-04-gpu-acceleration-48ffa5768ba62c54138f0a71ca2b68b8.pdf`

## Scope

This source introduces GPU acceleration techniques after the basic CUDA programming model: arithmetic intensity, memory-bound kernels, tiling, shared memory, coalesced memory access, bank conflicts, sparse matrix formats, and cuBLAS.

## Key Claims

- Memory access efficiency is critical because kernels may be bounded by load/store bandwidth rather than peak FLOPs.
- Naive matrix multiplication has poor compute-to-global-memory-access ratio because each multiply-add may reload operands from global memory.
- Tiling improves reuse by loading blocks of input matrices into shared memory and reusing them across multiple operations.
- Coalesced memory access lets consecutive accesses in a warp be served efficiently by memory transactions.
- Matrix transpose illustrates how a correct kernel may have inefficient memory access, and how shared memory can reshape access.
- Shared memory bank conflicts can serialize accesses; padding can avoid some conflict patterns.
- cuBLAS provides optimized BLAS routines such as dot, GEMV, and GEMM.

## Chapter 5 Use

- Build the memory-bound vs compute-bound distinction.
- Explain arithmetic intensity without turning it into a full roofline chapter.
- Use matrix multiplication tiling as the core concrete example.
- Use transpose as the clearest example of memory layout and coalescing.
- Introduce library kernels as an engineering boundary: use cuBLAS when possible; write custom kernels when operator structure demands it.

## Do Not Use As

- A universal benchmark source for specific A100/H100 numbers.
- A full sparse linear algebra chapter.
- A cuBLAS API reference.

## Candidate Source Cards

- `llmsys-04-memory-access-efficiency`
- `llmsys-04-naive-matmul-intensity`
- `llmsys-04-tiling-shared-memory`
- `llmsys-04-coalesced-access`
- `llmsys-04-bank-conflict`
- `llmsys-04-cublas`

Owner: Technical Researcher  
Purpose: Chapter 5 source extraction  
Evidence grade: A for course framing; exact hardware numbers need vendor confirmation before publication  
Open questions: Whether sparse matrix multiplication belongs in Chapter 5 main text or a sidebar  
Handoff: Book Architect for Chapter 5 brief
