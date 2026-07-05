# Source Note: llmsys-03 GPU Programming 2

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-03-gpu-programming2-b82b6ffdf554494747d00ce7ac606c3b.pdf`

## Scope

This source continues the GPU programming model with CUDA operations, device memory allocation, host-device copies, kernel declarations, launch configuration, built-in indexing variables, and vector/matrix examples.

## Key Claims

- A CUDA program typically allocates device memory, copies host data to the device, launches kernels, copies results back, and frees device memory.
- CUDA functions use qualifiers such as `__global__`, `__device__`, and `__host__` to describe where code is called and executed.
- Kernel launch configuration specifies grid dimensions and block dimensions.
- Built-in variables such as `gridDim`, `blockIdx`, `blockDim`, and `threadIdx` let each thread identify the data element it owns.
- Simple vector addition and matrix addition examples show how index arithmetic maps parallel threads to data.
- H100 introduces advanced mechanisms such as Tensor Memory Accelerator, which should be treated as later optimization context rather than the first programming model.

## Chapter 4 Use

- Explain the minimal CUDA lifecycle.
- Show one-dimensional and two-dimensional indexing patterns.
- Introduce launch configuration as an explicit performance and correctness decision.
- Use vector addition as the minimal example; defer matrix multiplication optimization to Chapter 5.

## Do Not Use As

- A full treatment of asynchronous Hopper features.
- A full matrix multiplication optimization guide.
- The final authority on CUDA API behavior; exact API details should be checked against NVIDIA documentation before publication.

## Candidate Source Cards

- `llmsys-03-cuda-memory-lifecycle`
- `llmsys-03-kernel-qualifiers`
- `llmsys-03-launch-configuration`
- `llmsys-03-thread-indexing`
- `llmsys-03-vector-addition`
- `llmsys-03-matrix-indexing`

Owner: Technical Researcher  
Purpose: Chapter 4 source extraction  
Evidence grade: A for course framing; CUDA API details should be checked against official documentation before final publication  
Open questions: Whether Chapter 4 should include compilable CUDA snippets or pseudocode only  
Handoff: Book Architect for Chapter 4 brief
