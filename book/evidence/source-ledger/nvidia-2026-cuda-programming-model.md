# Source Card: nvidia-2026-cuda-programming-model

Source ID: nvidia-2026-cuda-programming-model  
Title: CUDA Programming Guide, Programming Model  
Author/issuer: NVIDIA  
Date: 2026  
Source type: documentation  
File path: https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html  
Pages/slides: CUDA Programming Guide v13.3, Section 1.2  
Claim supported: CUDA uses a heterogeneous host/device model, kernels launch many GPU threads, threads are organized into blocks and grids, and SMs execute thread blocks with shared memory available within a block.  
Exact quote: "host and host memory"; "device and device memory"; "thread blocks are organized into a grid"  
Paraphrase: NVIDIA's programming model documentation anchors Chapter 4's host/device, grid/block/thread, SM, shared-memory, warp, and SIMT terminology.  
Evidence grade: A  
Technical risk: Low for current CUDA terminology; hardware-specific limits still vary by architecture.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Official source for Chapter 4 CUDA programming-model claims.
