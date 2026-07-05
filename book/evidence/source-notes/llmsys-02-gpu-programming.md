# Source Note: llmsys-02 GPU Programming

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-02-gpu-programming-c64a0141b96a1f384db7f6717ed8e039.pdf`

## Scope

This source introduces the GPU as the execution substrate for neural network operators. It connects high-level layers to low-level operators, then explains server components, GPU architecture, SMs, warps, grids, blocks, and CPU-GPU data movement.

## Key Claims

- Neural network layers reduce to repeated low-level operators such as matrix multiplication, elementwise operations, reductions, normalization, and softmax.
- GPU servers include CPUs, system memory, PCIe/NVLink interconnects, GPU memory, SMs, L2 cache, and device memory bandwidth.
- GPU programming separates host code running on CPU from device code running as kernels on GPU.
- CUDA kernels run many threads, organized as a grid of thread blocks.
- Thread blocks are scheduled on SMs; within an SM, execution is organized into warps.
- Data movement between CPU memory and GPU memory is a first-order cost, not bookkeeping.

## Chapter 4 Use

- Establish why LLM tensor programs need GPU execution.
- Define host/device split.
- Define grid, block, thread, SM, warp.
- Introduce memory hierarchy enough to prepare for Chapter 5.
- Explain why PCIe/NVLink and GPU memory bandwidth matter before distributed training appears.

## Do Not Use As

- A complete CUDA optimization guide.
- A detailed source for modern Hopper/Blackwell benchmark numbers without checking vendor specifications.
- A replacement for primary CUDA programming documentation when exact API semantics matter.

## Candidate Source Cards

- `llmsys-02-low-level-operators`
- `llmsys-02-gpu-server-components`
- `llmsys-02-gpu-architecture`
- `llmsys-02-cuda-programming-model`
- `llmsys-02-warp-execution`
- `llmsys-02-cpu-gpu-data-movement`

Owner: Technical Researcher  
Purpose: Chapter 4 source extraction  
Evidence grade: A for course framing; verify hardware specifications before printing exact current-device numbers  
Open questions: How much exact hardware table detail belongs in Chapter 4 versus Chapter 5  
Handoff: Book Architect for Chapter 4 brief
