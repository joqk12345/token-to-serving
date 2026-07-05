---
status: brief
chapter: 4
slug: 04-gpu-programming-model
title: Inside the GPU Programming Model
primary_sources:
  - llmsys-02-gpu-programming-c64a0141b96a1f384db7f6717ed8e039.pdf
  - llmsys-03-gpu-programming2-b82b6ffdf554494747d00ce7ac606c3b.pdf
secondary_sources: []
reader_level: engineer or graduate student with basic ML background
technical_depth: introductory-to-intermediate
---

# Inside the GPU Programming Model

## Chapter Thesis

GPU performance starts with a programming model: expose enough parallel work, map threads to data, keep work on the device, and respect the memory hierarchy.

## Reader Problem

The reader has seen Transformers as tensor programs, but may still treat the GPU as a black box that "runs matrix multiplication fast." This chapter should make the execution model legible: host, device, kernels, grids, blocks, threads, warps, SMs, and memory movement.

## System Bottleneck

Primary bottlenecks: programmability, parallelism, memory bandwidth, host-device transfer, and occupancy.

This chapter should not optimize a production GEMM. It should explain why later chapters care about tiling, shared memory, kernel fusion, FlashAttention, and communication overlap.

## Source Map

| Claim | Source card | Evidence grade | Notes |
|---|---|---|---|
| Neural networks reduce to repeated low-level operators. | `llmsys-02-low-level-operators` | A | Bridge from Chapters 2-3. |
| GPU systems include CPUs, GPUs, memory, and interconnects. | `llmsys-02-gpu-server-components` | A | Avoid universal hardware numbers. |
| CUDA separates host code from device kernels. | `llmsys-02-cuda-programming-model` | A | Core concept. |
| GPU execution is organized as grids, blocks, threads, and warps. | `llmsys-02-warp-execution` | A | Define SIMT without overgeneralizing to all accelerators. |
| CPU-GPU data movement can dominate naive programs. | `llmsys-02-cpu-gpu-data-movement` | A | Use as performance warning. |
| CUDA programs allocate device memory, copy data, launch kernels, copy results, and free memory. | `llmsys-03-cuda-memory-lifecycle` | A | Minimal lifecycle. |
| Launch configuration controls grid and block dimensions. | `llmsys-03-launch-configuration` | A | Treat launch shape as a design choice. |
| Built-in thread and block variables map threads to data. | `llmsys-03-thread-indexing` | A | Central code explanation. |
| Vector addition is the minimal example of one-thread-per-output mapping. | `llmsys-03-vector-addition` | A | Use as first code example. |
| Matrix indexing extends the same idea to two dimensions. | `llmsys-03-matrix-indexing` | A | Bridge to tensor computation. |

## Explanation Arc

1. A Transformer block becomes a pile of repeated operators.
2. The GPU is built to run many simple operations in parallel.
3. A CUDA program has two sides: host orchestration and device execution.
4. A kernel launch creates a grid of blocks; blocks contain threads.
5. SMs schedule blocks and execute threads in warps.
6. Each thread uses index arithmetic to decide which data element it owns.
7. Memory movement is part of the program, not an implementation detail.
8. A correct CUDA program is not necessarily a fast CUDA program.
9. The next chapter turns this model into performance reasoning for Transformer kernels.

## Required Figures

| Figure ID | Purpose | Form | Source |
|---|---|---|---|
| `fig-04-host-device-lifecycle` | Show CPU allocation/copy/launch/copy/free flow. | Sequence diagram | `llmsys-03-cuda-memory-lifecycle` |
| `fig-04-grid-block-thread` | Show grid, blocks, threads, and one thread's global index. | Hierarchy diagram | `llmsys-02-warp-execution`, `llmsys-03-thread-indexing` |
| `fig-04-sm-warp-scheduling` | Show blocks assigned to SMs and warps executing inside an SM. | Execution schematic | `llmsys-02-gpu-architecture`, `llmsys-02-warp-execution` |
| `fig-04-memory-paths` | Show system memory, PCIe/NVLink, GPU memory, L2, shared memory, registers. | Memory hierarchy diagram | `llmsys-02-cpu-gpu-data-movement`, `llmsys-02-gpu-server-components` |

## Main Sections

### From Operators to Kernels

Start from operators the reader already knows: linear layers, softmax, reductions, elementwise activation, and attention. Introduce the GPU as the device that executes these operators through kernels.

### Host and Device

Explain that CPU code orchestrates and GPU code computes. Introduce device memory and host-device copies. Make the point that moving data can cost more than the arithmetic in a naive program.

### Grids, Blocks, Threads

Define the launch hierarchy. Use vector addition as the minimal example:

```cuda
int i = blockDim.x * blockIdx.x + threadIdx.x;
if (i < n) {
  C[i] = A[i] + B[i];
}
```

Explain why the bounds check exists.

### Warps and SMs

Introduce SIMT, SMs, and warps. Keep this conceptual. The point is not to teach every scheduler detail; it is to make later performance claims about divergence, occupancy, and memory stalls understandable.

### Memory Is the Real Program Boundary

Explain registers, shared memory, global memory, GPU memory bandwidth, and CPU-GPU transfer. Defer shared-memory tiling and coalescing details to Chapter 5.

### From Correctness to Performance

Close by separating "runs on GPU" from "uses GPU well." This sets up Chapter 5.

## Technical Checks

- Formula correctness: Verify global thread-index equations for 1D and 2D examples.
- Complexity / memory accounting: Avoid giving exact transfer or hardware numbers as universal facts.
- Hardware assumptions: Qualify NVIDIA CUDA-specific concepts; do not imply all accelerators use CUDA terminology.
- Benchmark conditions: Do not include performance numbers in the brief.
- Terminology consistency: Use `host`, `device`, `kernel`, `grid`, `block`, `thread`, `warp`, `SM`, `global memory`, `shared memory`, and `registers` consistently.

## Sidebar Decisions

- Keep H100 Tensor Memory Accelerator as a later optimization note, not a main Chapter 4 concept.
- Do not include Blackwell/Hopper/Ampere performance tables in main text unless backed by vendor documentation.
- Keep compilation commands out of the main flow unless the chapter later includes runnable code.

## Open Questions

- Should Chapter 4 include compilable CUDA code or pseudocode-only examples?
- Should the book assume readers can run CUDA locally, or keep all code examples conceptual until assignments/appendices?

## Handoff

Owner: Book Architect  
Purpose: Chapter 4 brief from GPU programming source notes and source cards  
Evidence grade: A for course-framing claims; official CUDA documentation needed before final API-level publication  
Assumptions: Chapter 4 introduces the programming model; Chapter 5 handles performance tuning  
Handoff: Systems Explainer for draft
