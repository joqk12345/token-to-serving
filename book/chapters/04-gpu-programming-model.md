---
status: ready
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

The previous chapters turned text into a workload. Tokens become vectors. Vectors pass through attention, feed-forward layers, normalization, softmax, and output projections. Decoding repeats part of that computation one token at a time.

This chapter asks where that computation runs.

A GPU is not a magic box for matrix multiplication. It is a parallel processor with its own memory, execution hierarchy, programming model, and failure modes. To use it well, a program must expose parallel work, map that work onto threads, keep data movement under control, and respect the hardware's memory hierarchy.

The goal here is not to write a production Transformer kernel. The goal is to make the GPU programming model legible enough that later chapters can talk precisely about memory bandwidth, kernel fusion, tiling, FlashAttention, and serving bottlenecks.

## Operators Become Kernels

From the model's point of view, a Transformer block contains familiar operations:

- matrix multiplication;
- elementwise operations such as add, scale, and activation functions;
- reductions such as sum and average;
- normalization;
- softmax;
- memory movement.

These are the same low-level operator families that appear in simpler neural networks. A feed-forward classifier may use embeddings, linear layers, ReLU, average pooling, and softmax. A Transformer uses larger and more structured versions of the same computational ingredients. [CITE: llmsys-02-low-level-operators]

On a GPU, these operators run as kernels. A kernel is a function executed by many GPU threads. The source code for a single thread may look serial, but the launch creates many instances of that code running across different data elements. [CITE: llmsys-02-cuda-programming-model]

This is the first shift in thinking: GPU programming is not mainly about writing one clever loop. It is about describing a large set of similar operations so the device can run them in parallel.

## Host and Device

CUDA programs have two sides.

![A minimal CUDA program is split between host orchestration and device execution: the CPU allocates device memory, copies inputs to the GPU, launches a kernel, copies results back, and frees device memory.](../figures/artwork/ch04/fig-04-host-device-lifecycle.svg)

The CPU is the **host**. It runs the ordinary program, prepares data, allocates memory, launches kernels, and coordinates the computation. The GPU is the **device**. It runs kernel code over many threads. [CITE: llmsys-02-cuda-programming-model]

A minimal CUDA workflow looks like this:

```text
allocate device memory
copy input data from host memory to device memory
launch a kernel on the device
copy output data from device memory to host memory
free device memory
```

The common CUDA calls in that lifecycle are `cudaMalloc`, `cudaMemcpy`, kernel launch syntax, and `cudaFree`. [CITE: llmsys-03-cuda-memory-lifecycle]

This lifecycle is simple, but it already contains the performance trap. Copying data between CPU memory and GPU memory is not free. If a program repeatedly moves small tensors across the host-device boundary, it may spend more time moving data than computing. [CITE: llmsys-02-cpu-gpu-data-movement]

For LLM systems, this lesson scales up. We care not only about how many floating-point operations a model requires, but also where tensors live and how often they move. A serving system that constantly moves weights, activations, or KV cache blocks across slow paths will waste hardware even if the math kernels are fast.

## Launching Parallel Work

A kernel launch specifies how many GPU threads should run and how those threads are grouped.

![A CUDA kernel launch creates a grid of thread blocks. Each thread computes its identity from block and thread indices, then uses that identity to choose the data element it owns.](../figures/artwork/ch04/fig-04-grid-block-thread.svg)

CUDA organizes work as:

```text
grid -> blocks -> threads
```

A grid contains many thread blocks. A block contains many threads. The host chooses the grid dimensions and block dimensions when it launches the kernel. [CITE: llmsys-02-warp-execution; llmsys-03-launch-configuration]

The launch configuration is not decoration. It decides how the program's logical work is divided across the device. A vector operation with one million elements needs enough threads to cover one million outputs. A matrix operation often uses two-dimensional blocks and grids so that thread coordinates map naturally to rows and columns.

The key idea is that each thread needs to know which piece of data it owns.

## One Thread, One Output

Vector addition is the smallest useful CUDA example. Given arrays `A`, `B`, and `C`, each output element can be computed independently:

```text
C[i] = A[i] + B[i]
```

On the GPU, a common mapping is one thread per output element. Each thread computes its global index from its block and thread position:

```cuda
int i = blockDim.x * blockIdx.x + threadIdx.x;
if (i < n) {
  C[i] = A[i] + B[i];
}
```

The built-in variables expose the launch structure. `blockIdx.x` identifies the block. `threadIdx.x` identifies the thread within that block. `blockDim.x` tells the thread how many threads are in each block. [CITE: llmsys-03-thread-indexing; llmsys-03-vector-addition]

The bounds check matters because launches are usually rounded up. If `n = 1000` and the program launches blocks of 256 threads, it needs four blocks, or 1024 total threads. The last 24 threads do not own valid output elements. They must exit without writing.

This tiny example contains the core CUDA habit:

```text
thread identity -> data index -> guarded computation
```

Most GPU kernels are more complex, but they still begin with this mapping problem.

## From Vectors to Matrices

For a matrix, one-dimensional indexing is no longer the most natural shape. A matrix element has a row and a column.

A two-dimensional launch can map one thread to one matrix element:

```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

if (row < n && col < n) {
  C[row * n + col] = A[row * n + col] + B[row * n + col];
}
```

The principle is unchanged. Each thread computes an identity from the launch geometry and uses that identity to decide which output element to write. [CITE: llmsys-03-matrix-indexing]

Matrix addition is still not matrix multiplication. It has little data reuse and little arithmetic per element. Matrix multiplication is more interesting because each output element depends on many input elements, and good performance depends on reusing data that would otherwise be read repeatedly. That is a Chapter 5 problem.

For now, the important point is that tensors are not abstract once they reach the GPU. Their shape affects the launch shape. Their layout affects memory access. Their reuse pattern affects whether the kernel is limited by arithmetic or by data movement.

## SMs, Warps, and SIMT

Threads do not float around independently. GPU hardware groups and schedules them.

![Thread blocks are scheduled onto streaming multiprocessors, and threads within a block execute in warps under CUDA's SIMT model.](../figures/artwork/ch04/fig-04-sm-warp-scheduling.svg)

NVIDIA GPUs are built from streaming multiprocessors, or SMs. Thread blocks are assigned to SMs. Within an SM, threads execute in groups called warps. In CUDA, a warp contains 32 threads. Those threads execute in a single-instruction, multiple-thread style: the warp issues one instruction across its active threads. [CITE: llmsys-02-gpu-architecture; llmsys-02-warp-execution]

This model explains several later performance ideas.

First, a GPU needs enough active warps to hide latency. If one warp waits for memory, the scheduler can run another warp. This is why occupancy matters, although high occupancy alone does not guarantee high performance.

Second, branch behavior matters. If threads in the same warp take different branches, execution may have to cover both paths for different active lanes. That can reduce efficiency.

Third, resource use matters. Registers and shared memory are limited. A kernel that uses too many per-thread or per-block resources may reduce how many warps can reside on an SM at once.

This chapter does not need every scheduler detail. It needs the durable mental model: the GPU runs many threads, but the hardware schedules them in structured groups with finite local resources.

## Memory Is Part of the Program

A fast GPU program often starts by avoiding unnecessary movement.

![GPU performance depends on where data lives: host memory, interconnects, GPU global memory, caches, shared memory, and registers all impose different movement costs and visibility rules.](../figures/artwork/ch04/fig-04-memory-paths.svg)

The memory hierarchy includes several levels:

- registers, private to a thread and very fast;
- shared memory, visible to threads in a block;
- global GPU memory, visible across kernels and threads;
- caches such as L2;
- host system memory, reached through CPU-GPU transfer paths.

Registers, shared memory, global GPU memory, L2 cache, system memory, and interconnect bandwidth belong to the same performance story. [CITE: llmsys-02-gpu-architecture; llmsys-02-cpu-gpu-data-movement]

This is where a correct CUDA program can be far from a fast CUDA program.

A naive kernel may launch many threads and still perform poorly because each thread reads scattered data from global memory. Another kernel may do the same arithmetic but load tiles into shared memory, reuse them, and reduce global-memory traffic. Another may fuse several elementwise operations so intermediate tensors never need to be written out and read back.

These choices are not micro-optimizations for LLM systems. They determine whether Transformer blocks run near hardware capability or spend most of their time waiting for data.

## The GPU Server Is a System

An LLM usually runs on a server, not on an isolated GPU diagram.

A modern GPU server includes CPUs, system memory, storage, GPUs, GPU memory, and interconnects such as PCIe or NVLink. [CITE: llmsys-02-gpu-server-components]

That matters because LLM workloads cross boundaries:

- input data moves from storage and host memory toward the GPU;
- training gradients may move between GPUs;
- model-parallel layers may exchange activations;
- inference systems may move KV cache blocks or route requests across devices;
- checkpoints may move large parameter states through storage and network paths.

Chapter 1 framed LLM systems as compute, memory, bandwidth, communication, and scheduling. The GPU programming model gives those words a concrete substrate. There is device memory, host memory, interconnect bandwidth, SM scheduling, warp execution, and kernel launch overhead.

Once these are visible, "make it faster" becomes a sharper question: faster arithmetic, fewer memory reads, better data reuse, fewer host-device transfers, more useful occupancy, less synchronization, or better scheduling?

## Correct Is Not Fast

The first goal of GPU programming is correctness: launch enough threads, index the right elements, avoid out-of-bounds writes, copy the right data, and free memory when done.

The second goal is performance. That is harder.

A correct vector-add kernel teaches the programming model, but it does not teach high-performance LLM kernels. Transformer workloads need better answers to harder questions:

- How are matrix multiplications tiled?
- Which tensors fit in shared memory?
- Which operations are memory-bandwidth bound?
- Which elementwise operations can be fused?
- Which intermediate values must be materialized?
- Which memory layout gives coalesced access?
- How much parallelism is available at a given sequence length and batch size?

Those questions belong to the next two chapters. Chapter 5 studies kernels, memory movement, and Transformer blocks. Chapter 6 uses FlashAttention as a concrete case where algorithm design and GPU memory hierarchy meet.

The point of this chapter is the foundation: a GPU program is a host-orchestrated, device-executed, massively threaded program whose performance is shaped as much by memory movement and execution layout as by arithmetic.

Owner: Principal Author  
Purpose: Chapter 4 ready draft after source extraction, brief, technical review, and red-team review  
Evidence grade: A for course lecture claims and official CUDA documentation cards; no benchmark numbers used  
Assumptions: Chapter 4 uses minimal CUDA snippets, not full compilable examples  
Open questions: Decide whether runnable CUDA examples belong in appendix or examples directory  
Handoff: Production can move to front-half Chapter 1-3 reviews or book-level consistency audit
