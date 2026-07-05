---
status: ready
chapter: 6
slug: 06-flashattention-transformer-acceleration
title: FlashAttention and Attention Acceleration
primary_sources:
  - llmsys-21-FlashAttention_tridao2026.4-50476379a6127697ae7fbf974ad28348.pdf
  - 2205.14135.pdf
secondary_sources: []
reader_level: engineer or graduate student who has read Chapters 4-5
technical_depth: intermediate-to-advanced
---

# FlashAttention and Attention Acceleration

Chapter 5 introduced the recurring GPU performance pattern: a correct tensor program may still be slow if it moves too much data. Tiling, coalescing, fusion, reduction-aware kernels, mixed precision, and memory reuse are all ways to change the shape of memory traffic.

Attention is where those ideas become unavoidable.

The mathematical expression is compact:

```text
O = softmax(QK^T)V
```

The standard implementation is not compact. It creates large intermediate matrices, applies row-wise softmax, and then multiplies by `V`. FlashAttention is important because it does not merely optimize one kernel in isolation. It reorganizes the attention algorithm around the GPU memory hierarchy while still computing exact attention.

That is why FlashAttention is a systems case study, not only an attention trick.

## Standard Attention Materializes the Problem

For one attention head, let:

```text
Q: N x d
K: N x d
V: N x d
```

Here `N` is sequence length and `d` is head dimension.

Standard attention computes:

```text
S = QK^T
P = softmax(S)
O = PV
```

The score matrix `S` has shape `N x N`. The probability matrix `P` also has shape `N x N`. Standard implementations materialize these matrices in HBM. [CITE: dao-2022-standard-attention-materialization]

This is the key systems fact. The modeler sees attention weights. The GPU sees a large matrix written to and read from relatively slow memory.

As context grows, `N x N` grows quickly. A sequence twice as long creates four times as many score entries. The arithmetic grows, but the memory traffic and intermediate storage also grow. For long sequences, materializing attention can become the limiting factor before the rest of the model is the issue.

## IO-Awareness

FlashAttention starts from a different performance question:

```text
How many reads and writes occur between HBM and on-chip SRAM?
```

The original paper names this principle IO-awareness: account for traffic between memory levels, not only floating-point operations. [CITE: dao-2022-flashattention-io-awareness]

This is exactly the lesson from Chapter 5, applied to attention. HBM is large but slower. On-chip SRAM is much smaller but much faster. A good attention algorithm should avoid repeatedly writing and reading the `N x N` attention matrix through HBM if the computation can be organized around smaller on-chip blocks.

That statement is stronger than "use a faster kernel." It says the algorithm should be shaped by the memory hierarchy.

## Tiling Attention

FlashAttention splits the inputs into blocks. Blocks of `Q`, `K`, and `V` are loaded from HBM into SRAM. The kernel computes attention contributions for those blocks, updates the output, and proceeds to the next block. [CITE: dao-2022-flashattention-tiling]

The rough shape is:

```text
load Q block
for each K,V block:
  load K,V block
  compute local scores
  update softmax statistics
  update output block
write output block
```

The goal is not to change attention from dense to sparse. The goal is to avoid writing the full score matrix and full probability matrix to HBM.

This is the same tiling idea from matrix multiplication, but the softmax makes it harder. In matmul, partial sums can be accumulated directly. In attention, softmax normalizes across a whole row. If the row is processed block by block, the kernel must still produce the same answer as if it had seen the whole row at once.

## Online Softmax Makes Blocks Exact

The challenge is the denominator of softmax.

For one row of scores, softmax needs a normalization term over all columns. If the kernel sees only one block of columns at a time, a local softmax over that block is not enough. It would use the wrong denominator.

FlashAttention keeps extra row-wise statistics while processing blocks: a running maximum and a running normalization term. When a new block arrives, the algorithm updates those statistics and rescales the accumulated output so it remains consistent with the global row-wise softmax. [CITE: dao-2022-online-softmax]

For one row, let the old blocks have running maximum `m_old`, normalization sum `l_old`, and accumulated output `O_old`. Let the new block have scores `s_block` and values `V_block`.

The block computes:

```text
m_block = max(s_block)
p_block = exp(s_block - m_block)
l_block = sum(p_block)
O_block = p_block V_block
```

Then the merged maximum is:

```text
m_new = max(m_old, m_block)
```

The old and new contributions must be rescaled to the same denominator:

```text
l_new =
  exp(m_old - m_new) * l_old
  + exp(m_block - m_new) * l_block
```

The output update follows the same rescaling:

```text
O_new =
  (exp(m_old - m_new) * l_old * O_old
   + exp(m_block - m_new) * O_block)
  / l_new
```

Here `O_old` is the normalized accumulated output for earlier blocks, while `O_block` is the unnormalized weighted value sum from the new block.

The invariant is:

```text
after each block, the partial output is scaled as if all blocks seen so far
had been normalized together
```

That is what makes the tiled algorithm exact. It is not an approximation to attention, and it is not a sparse attention pattern. It computes the same attention result while changing the order and location of intermediate computation.

The formulas are compact, but the systems meaning is the important part: online softmax turns a global row-wise normalization into a blockwise computation with correct rescaling.

## Backward Uses Recomputation

Training needs gradients. A naive backward pass would like to reuse the attention probabilities from forward. Standard implementations may therefore store large `N x N` intermediates.

FlashAttention chooses a different tradeoff. It stores the output and softmax normalization statistics from the forward pass, then recomputes attention blocks during backward from `Q`, `K`, and `V` as needed. [CITE: dao-2022-backward-recomputation]

This adds some computation. But it avoids storing and rereading the full attention matrix through HBM.

That tradeoff should now feel familiar:

```text
more arithmetic
less HBM traffic
lower memory footprint
```

In many GPU workloads, that is a good trade. If the bottleneck is memory movement rather than arithmetic throughput, recomputation can make the program faster and smaller at the same time.

This is not generic checkpointing as a slogan. It is selective recomputation aligned with the attention algorithm and GPU memory hierarchy.

## IO Complexity Is the Argument

The FlashAttention paper does not only report speedups. It analyzes HBM accesses. The claim is that FlashAttention requires fewer HBM reads and writes than standard attention for the relevant SRAM sizes. [CITE: dao-2022-io-complexity]

For this book, the exact asymptotic expression matters less than the reasoning habit:

```text
count the scarce memory traffic, not only the FLOPs
```

Standard attention materializes `N x N` intermediates in HBM. FlashAttention keeps block computation on chip, stores smaller row-wise statistics, and writes the final output. The result is lower memory use and lower HBM traffic while preserving exactness.

This is the strongest case study so far for model-algorithm-system co-design. The model operation is attention. The algorithm is blockwise online softmax plus recomputation. The system target is the GPU memory hierarchy.

## Benchmarks Need Conditions

FlashAttention is often summarized as "faster attention with less memory." That is true as a direction, but publication-quality writing needs conditions.

The original paper reports benchmark results under specific settings: hardware, sequence lengths, head dimensions, batch sizes, masking, dropout, baselines, and whether the measurement covers forward, backward, or both. [CITE: dao-2022-benchmark-context]

This chapter therefore does not use a universal speedup number. The robust claim is narrower and stronger:

```text
FlashAttention reduces HBM traffic by reorganizing exact attention around SRAM-resident blocks.
```

Performance follows from that design when the workload is bottlenecked by the avoided memory traffic. The exact gain depends on shape, hardware, precision, masking, dropout, and implementation.

## Why This Belongs in a Systems Book

FlashAttention is not only a clever attention implementation.

It changes the boundary between algorithm and kernel. In a framework-level expression, attention looks like three operations:

```text
matmul -> softmax -> matmul
```

That decomposition materializes the wrong intermediate for the hardware. FlashAttention fuses the full attention computation into a memory-aware kernel structure. It keeps the mathematical operation but changes the schedule, storage, and recomputation plan.

This is why high-level tensor code can be too coarse for some LLM systems problems. The expression is correct, but the implied memory behavior is expensive. A better system sometimes requires changing the algorithmic schedule itself.

## Modern Hardware Keeps Moving

Later FlashAttention work continues the same pattern on newer GPUs. FlashAttention-2 improves work partitioning and parallelism. FlashAttention-3 targets Hopper-era asynchrony and low precision. FlashAttention-4 targets Blackwell-era asymmetric hardware scaling. [CITE: dao-2023-flashattention-2; shah-2024-flashattention-3; zadouri-2026-flashattention-4]

Those details are not the main thread of this chapter. They should stay as sidebar direction unless the chapter expands into a version-by-version treatment.

The broader lesson is recurring: hardware changes which algorithmic schedule is favorable. When tensor cores get faster, another unit may become the bottleneck. When memory movement becomes asynchronous, kernels can overlap work differently. When precision formats change, the numerical and layout choices change with them.

FlashAttention is therefore not a single frozen trick. It is an example of attention being redesigned around the active bottleneck.

## Decode Attention Is Different

Most of this chapter has described training or prefill-like attention, where `Q`, `K`, and `V` blocks can be large.

Decoding has a different shape. The query length may be only one or a few tokens, while the KV context can be very long. [CITE: llmsys-21-decoding-attention-shape]

That changes GPU occupancy and memory behavior. There may not be enough query-side work to fill the GPU unless the implementation splits or packs work differently. This issue will return in the serving chapters, where KV cache, batching, and decode scheduling become central.

For now, the point is simply that "attention optimization" is not one workload. Training, prefill, and decode expose different shapes to the same hardware.

## The Case Study

FlashAttention ties together the first six chapters.

From Part I, it uses the Transformer operation that makes context interaction possible. From Chapter 4, it depends on the GPU execution and memory model. From Chapter 5, it uses tiling, fusion, reductions, and recomputation. Its contribution is to combine those ideas into an exact attention algorithm whose memory behavior is better aligned with hardware.

The lesson is not "memorize FlashAttention." The lesson is the engineering move:

```text
find the active bottleneck
change the algorithmic schedule
preserve the mathematical contract
map the new schedule to the hardware
```

That pattern will repeat in later chapters on frameworks, distributed training, ZeRO, vLLM, KV cache, and serving.

Owner: Principal Author  
Purpose: Chapter 6 ready draft after source extraction, brief, technical review, formula check, and red-team review  
Evidence grade: A for FlashAttention v1 primary-paper claims, course lecture, later-generation primary cards, and formula-check memo; no benchmark numbers used  
Assumptions: Chapter 6 explains IO-aware exact attention and avoids unqualified benchmark claims  
Open questions: Decide whether final publication should keep the one-row online-softmax derivation or use paper notation  
Handoff: Production can move to front-half Chapter 1-3 reviews or book-level consistency audit
