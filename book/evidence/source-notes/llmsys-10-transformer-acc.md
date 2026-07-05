# Source Note: llmsys-10 Transformer Acceleration

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-10-transformer-acc-5ba466406bf7296f86cd244ad0405867.pdf`

## Scope

This source covers accelerated Transformer training and inference, mainly through the LightSeq and LightSeq2 examples. It focuses on kernel fusion, non-GEMM operator optimization, reduction rewriting, mixed precision, memory reuse, and decoding acceleration.

## Key Claims

- Transformer performance depends heavily on non-GEMM operators as well as matrix multiplication.
- Kernel fusion reduces kernel launch overhead and avoids extra intermediate memory reads/writes.
- Embedding, bias/dropout/residual, softmax, cross entropy, LayerNorm, and trainer updates can be optimized with custom kernels.
- Some reductions can be algebraically rewritten to reduce synchronization.
- Mixed precision reduces storage and transfer cost, but optimizer updates often require FP32 arithmetic/state.
- Training backward passes can reuse memory once intermediate tensors are no longer needed.
- Inference can share tensor memory across layers and avoid storing backward-pass intermediates.

## Chapter 5 Use

- Connect generic GPU acceleration ideas to Transformer blocks.
- Explain why GEMM libraries are necessary but not sufficient.
- Introduce kernel fusion as a memory-traffic reduction technique.
- Use LayerNorm and softmax as examples of reduction-heavy operators.
- Introduce mixed precision and memory reuse as system-level optimization themes.

## Do Not Use As

- A primary source for LightSeq benchmark claims without reading the LightSeq papers.
- A final source for exact speedup numbers.
- A replacement for Chapter 6's FlashAttention treatment.

## Candidate Source Cards

- `llmsys-10-kernel-fusion`
- `llmsys-10-fused-embedding`
- `llmsys-10-layernorm-reduction-rewrite`
- `llmsys-10-softmax-reduction`
- `llmsys-10-mixed-precision`
- `llmsys-10-memory-reuse`
- `llmsys-10-transformer-operator-stack`

Owner: Technical Researcher  
Purpose: Chapter 5 source extraction  
Evidence grade: A for course framing; LightSeq papers needed for publication-level speedup claims  
Open questions: Whether LightSeq should be a main case study or a supporting example  
Handoff: Book Architect for Chapter 5 brief
