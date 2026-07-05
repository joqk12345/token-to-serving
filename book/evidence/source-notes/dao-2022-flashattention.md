# Source Note: Dao et al. 2022 FlashAttention

Source PDF: `downloads/llmsystem2026spring/source_pdfs/2205.14135.pdf`

## Scope

This is the primary paper for FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. It argues that attention acceleration should account for IO between GPU HBM and on-chip SRAM, not only FLOPs.

## Key Claims

- Standard attention materializes the `N x N` score matrix and softmax probability matrix in HBM.
- FlashAttention computes exact attention with fewer HBM reads/writes by tiling `Q`, `K`, and `V`.
- Online softmax statistics allow blockwise computation without losing exactness.
- Backward pass can avoid storing the full attention matrix by storing normalization statistics and recomputing attention blocks.
- FlashAttention is IO-aware and analyzes HBM access complexity.
- The paper reports wall-clock speedups and memory reductions under specific benchmark settings.

## Chapter 6 Use

- Primary anchor for exact attention, tiling, IO-awareness, and recomputation.
- Use performance claims only with their benchmark context, or keep them qualitative.
- Use the algorithm as the case study for model-algorithm-system co-design.

## Do Not Use As

- A source for FA2/FA3/FA4 details.
- A source for current hardware performance beyond the paper's 2022 experimental context.
- A full replacement for kernel-level implementation documentation.

## Candidate Source Cards

- `dao-2022-flashattention-io-awareness`
- `dao-2022-standard-attention-materialization`
- `dao-2022-flashattention-tiling`
- `dao-2022-online-softmax`
- `dao-2022-backward-recomputation`
- `dao-2022-io-complexity`
- `dao-2022-benchmark-context`

Owner: Technical Researcher  
Purpose: Chapter 6 primary-paper extraction  
Evidence grade: A  
Open questions: Which benchmark numbers, if any, should appear in the main text  
Handoff: Book Architect for Chapter 6 brief
