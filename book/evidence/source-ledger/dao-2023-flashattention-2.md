# Source Card: dao-2023-flashattention-2

Source ID: dao-2023-flashattention-2  
Title: FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning  
Author/issuer: Tri Dao  
Date: 2023  
Source type: paper  
File path: https://arxiv.org/abs/2307.08691  
Pages/slides: Abstract  
Claim supported: FlashAttention-2 improves FlashAttention through better work partitioning and parallelism, reducing non-matmul FLOPs, parallelizing across sequence length, and distributing work between warps.  
Exact quote: "better work partitioning"  
Paraphrase: The paper identifies suboptimal thread-block and warp partitioning as a bottleneck in FlashAttention and proposes FA2 to increase occupancy and reduce shared-memory communication.  
Evidence grade: A  
Technical risk: Medium; benchmark numbers should be tied to hardware and precision conditions.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Use for Chapter 6 modern-hardware sidebar and future primary-paper review.
