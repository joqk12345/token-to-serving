# Source Card: zadouri-2026-flashattention-4

Source ID: zadouri-2026-flashattention-4  
Title: FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling  
Author/issuer: Ted Zadouri, Markus Hoehnerbach, Jay Shah, Timmy Liu, Vijay Thakkar, Tri Dao  
Date: 2026  
Source type: paper  
File path: https://arxiv.org/abs/2603.05451  
Pages/slides: Abstract  
Claim supported: FlashAttention-4 targets Blackwell-era asymmetric hardware scaling through algorithm and kernel pipelining co-design.  
Exact quote: "asymmetric hardware scaling"  
Paraphrase: The paper argues that Blackwell GPUs shift attention bottlenecks because tensor core throughput scales faster than shared memory bandwidth and exponential units, motivating redesigned pipelines and softmax-related changes.  
Evidence grade: A  
Technical risk: High; this is recent hardware-specific work, so avoid broad claims without benchmark context.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Use for Chapter 6 sidebar and later update pass; do not generalize benchmark results.
