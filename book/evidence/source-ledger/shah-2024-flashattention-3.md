# Source Card: shah-2024-flashattention-3

Source ID: shah-2024-flashattention-3  
Title: FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision  
Author/issuer: Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao  
Date: 2024  
Source type: paper  
File path: https://arxiv.org/abs/2407.08608  
Pages/slides: Abstract  
Claim supported: FlashAttention-3 targets Hopper GPUs using asynchrony, warp specialization, interleaving matmul and softmax, and FP8 low-precision techniques.  
Exact quote: "asynchrony and Low-precision"  
Paraphrase: The paper adapts FlashAttention to Hopper hardware by overlapping computation and data movement and by using FP8-related techniques to improve throughput while managing numerical error.  
Evidence grade: A  
Technical risk: Medium; hardware-specific claims should stay Hopper-qualified.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Use for Chapter 6 modern-hardware sidebar only unless the chapter expands into FA3 details.
