# Source Card: dao-2022-backward-recomputation

Source ID: dao-2022-backward-recomputation  
Title: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness  
Author/issuer: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re  
Date: 2022  
Source type: paper  
File path: `downloads/llmsystem2026spring/source_pdfs/2205.14135.pdf`  
Pages/slides: Section 3.1, backward discussion  
Claim supported: FlashAttention stores output and softmax normalization statistics, then recomputes attention blocks during backward instead of storing the full attention matrix.  
Exact quote: "not store O(N^2) intermediate values"  
Paraphrase: The backward pass trades additional computation for reduced HBM traffic and lower memory footprint.  
Evidence grade: A  
Technical risk: Low conceptually; exact backward equations need technical review.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Use to connect Chapter 5 memory reuse with attention-specific recomputation.
