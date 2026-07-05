# Source Card: dao-2022-flashattention-tiling

Source ID: dao-2022-flashattention-tiling  
Title: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness  
Author/issuer: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re  
Date: 2022  
Source type: paper  
File path: `downloads/llmsystem2026spring/source_pdfs/2205.14135.pdf`  
Pages/slides: Section 3.1  
Claim supported: FlashAttention splits `Q`, `K`, and `V` into blocks, loads blocks into SRAM, and computes attention outputs blockwise.  
Exact quote: "split the inputs Q, K, V into blocks"  
Paraphrase: The algorithm uses tiling to move blocks from HBM into SRAM and compute attention without materializing the full attention matrix in HBM.  
Evidence grade: A  
Technical risk: Low for conceptual explanation; code-level details require implementation review.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Central algorithm card.
