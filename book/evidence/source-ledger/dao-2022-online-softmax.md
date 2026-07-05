# Source Card: dao-2022-online-softmax

Source ID: dao-2022-online-softmax  
Title: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness  
Author/issuer: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re  
Date: 2022  
Source type: paper  
File path: `downloads/llmsystem2026spring/source_pdfs/2205.14135.pdf`  
Pages/slides: Section 3.1  
Claim supported: FlashAttention maintains softmax statistics so blockwise computation remains exact.  
Exact quote: "keep track of some extra statistics"  
Paraphrase: The algorithm tracks row-wise maximum and normalization terms, allowing softmax over concatenated blocks to be updated incrementally and rescaled.  
Evidence grade: A  
Technical risk: Medium; final formula notation should be reviewed carefully.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Use for exactness of tiled attention.
