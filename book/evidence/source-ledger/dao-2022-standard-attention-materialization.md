# Source Card: dao-2022-standard-attention-materialization

Source ID: dao-2022-standard-attention-materialization  
Title: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness  
Author/issuer: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re  
Date: 2022  
Source type: paper  
File path: `downloads/llmsystem2026spring/source_pdfs/2205.14135.pdf`  
Pages/slides: Section 2.2  
Claim supported: Standard attention materializes `S = QK^T` and `P = softmax(S)` as `N x N` matrices in HBM.  
Exact quote: "Standard attention implementations materialize the matrices S and P to HBM"  
Paraphrase: The standard implementation writes the score matrix and probability matrix to HBM, creating quadratic memory traffic and storage pressure.  
Evidence grade: A  
Technical risk: Low.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Use before introducing FlashAttention.
