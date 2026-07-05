# Source Card: dao-2022-flashattention-io-awareness

Source ID: dao-2022-flashattention-io-awareness  
Title: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness  
Author/issuer: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re  
Date: 2022  
Source type: paper  
File path: `downloads/llmsystem2026spring/source_pdfs/2205.14135.pdf`  
Pages/slides: Abstract, Introduction, Section 2  
Claim supported: Attention algorithms should account for reads and writes between GPU memory levels, not only FLOP count.  
Exact quote: "a missing principle is making attention algorithms IO-aware"  
Paraphrase: The paper argues that wall-clock attention performance depends heavily on HBM/SRAM traffic and proposes IO-awareness as the organizing principle.  
Evidence grade: A  
Technical risk: Low.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Core Chapter 6 claim.
