# Source Card: dao-2022-io-complexity

Source ID: dao-2022-io-complexity  
Title: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness  
Author/issuer: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re  
Date: 2022  
Source type: paper  
File path: `downloads/llmsystem2026spring/source_pdfs/2205.14135.pdf`  
Pages/slides: Abstract, Section 3.2  
Claim supported: FlashAttention analyzes HBM accesses and shows reduced IO versus standard attention for typical SRAM sizes.  
Exact quote: "requires fewer HBM accesses than standard attention"  
Paraphrase: The paper gives an IO-complexity analysis that treats HBM accesses as the relevant scarce resource.  
Evidence grade: A  
Technical risk: Medium; exact asymptotic statement should be checked before final mathematical presentation.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Keep high level in draft unless formulas are reviewed.
