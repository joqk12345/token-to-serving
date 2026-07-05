# Source Card: dao-2022-benchmark-context

Source ID: dao-2022-benchmark-context  
Title: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness  
Author/issuer: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re  
Date: 2022  
Source type: paper  
File path: `downloads/llmsystem2026spring/source_pdfs/2205.14135.pdf`  
Pages/slides: Abstract, Appendix E  
Claim supported: The paper reports speedups and memory reductions under specified sequence lengths, head dimensions, hardware, and benchmark settings.  
Exact quote: "Setup We measure runtime and memory usage"  
Paraphrase: The benchmark results are conditional on specific configurations such as A100, sequence lengths, head dimensions, masking/dropout, and baseline implementations.  
Evidence grade: A  
Technical risk: Medium; do not generalize reported speedups beyond setup.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Use only if Chapter 6 includes carefully qualified numbers.
