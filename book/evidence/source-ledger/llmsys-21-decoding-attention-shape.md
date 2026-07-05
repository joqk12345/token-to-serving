# Source Card: llmsys-21-decoding-attention-shape

Source ID: llmsys-21-decoding-attention-shape  
Title: Optimizing Attention for Modern Hardware  
Author/issuer: Tri Dao / CMU 11868/11968 LLM Systems guest lecture  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-21-FlashAttention_tridao2026.4-50476379a6127697ae7fbf974ad28348.pdf`  
Pages/slides: decoding inference section  
Claim supported: Decode attention differs from training/prefill attention because query length is short while context length can be long.  
Exact quote: "query length is short"; "context length is long"  
Paraphrase: In decoding, attention optimization must handle a small number of query tokens attending over a long KV history.  
Evidence grade: A  
Technical risk: Low conceptually; implementation details belong to serving chapters.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Use as a bridge to inference chapters.
