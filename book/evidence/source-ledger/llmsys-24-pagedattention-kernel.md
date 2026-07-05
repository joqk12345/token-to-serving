# Source Card: llmsys-24-pagedattention-kernel

Source ID: llmsys-24-pagedattention-kernel  
Title: Paged Attention & vLLM for Efficient LLM Inference Engine  
Author/issuer: Woosuk Kwon / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-24-vLLM_woosuk_kwon-b6a0750bb310949461ba5a635a1126eb.pdf`  
Pages / slides / sections: Slides 20-21  
Claim supported: PagedAttention fetches non-contiguous KV blocks through the block table and is implemented as a custom GPU kernel to avoid materializing gathered keys and values.  
Exact quote: "avoid materializing gathered keys"  
Paraphrase: The memory indirection is handled inside the attention computation rather than by first copying the cache into a contiguous buffer.  
Evidence grade: A  
Technical sensitivity: GPU kernel | memory layout  
Conditions:
  model: Transformer attention with virtualized KV cache
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: The slide mentions a kernel-latency slowdown due to indirection; do not use that number without context.
