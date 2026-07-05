# Source Card: llmsys-24-pagedattention-definition

Source ID: llmsys-24-pagedattention-definition  
Title: Paged Attention & vLLM for Efficient LLM Inference Engine  
Author/issuer: Woosuk Kwon / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-24-vLLM_woosuk_kwon-b6a0750bb310949461ba5a635a1126eb.pdf`  
Pages / slides / sections: Slide 17  
Claim supported: PagedAttention applies application-level paging and virtualization to attention KV cache.  
Exact quote: "Application-level memory paging"  
Paraphrase: PagedAttention adapts virtual-memory ideas to the KV-cache allocator used by an LLM inference engine.  
Evidence grade: A  
Technical sensitivity: memory management | algorithm  
Conditions:
  model: Transformer inference with KV cache
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use with the original paper card for publication-level support.
