# Source Card: llmsys-24-block-table-virtualization

Source ID: llmsys-24-block-table-virtualization  
Title: Paged Attention & vLLM for Efficient LLM Inference Engine  
Author/issuer: Woosuk Kwon / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-24-vLLM_woosuk_kwon-b6a0750bb310949461ba5a635a1126eb.pdf`  
Pages / slides / sections: Slide 19  
Claim supported: PagedAttention represents a request's logical KV blocks through a block table that maps them to physical KV blocks.  
Exact quote: "Block table"  
Paraphrase: Logical sequence order is decoupled from physical GPU-memory placement by an indirection table, so adjacent logical blocks do not need adjacent physical addresses.  
Evidence grade: A  
Technical sensitivity: memory layout | attention kernel  
Conditions:
  model: Transformer inference with PagedAttention-style KV cache
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use when explaining why the attention kernel must understand the block table.
