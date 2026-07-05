# Source Card: llmsys-24-kv-block-definition

Source ID: llmsys-24-kv-block-definition  
Title: Paged Attention & vLLM for Efficient LLM Inference Engine  
Author/issuer: Woosuk Kwon / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-24-vLLM_woosuk_kwon-b6a0750bb310949461ba5a635a1126eb.pdf`  
Pages / slides / sections: Slide 18  
Claim supported: A KV block is a fixed-size memory block that stores KV-cache state for a contiguous span of tokens.  
Exact quote: "fixed-size block of memory"  
Paraphrase: PagedAttention divides a sequence's KV cache into uniform blocks instead of requiring one contiguous allocation for the entire request.  
Evidence grade: A  
Technical sensitivity: memory layout  
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
Notes: The lecture illustration uses block size 4 for explanation, not as a recommended value.
