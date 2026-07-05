# Source Card: llmsys-24-contiguous-preallocation-fragmentation

Source ID: llmsys-24-contiguous-preallocation-fragmentation  
Title: Paged Attention & vLLM for Efficient LLM Inference Engine  
Author/issuer: Woosuk Kwon / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-24-vLLM_woosuk_kwon-b6a0750bb310949461ba5a635a1126eb.pdf`  
Pages / slides / sections: Slide 15  
Claim supported: Preallocating contiguous KV-cache memory to a request's maximum length can cause internal and external fragmentation.  
Exact quote: "Pre-allocates contiguous memory"  
Paraphrase: If the engine reserves a contiguous region for each request up to its maximum possible length, unused output slots and mismatched request lengths waste memory.  
Evidence grade: A  
Technical sensitivity: memory allocation | serving  
Conditions:
  model: autoregressive Transformer serving
  hardware:
  batch size:
  sequence length: variable request/output lengths
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Pair with the PagedAttention cards when explaining the design response.
