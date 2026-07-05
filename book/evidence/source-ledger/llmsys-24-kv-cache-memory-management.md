# Source Card: llmsys-24-kv-cache-memory-management

Source ID: llmsys-24-kv-cache-memory-management  
Title: Paged Attention & vLLM for Efficient LLM Inference Engine  
Author/issuer: Woosuk Kwon / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-24-vLLM_woosuk_kwon-b6a0750bb310949461ba5a635a1126eb.pdf`  
Pages / slides / sections: Slides 12-13  
Claim supported: Efficient KV-cache management is a central requirement for high-throughput LLM serving.  
Exact quote: "Efficient management of KV cache is crucial"  
Paraphrase: The lecture identifies KV cache, alongside model parameters, as a major GPU-memory consumer that constrains the number of concurrent requests an engine can serve.  
Evidence grade: A  
Technical sensitivity: memory | throughput  
Conditions:
  model: lecture example uses a 13B LLM
  hardware: lecture example uses A100 40GB
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: If using the slide's memory breakdown, carry the 13B/A100-40GB context and treat it as a lecture example.
