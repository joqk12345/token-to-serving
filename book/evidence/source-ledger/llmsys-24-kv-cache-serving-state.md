# Source Card: llmsys-24-kv-cache-serving-state

Source ID: llmsys-24-kv-cache-serving-state  
Title: Paged Attention & vLLM for Efficient LLM Inference Engine  
Author/issuer: Woosuk Kwon / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-24-vLLM_woosuk_kwon-b6a0750bb310949461ba5a635a1126eb.pdf`  
Pages / slides / sections: Slides 10-11  
Claim supported: Attention KV cache stores per-token key/value state across Transformer layers during inference.  
Exact quote: "Attention KV cache"  
Paraphrase: During generation, a request accumulates attention keys and values that later decode steps read instead of recomputing all prior token states.  
Evidence grade: A  
Technical sensitivity: memory | inference  
Conditions:
  model: autoregressive Transformer inference
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Avoid the slide's approximate per-token memory number unless model and precision assumptions are supplied.
