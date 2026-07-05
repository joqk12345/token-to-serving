# Source Card: llmsys-30-lightllm-async-token-attention

Source ID: llmsys-30-lightllm-async-token-attention  
Title: LLM Serving  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2024  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-30-serving-c4a70ab21cde01fb60068a256c6e163a.pdf`  
Pages / slides / sections: Slides 36-43  
Claim supported: LightLLM performs tokenization, model inference, and detokenization asynchronously and includes token-wise KV cache memory management and routing.  
Exact quote: "tokenization, model inference, and detokenization asynchronously"  
Paraphrase: Serving frameworks may optimize not only model kernels but also tokenization/detokenization, routing, and per-token KV memory management.  
Evidence grade: A  
Technical sensitivity: serving framework  
Conditions:
  model: LightLLM example
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use as architecture example, not current product-status claim.
