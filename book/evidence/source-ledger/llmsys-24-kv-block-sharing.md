# Source Card: llmsys-24-kv-block-sharing

Source ID: llmsys-24-kv-block-sharing  
Title: Paged Attention & vLLM for Efficient LLM Inference Engine  
Author/issuer: Woosuk Kwon / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-24-vLLM_woosuk_kwon-b6a0750bb310949461ba5a635a1126eb.pdf`  
Pages / slides / sections: Slides 28-33  
Claim supported: KV blocks can be shared across related sequences, such as parallel samples with the same prompt prefix, reducing duplicate KV-cache storage.  
Exact quote: "Shared btw. sequences"  
Paraphrase: Block-level indirection lets multiple logical sequences point to the same physical prefix blocks until their continuations diverge.  
Evidence grade: A  
Technical sensitivity: memory sharing | decoding  
Conditions:
  model: Transformer inference with shared prompt prefixes or parallel sampling
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Explain copy-on-write only if backed by the original paper card or another explicit source.
