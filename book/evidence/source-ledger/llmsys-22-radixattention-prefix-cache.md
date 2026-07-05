# Source Card: llmsys-22-radixattention-prefix-cache

Source ID: llmsys-22-radixattention-prefix-cache  
Title: Design of Efficient LLM Inference Server  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-22-llm-serving-scheduler-radixattention-dfa87a4515092525676277a85bc4425d.pdf`  
Pages / slides / sections: Slides 21-32  
Claim supported: RadixAttention stores KV memory pointers in a radix tree keyed by prompt prefixes so shared prefixes can reuse KV cache entries.  
Exact quote: "KV mem pointers are stored in a radix tree"  
Paraphrase: Prefix-aware KV reuse stores prompt-prefix structure on CPU while nodes point to GPU KV memory.  
Evidence grade: A  
Technical sensitivity: KV cache | prefix reuse  
Conditions:
  model: LLM serving with shared prompt prefixes
  hardware: GPU KV memory plus CPU-managed radix tree in lecture framing
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Keep details light in Chapter 12; Chapter 13 handles KV cache depth.
