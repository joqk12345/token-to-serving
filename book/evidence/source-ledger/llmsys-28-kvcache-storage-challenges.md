# Source Card: llmsys-28-kvcache-storage-challenges

Source ID: llmsys-28-kvcache-storage-challenges  
Title: LLM Serving on Heterogeneous Hardware  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-28-mooncake-kTransformer-1243bfbecbb0c3610bf95eac030acb2a.pdf`  
Pages / slides / sections: KVCache Cache introduces High Challenges to Storage System  
Claim supported: KV-cache caching creates storage-system challenges because both cache size and transfer bandwidth affect serving.  
Exact quote: "high transfer bandwidth"  
Paraphrase: The lecture frames distributed KV caching as a storage and bandwidth problem, not just a GPU memory allocation problem.  
Evidence grade: A  
Technical sensitivity: storage | bandwidth | KV cache  
Conditions:
  model: LLM serving with reusable KV cache
  hardware: distributed storage/memory hierarchy
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: numeric examples require conditions  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Do not use token-to-KV byte examples without model/layer/hidden/precision assumptions.
