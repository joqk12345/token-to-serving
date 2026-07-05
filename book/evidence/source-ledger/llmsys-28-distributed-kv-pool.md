# Source Card: llmsys-28-distributed-kv-pool

Source ID: llmsys-28-distributed-kv-pool  
Title: LLM Serving on Heterogeneous Hardware  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-28-mooncake-kTransformer-1243bfbecbb0c3610bf95eac030acb2a.pdf`  
Pages / slides / sections: KVCache Cache: Local or Global; Mooncake Store  
Claim supported: Mooncake uses distributed multi-layer KV-cache storage/pools to make cache capacity exceed one machine.  
Exact quote: "Distributed KVCache Pool"  
Paraphrase: The lecture presents global KV-cache storage as a way to increase cache hit opportunities and share cache across serving components.  
Evidence grade: A  
Technical sensitivity: distributed cache | storage hierarchy  
Conditions:
  model: LLM serving with global KV-cache reuse
  hardware: CPU/DRAM/SSD and GPU/VRAM tiers
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Avoid PB-level claims without workload/system sizing context.
