# Source Card: llmsys-28-mooncake-kvcache-centric

Source ID: llmsys-28-mooncake-kvcache-centric  
Title: LLM Serving on Heterogeneous Hardware  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-28-mooncake-kTransformer-1243bfbecbb0c3610bf95eac030acb2a.pdf`  
Pages / slides / sections: Mooncake: A KVCache-centric Disaggregated Architecture  
Claim supported: Mooncake is presented as a KV-cache-centric disaggregated architecture with prefill scheduling, KV-cache pool, KV-cache balancing, and decoding scheduling.  
Exact quote: "KVCache-centric"  
Paraphrase: The lecture presents Mooncake as organizing serving around distributed KV-cache management and separate prefill/decode components.  
Evidence grade: A  
Technical sensitivity: distributed serving | KV cache  
Conditions:
  model: LLM serving architecture
  hardware: distributed GPU/CPU/storage serving setting
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use as architecture example; avoid evaluation numbers without setup.
