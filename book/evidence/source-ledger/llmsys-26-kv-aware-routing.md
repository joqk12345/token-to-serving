# Source Card: llmsys-26-kv-aware-routing

Source ID: llmsys-26-kv-aware-routing  
Title: Inference at Scale: Opportunities and Challenges  
Author/issuer: Vikram Sharma Mailthody / CMU LLM Systems  
Date: 2025  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-26-dynamo-vikram_mailthody-0ddbeb69382d5c168b6d4636a82185d0.pdf`  
Pages / slides / sections: Scheduling / KV-aware routing  
Claim supported: KV-aware routing considers KV-cache hit rate and worker load when routing requests.  
Exact quote: "KV hit rate"  
Paraphrase: The lecture uses KV-aware routing to show that distributed serving schedulers should consider where reusable cache state already resides, not only raw worker availability.  
Evidence grade: A  
Technical sensitivity: routing | cache reuse | benchmark  
Conditions:
  model:
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: numbers require source context  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Do not use the slide's TTFT/TPOT/TPS speedup numbers without full Baseten/Qwen3 deployment conditions.
