# Source Card: llmsys-26-scale-serving-challenges

Source ID: llmsys-26-scale-serving-challenges  
Title: Inference at Scale: Opportunities and Challenges  
Author/issuer: Vikram Sharma Mailthody / CMU LLM Systems  
Date: 2025  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-26-dynamo-vikram_mailthody-0ddbeb69382d5c168b6d4636a82185d0.pdf`  
Pages / slides / sections: Challenges Serving AI Inference at Scale  
Claim supported: Large-scale inference must handle variable request lengths, multi-turn/agent workloads, heterogeneous hardware, fault tolerance, multiple models, TCO, goodput, power, and KV-cache hit rate.  
Exact quote: "Optimize for goodput"  
Paraphrase: The lecture frames inference at scale as a multi-objective systems problem rather than only a model-kernel problem.  
Evidence grade: A  
Technical sensitivity: serving architecture | operations  
Conditions:
  model: LLM serving at scale
  hardware: heterogeneous datacenter hardware
  batch size:
  sequence length: varying ISL/OSL
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Avoid datacenter-size examples unless their source context is carried.
