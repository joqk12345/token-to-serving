# Source Card: llmsys-28-pd-disaggregation-interference

Source ID: llmsys-28-pd-disaggregation-interference  
Title: LLM Serving on Heterogeneous Hardware  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-28-mooncake-kTransformer-1243bfbecbb0c3610bf95eac030acb2a.pdf`  
Pages / slides / sections: P&D Disaggregated Inference  
Claim supported: P/D disaggregation is presented as a way to avoid interference between prefill and decode in a mixed batch and decouple resources/parallelism.  
Exact quote: "Avoid interference"  
Paraphrase: The Mooncake lecture reinforces the DistServe framing that prefill and decode have different resource needs and should sometimes be scheduled separately.  
Evidence grade: A  
Technical sensitivity: scheduling | disaggregation  
Conditions:
  model: LLM serving with mixed prefill/decode workload
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use with `llmsys-29-disaggregation-opportunity`.
