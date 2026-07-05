# Source Card: llmsys-29-disaggregation-challenges

Source ID: llmsys-29-disaggregation-challenges  
Title: Disaggregating Prefill and Decode for Goodput-Optimized LLM Serving  
Author/issuer: Hao Zhang / CMU LLM Systems  
Date: 2025  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-29-disaggregating_prefill_decode_hao_zhang-c0e55139d20512a2348783423397cc7f.pdf`  
Pages / slides / sections: Challenges of Disaggregation  
Claim supported: Disaggregation introduces KV-cache transmission overhead and makes per-GPU goodput optimization depend on workload, SLOs, parallelism, resource allocation, and network bandwidth.  
Exact quote: "Communication overhead"  
Paraphrase: Separating prefill and decode adds a data-movement and placement problem because KV cache must be transferred between phases.  
Evidence grade: A  
Technical sensitivity: communication | goodput  
Conditions:
  model: LLM prefill/decode disaggregation
  hardware: networked GPU serving cluster
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Avoid communication-latency claims without model/hardware assumptions.
