# Source Card: llmsys-29-cb-vs-disaggregation

Source ID: llmsys-29-cb-vs-disaggregation  
Title: Disaggregating Prefill and Decode for Goodput-Optimized LLM Serving  
Author/issuer: Hao Zhang / CMU LLM Systems  
Date: 2025  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-29-disaggregating_prefill_decode_hao_zhang-c0e55139d20512a2348783423397cc7f.pdf`  
Pages / slides / sections: Continuous batching vs. disaggregation  
Claim supported: Continuous batching targets utilization/throughput, while disaggregation targets goodput under SLOs.  
Exact quote: "throughput s.t. SLOs"  
Paraphrase: The lecture positions disaggregation as complementary to continuous batching rather than a reversal of it.  
Evidence grade: A  
Technical sensitivity: scheduling | serving architecture  
Conditions:
  model: LLM serving
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use to bridge Chapter 12 and Chapter 14.
