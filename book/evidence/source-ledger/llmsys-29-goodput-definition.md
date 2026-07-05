# Source Card: llmsys-29-goodput-definition

Source ID: llmsys-29-goodput-definition  
Title: Disaggregating Prefill and Decode for Goodput-Optimized LLM Serving  
Author/issuer: Hao Zhang / CMU LLM Systems  
Date: 2025  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-29-disaggregating_prefill_decode_hao_zhang-c0e55139d20512a2348783423397cc7f.pdf`  
Pages / slides / sections: High Throughput != High Goodput  
Claim supported: Goodput counts completed requests within SLO criteria, so high throughput can still produce low goodput.  
Exact quote: "completed request within SLO"  
Paraphrase: The lecture defines goodput as the rate of requests that satisfy latency SLOs, distinguishing it from raw completed-request throughput.  
Evidence grade: A  
Technical sensitivity: serving metrics  
Conditions:
  model: LLM serving under SLO criteria
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Goodput examples should state SLO thresholds.
