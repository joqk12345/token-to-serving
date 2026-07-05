# Source Card: llmsys-29-ttft-tpot-slos

Source ID: llmsys-29-ttft-tpot-slos  
Title: Disaggregating Prefill and Decode for Goodput-Optimized LLM Serving  
Author/issuer: Hao Zhang / CMU LLM Systems  
Date: 2025  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-29-disaggregating_prefill_decode_hao_zhang-c0e55139d20512a2348783423397cc7f.pdf`  
Pages / slides / sections: Motivation: Applications have Diverse SLO  
Claim supported: TTFT and TPOT represent different latency constraints for initial response and subsequent generated-token cadence.  
Exact quote: "Time to first token"  
Paraphrase: The lecture distinguishes first-token latency from per-output-token latency as separate SLO dimensions.  
Evidence grade: A  
Technical sensitivity: serving metrics  
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
Notes: Numeric P99 examples require workload/SLO context if used.
