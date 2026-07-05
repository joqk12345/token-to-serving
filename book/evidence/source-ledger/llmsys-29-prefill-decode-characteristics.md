# Source Card: llmsys-29-prefill-decode-characteristics

Source ID: llmsys-29-prefill-decode-characteristics  
Title: Disaggregating Prefill and Decode for Goodput-Optimized LLM Serving  
Author/issuer: Hao Zhang / CMU LLM Systems  
Date: 2025  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-29-disaggregating_prefill_decode_hao_zhang-c0e55139d20512a2348783423397cc7f.pdf`  
Pages / slides / sections: Prefill and Decode have Distinct Characteristics  
Claim supported: Prefill is compute-bound, while decode is memory-bound and needs many batched requests to saturate compute.  
Exact quote: "Prefill Compute-bound"  
Paraphrase: The lecture motivates disaggregation by showing that prefill and decode stress different hardware resources and batching regimes.  
Evidence grade: A  
Technical sensitivity: compute | memory | serving  
Conditions:
  model: LLM inference with prefill/decode phases
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Treat as qualitative unless adding roofline/model-specific evidence.
