# Source Card: llmsys-29-colocation-interference

Source ID: llmsys-29-colocation-interference  
Title: Disaggregating Prefill and Decode for Goodput-Optimized LLM Serving  
Author/issuer: Hao Zhang / CMU LLM Systems  
Date: 2025  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-29-disaggregating_prefill_decode_hao_zhang-c0e55139d20512a2348783423397cc7f.pdf`  
Pages / slides / sections: Continuous Batching Cause Interference; Colocation -> Coupled Parallelism  
Claim supported: Colocating prefill and decode can cause phase interference and couple the parallelism strategy for workloads with different TTFT/TPOT needs.  
Exact quote: "Continuous Batching Cause Interference"  
Paraphrase: The lecture argues that mixing prefill and decode on the same resources can waste time and force one parallelism configuration to serve both phases.  
Evidence grade: A  
Technical sensitivity: scheduling | parallelism  
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
Notes: Do not imply colocation is always bad; phrase as can cause under these workload/resource conditions.
