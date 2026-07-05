# Source Card: llmsys-29-distserve-placement

Source ID: llmsys-29-distserve-placement  
Title: Disaggregating Prefill and Decode for Goodput-Optimized LLM Serving  
Author/issuer: Hao Zhang / CMU LLM Systems  
Date: 2025  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-29-disaggregating_prefill_decode_hao_zhang-c0e55139d20512a2348783423397cc7f.pdf`  
Pages / slides / sections: DistServe Design Overview; Core Problems  
Claim supported: DistServe placement chooses the parallelism strategy, number of prefill/decode instances, and physical cluster placement to optimize goodput.  
Exact quote: "number of each instance"  
Paraphrase: The lecture frames disaggregated serving placement as deciding how many prefill and decode instances to deploy, how to parallelize them, and where to place them on a cluster.  
Evidence grade: A  
Technical sensitivity: placement | distributed serving  
Conditions:
  model: DistServe-style prefill/decode disaggregation
  hardware: GPU cluster
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Add the DistServe paper before reproducing algorithms in detail.
