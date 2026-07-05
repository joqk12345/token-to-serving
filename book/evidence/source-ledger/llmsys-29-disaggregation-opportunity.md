# Source Card: llmsys-29-disaggregation-opportunity

Source ID: llmsys-29-disaggregation-opportunity  
Title: Disaggregating Prefill and Decode for Goodput-Optimized LLM Serving  
Author/issuer: Hao Zhang / CMU LLM Systems  
Date: 2025  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-29-disaggregating_prefill_decode_hao_zhang-c0e55139d20512a2348783423397cc7f.pdf`  
Pages / slides / sections: Opportunity: Disaggregating Prefill and Decoding  
Claim supported: Separating prefill and decode can divide SLO optimization so prefill instances optimize TTFT and decode instances optimize TPOT.  
Exact quote: "Prefill instance optimizes for TTFT"  
Paraphrase: Disaggregation treats prefill and decode as separately provisioned services, allowing different resource and parallelism choices for each phase.  
Evidence grade: A  
Technical sensitivity: distributed serving | SLOs  
Conditions:
  model: LLM serving with prefill/decode phases
  hardware: separate prefill/decode resources
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Do not claim universal win; pair with challenges card.
