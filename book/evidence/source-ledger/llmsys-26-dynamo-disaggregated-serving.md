# Source Card: llmsys-26-dynamo-disaggregated-serving

Source ID: llmsys-26-dynamo-disaggregated-serving  
Title: Inference at Scale: Opportunities and Challenges  
Author/issuer: Vikram Sharma Mailthody / CMU LLM Systems  
Date: 2025  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-26-dynamo-vikram_mailthody-0ddbeb69382d5c168b6d4636a82185d0.pdf`  
Pages / slides / sections: Disaggregated serving  
Claim supported: Disaggregated serving scales prefill and decode independently on separate GPUs and can apply suitable parallelism to each phase.  
Exact quote: "Scale prefill & decode independently"  
Paraphrase: The lecture presents prefill/decode disaggregation as separating compute-bound prefill from memory-bound decode so each can be provisioned differently.  
Evidence grade: A  
Technical sensitivity: distributed serving | scheduling  
Conditions:
  model: LLM serving with separate prefill/decode phases
  hardware: separate GPU pools
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Pair with DistServe/PD cards for mechanism and goodput framing.
