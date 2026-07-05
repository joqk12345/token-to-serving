# Source Card: llmsys-17-switch-top1-routing

Source ID: llmsys-17-switch-top1-routing  
Title: System for MOE Models  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-17-MoE-3aa3125f9ccdd4bb7109ef077fbe9260.pdf`  
Pages / slides / sections: Slides 6-7  
Claim supported: Switch-style Transformer MoE routes a token through a selected expert FFN using a gating network.  
Exact quote: "one token is only passed through one selected FFN"  
Paraphrase: Switch-style routing activates a small subset of experts per token rather than all expert FFNs.  
Evidence grade: A  
Technical sensitivity: algorithm  
Conditions:
  model: Switch Transformer-style MoE
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use Switch paper for paper-level details and training claims.
