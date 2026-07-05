# Source Card: llmsys-17-shared-routed-experts

Source ID: llmsys-17-shared-routed-experts  
Title: System for MOE Models  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-17-MoE-3aa3125f9ccdd4bb7109ef077fbe9260.pdf`  
Pages / slides / sections: Slide 8  
Claim supported: Shared-routed expert designs combine an always-used shared expert with routed experts selected by a router.  
Exact quote: "Shared expert: calculating common knowledge"  
Paraphrase: Some MoE designs reserve shared capacity for common computation while routing token-specific computation to selected experts.  
Evidence grade: A  
Technical sensitivity: architecture  
Conditions:
  model: DeepSpeed-MoE / DeepSeek-style shared-routed experts
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use original papers for specific architecture and quality claims.
