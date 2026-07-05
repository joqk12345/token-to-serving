# Source Card: llmsys-17-moe-load-balancing

Source ID: llmsys-17-moe-load-balancing  
Title: System for MOE Models  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-17-MoE-3aa3125f9ccdd4bb7109ef077fbe9260.pdf`  
Pages / slides / sections: Slides 19, 33  
Claim supported: MoE training uses load-balancing losses to avoid routing collapse and balance computation across experts or devices.  
Exact quote: "avoid routing collapse to experts"  
Paraphrase: Router decisions affect both model learning and system load; auxiliary balancing terms can encourage more even expert/device utilization.  
Evidence grade: A  
Technical sensitivity: training objective | scheduling  
Conditions:
  model: MoE Transformer
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Avoid reproducing formulas unless notation is checked against Switch/DeepSeek papers.
