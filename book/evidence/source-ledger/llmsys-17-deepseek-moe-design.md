# Source Card: llmsys-17-deepseek-moe-design

Source ID: llmsys-17-deepseek-moe-design  
Title: System for MOE Models  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-17-MoE-3aa3125f9ccdd4bb7109ef077fbe9260.pdf`  
Pages / slides / sections: Slides 30, 32-34  
Claim supported: DeepSeek-style MoE uses fine-grained experts, shared experts, routed experts, top-k routing, and expert/device load-balancing mechanisms.  
Exact quote: "Fine-grained experts"  
Paraphrase: DeepSeek-style MoE modifies standard routed-expert designs with finer expert segmentation and shared/routed expert structure.  
Evidence grade: A  
Technical sensitivity: architecture  
Conditions:
  model: DeepSeek-style MoE
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use DeepSeekMoE paper for publication-level details; avoid DeepSeek-V3 numeric claims unless separately sourced.
