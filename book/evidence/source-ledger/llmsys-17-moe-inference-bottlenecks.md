# Source Card: llmsys-17-moe-inference-bottlenecks

Source ID: llmsys-17-moe-inference-bottlenecks  
Title: System for MOE Models  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-17-MoE-3aa3125f9ccdd4bb7109ef077fbe9260.pdf`  
Pages / slides / sections: Slides 21-22  
Claim supported: MoE inference performance depends on model size, number of activated experts, memory bandwidth, token grouping, communication scheduling, and MoE kernels.  
Exact quote: "MoE inference performance depends on"  
Paraphrase: Sparse activation does not remove systems bottlenecks; inference must manage memory bandwidth, token grouping, communication, and kernels.  
Evidence grade: A  
Technical sensitivity: inference | memory bandwidth  
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
Notes: Chapter 10 can preview inference issues but should leave serving depth to Part V.
