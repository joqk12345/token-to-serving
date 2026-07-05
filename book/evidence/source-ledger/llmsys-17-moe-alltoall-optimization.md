# Source Card: llmsys-17-moe-alltoall-optimization

Source ID: llmsys-17-moe-alltoall-optimization  
Title: System for MOE Models  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-17-MoE-3aa3125f9ccdd4bb7109ef077fbe9260.pdf`  
Pages / slides / sections: Slide 25  
Claim supported: Expert parallelism creates all-to-all communication, and optimized systems may use hierarchical or parallelism-coordinated communication schedules.  
Exact quote: "Expert parallelism requires all-to-all communication"  
Paraphrase: The expert-parallel communication bottleneck is token/expert exchange across devices, making all-to-all scheduling a central MoE systems issue.  
Evidence grade: A  
Technical sensitivity: communication  
Conditions:
  model: expert-parallel MoE
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Do not use the slide's linear-latency language as a universal law without topology conditions.
