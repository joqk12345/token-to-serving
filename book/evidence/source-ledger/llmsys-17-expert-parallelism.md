# Source Card: llmsys-17-expert-parallelism

Source ID: llmsys-17-expert-parallelism  
Title: System for MOE Models  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-17-MoE-3aa3125f9ccdd4bb7109ef077fbe9260.pdf`  
Pages / slides / sections: Slides 15-16, 40  
Claim supported: Expert parallelism places experts on different worker devices, replicates non-expert components, and requires all-to-all communication.  
Exact quote: "expert parallelism: split experts and replicate non-expert"  
Paraphrase: MoE parallelism partitions experts across devices while other layers may be replicated, so token dispatch and expert outputs become communication work.  
Evidence grade: A  
Technical sensitivity: distributed training | communication  
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
Notes: Pair with GShard/DeepSpeed-MoE paper cards for named systems.
