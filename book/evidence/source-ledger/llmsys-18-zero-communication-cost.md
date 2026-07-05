# Source Card: llmsys-18-zero-communication-cost

Source ID: llmsys-18-zero-communication-cost  
Title: Memory Optimization in Distributed Training  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2025  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-18-zero-20eb6c8d8c1e7092e1b922abf03d8cdd.pdf`  
Pages / slides / sections: Slide 65  
Claim supported: ZeRO stage 3 introduces additional parameter-transfer communication relative to earlier ZeRO stages.  
Exact quote: "at the cost of additional parameter transfer"  
Paraphrase: Parameter partitioning reduces resident memory but moves parameter availability into the communication schedule.  
Evidence grade: A  
Technical sensitivity: communication cost  
Conditions:
  model: data-parallel training with ZeRO stages
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Avoid fixed communication multipliers unless topology and accounting assumptions are carried.
