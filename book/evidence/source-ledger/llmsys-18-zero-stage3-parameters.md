# Source Card: llmsys-18-zero-stage3-parameters

Source ID: llmsys-18-zero-stage3-parameters  
Title: Memory Optimization in Distributed Training  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2025  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-18-zero-20eb6c8d8c1e7092e1b922abf03d8cdd.pdf`  
Pages / slides / sections: Slides 51-63  
Claim supported: ZeRO stage 3 partitions parameters and communicates parameter partitions during forward/backward computation.  
Exact quote: "Partition parameters to K parts"  
Paraphrase: Stage 3 reduces resident parameter memory by gathering/broadcasting parameter shards when computation needs them, then releasing unowned shards.  
Evidence grade: A  
Technical sensitivity: memory partitioning | communication  
Conditions:
  model: data-parallel training
  hardware: K GPUs in lecture examples
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Avoid implementation-specific claims about all-gather versus broadcast unless tied to source wording.
