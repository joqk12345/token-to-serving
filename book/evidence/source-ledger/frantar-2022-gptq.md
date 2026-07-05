# Source Card: frantar-2022-gptq

Source ID: frantar-2022-gptq  
Title: GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers  
Author/issuer: Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh  
Date: 2022  
Source type: paper  
File path / URL: `https://arxiv.org/abs/2210.17323`  
Pages / slides / sections: Abstract and method framing  
Claim supported: GPTQ is a one-shot weight quantization method for generative Transformers based on approximate second-order information.  
Exact quote: "approximate second-order information"  
Paraphrase: GPTQ quantizes large Transformer weights after training while using calibration-derived curvature information to compensate quantization error.  
Evidence grade: A  
Technical sensitivity: algorithm | benchmark  
Conditions:
  model: generative pretrained Transformers
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Do not quote bitwidth/runtime/speedup claims without full setup.
