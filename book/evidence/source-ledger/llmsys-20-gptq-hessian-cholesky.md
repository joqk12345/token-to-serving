# Source Card: llmsys-20-gptq-hessian-cholesky

Source ID: llmsys-20-gptq-hessian-cholesky  
Title: LLM Quantization - GPTQ  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-20-quantization2-ba573d7e5d82e68027bbd3a92c3cd819.pdf`  
Pages / slides / sections: Slides 7-8, 21  
Claim supported: GPTQ precomputes information from the inverse Hessian of layer inputs using Cholesky decomposition.  
Exact quote: "Pre-compute Cholesky decomposition"  
Paraphrase: GPTQ uses second-order information derived from calibration inputs to guide quantization-error compensation efficiently.  
Evidence grade: A  
Technical sensitivity: second-order method  
Conditions:
  model: linear layer with calibration input X
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Avoid deep derivation unless notation is reviewed.
