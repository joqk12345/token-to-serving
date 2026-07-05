# Source Card: llmsys-20-gptq-blockwise-compensation

Source ID: llmsys-20-gptq-blockwise-compensation  
Title: LLM Quantization - GPTQ  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-20-quantization2-ba573d7e5d82e68027bbd3a92c3cd819.pdf`  
Pages / slides / sections: Slides 6-12  
Claim supported: GPTQ quantizes one column block at a time and updates not-yet-quantized weights to compensate for quantization error.  
Exact quote: "Updates all the not-yet-quantized weights"  
Paraphrase: GPTQ reduces accumulated quantization error by propagating rounding-error compensation into remaining full-precision weights before they are quantized.  
Evidence grade: A  
Technical sensitivity: algorithm  
Conditions:
  model: Transformer linear layer weight matrix
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Main draft can use qualitative mechanism instead of formulas.
