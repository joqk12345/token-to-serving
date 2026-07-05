# Source Card: llmsys-19-direct-quantization-errors

Source ID: llmsys-19-direct-quantization-errors  
Title: LLM Quantization -- Basic methods  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-19-quantization-da7a2abad092c802b03672ce1cc7bee9.pdf`  
Pages / slides / sections: Slide 9  
Claim supported: Direct low-precision conversion can reduce accuracy through precision loss, clipping/range mismatch, and rounding error.  
Exact quote: "Quantization error ➔ rounding errors"  
Paraphrase: Quantization introduces numerical error through several mechanisms, so compression must be evaluated against model behavior.  
Evidence grade: A  
Technical sensitivity: numerical accuracy  
Conditions:
  model:
  hardware:
  batch size:
  sequence length:
  precision: FP32 to INT8/INT4 examples in lecture
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use as the misconception-correction source.
