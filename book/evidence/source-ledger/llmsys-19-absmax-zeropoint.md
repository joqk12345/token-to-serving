# Source Card: llmsys-19-absmax-zeropoint

Source ID: llmsys-19-absmax-zeropoint  
Title: LLM Quantization -- Basic methods  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-19-quantization-da7a2abad092c802b03672ce1cc7bee9.pdf`  
Pages / slides / sections: Slides 10-11, 25  
Claim supported: Basic direct quantization examples include absmax scaling and zero-point scaling.  
Exact quote: "absmax: linearly scale according to max abs value"  
Paraphrase: Absmax and zero-point quantization both map real-valued tensors into integer ranges using scale information, with zero-point also using an offset.  
Evidence grade: A  
Technical sensitivity: quantization math  
Conditions:
  model:
  hardware:
  batch size:
  sequence length:
  precision: INT8 examples
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Keep equations simple; implementation details vary.
