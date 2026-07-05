# Source Card: llmsys-19-zeroquant

Source ID: llmsys-19-zeroquant  
Title: LLM Quantization -- Basic methods  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-19-quantization-da7a2abad092c802b03672ce1cc7bee9.pdf`  
Pages / slides / sections: Slides 21-22  
Claim supported: ZeroQuant uses layer-by-layer knowledge distillation with the original model as teacher and the quantized model as student.  
Exact quote: "Use the original model as Teacher"  
Paraphrase: ZeroQuant calibrates quantized layers using teacher-student layer output matching and optimized transformer kernels.  
Evidence grade: A  
Technical sensitivity: algorithm | kernels  
Conditions:
  model: large-scale Transformer
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Do not quote runtime/model-size claims without original paper setup.
