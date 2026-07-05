# Source Card: llmsys-19-layerwise-objective

Source ID: llmsys-19-layerwise-objective  
Title: LLM Quantization -- Basic methods  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-19-quantization-da7a2abad092c802b03672ce1cc7bee9.pdf`  
Pages / slides / sections: Slide 17  
Claim supported: Layer-wise quantization can minimize the difference between full-precision and quantized linear-layer outputs.  
Exact quote: "minimizes the linear layer’s output"  
Paraphrase: Layer-wise PTQ uses calibration inputs to keep quantized layer outputs close to full-precision outputs.  
Evidence grade: A  
Technical sensitivity: objective  
Conditions:
  model: Transformer linear projection layers
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Accumulated error across layers remains a risk.
