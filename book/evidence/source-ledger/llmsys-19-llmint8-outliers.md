# Source Card: llmsys-19-llmint8-outliers

Source ID: llmsys-19-llmint8-outliers  
Title: LLM Quantization -- Basic methods  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-19-quantization-da7a2abad092c802b03672ce1cc7bee9.pdf`  
Pages / slides / sections: Slides 23-24  
Claim supported: LLM.int8 uses 8-bit matrix multiplication for most values while keeping outlier features in higher precision.  
Exact quote: "Keep outliers in higher precision (FP16)"  
Paraphrase: LLM.int8 treats activation outliers separately to reduce accuracy degradation from blanket 8-bit quantization.  
Evidence grade: A  
Technical sensitivity: algorithm  
Conditions:
  model: Transformer matrix multiplications
  hardware:
  batch size:
  sequence length:
  precision: INT8 plus FP16 outlier path
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use Dettmers paper for publication-level details.
