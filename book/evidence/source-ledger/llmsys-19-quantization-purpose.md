# Source Card: llmsys-19-quantization-purpose

Source ID: llmsys-19-quantization-purpose  
Title: LLM Quantization -- Basic methods  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-19-quantization-da7a2abad092c802b03672ce1cc7bee9.pdf`  
Pages / slides / sections: Slide 5  
Claim supported: Quantization uses low-bit precision to store parameters and layer outputs, reducing memory with possible accuracy loss.  
Exact quote: "Use low-bit precision to store parameters and layer outputs"  
Paraphrase: Quantization is a representation tradeoff: lower memory and possibly faster arithmetic, with accuracy risk.  
Evidence grade: A  
Technical sensitivity: memory | accuracy  
Conditions:
  model:
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Do not claim speedup without hardware/kernel conditions.
