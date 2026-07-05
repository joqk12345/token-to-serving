# Source Card: llmsys-30-serving-framework-boundary

Source ID: llmsys-30-serving-framework-boundary  
Title: LLM Serving  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2024  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-30-serving-c4a70ab21cde01fb60068a256c6e163a.pdf`  
Pages / slides / sections: Slides 20-24  
Claim supported: LLM serving frameworks separate request serving concerns from optimized inference engines that execute batched model computation.  
Exact quote: "Triton groups multiple client requests into a batch"  
Paraphrase: The serving stack includes API/server scheduling layers and model execution engines, which may be combined but solve different problems.  
Evidence grade: A  
Technical sensitivity: serving architecture  
Conditions:
  model: LLM serving framework examples
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use as architectural example, not product recommendation.
