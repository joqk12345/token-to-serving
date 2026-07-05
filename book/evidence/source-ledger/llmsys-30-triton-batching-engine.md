# Source Card: llmsys-30-triton-batching-engine

Source ID: llmsys-30-triton-batching-engine  
Title: LLM Serving  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2024  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-30-serving-c4a70ab21cde01fb60068a256c6e163a.pdf`  
Pages / slides / sections: Slides 21-24  
Claim supported: NVIDIA Triton is an inference serving layer that can group client requests into batches while an inference engine such as TensorRT-LLM conducts batched model execution.  
Exact quote: "TensorRT-LLM conducts the inference procedure in the batched manner"  
Paraphrase: Batching spans serving-layer request grouping and backend model execution.  
Evidence grade: A  
Technical sensitivity: serving framework  
Conditions:
  model: LLM serving with Triton plus inference engine
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Official docs required for current feature claims.
