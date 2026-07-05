# Source Card: llmsys-22-selective-batching

Source ID: llmsys-22-selective-batching  
Title: Design of Efficient LLM Inference Server  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-22-llm-serving-scheduler-radixattention-dfa87a4515092525676277a85bc4425d.pdf`  
Pages / slides / sections: Slide 15  
Claim supported: Selective batching batches non-attention operations while attention is handled separately by an attention engine.  
Exact quote: "Selectively batch all non-attention operations"  
Paraphrase: Serving systems may batch some Transformer operations while using specialized attention handling for variable sequence lengths.  
Evidence grade: A  
Technical sensitivity: batching | attention  
Conditions:
  model: Transformer generative inference
  hardware:
  batch size:
  sequence length: variable sequence lengths
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Chapter 13 can revisit attention/KV memory details.
