# Source Card: llmsys-22-continuous-batching

Source ID: llmsys-22-continuous-batching  
Title: Design of Efficient LLM Inference Server  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-22-llm-serving-scheduler-radixattention-dfa87a4515092525676277a85bc4425d.pdf`  
Pages / slides / sections: Slide 14  
Claim supported: Continuous batching schedules at iteration/token-generation granularity and can add new requests when existing requests finish.  
Exact quote: "at each iteration"  
Paraphrase: Continuous batching updates batch membership between decode iterations rather than waiting for a whole request batch to finish.  
Evidence grade: A  
Technical sensitivity: scheduling  
Conditions:
  model: autoregressive LLM serving
  hardware:
  batch size:
  sequence length: variable generation lengths
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use ORCA paper card for primary support.
