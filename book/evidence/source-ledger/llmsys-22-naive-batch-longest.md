# Source Card: llmsys-22-naive-batch-longest

Source ID: llmsys-22-naive-batch-longest  
Title: Design of Efficient LLM Inference Server  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-22-llm-serving-scheduler-radixattention-dfa87a4515092525676277a85bc4425d.pdf`  
Pages / slides / sections: Slide 13  
Claim supported: Naive request batching can wait for the longest sequence in a batch when generation lengths differ.  
Exact quote: "wait for the longest seq"  
Paraphrase: Variable output lengths make request-level batching inefficient for autoregressive generation.  
Evidence grade: A  
Technical sensitivity: batching | latency  
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
Notes: Pair with ORCA for paper support.
