# Source Card: llmsys-22-request-scheduler

Source ID: llmsys-22-request-scheduler  
Title: Design of Efficient LLM Inference Server  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-22-llm-serving-scheduler-radixattention-dfa87a4515092525676277a85bc4425d.pdf`  
Pages / slides / sections: Slides 9-10, 16  
Claim supported: An LLM request scheduler receives inputs, streams outputs, checks stop conditions, reorders requests, prepares batches, and allocates memory for current and next batches.  
Exact quote: "Reorder requests and prepare a batch"  
Paraphrase: The scheduler is responsible for both request progression and memory/batch preparation.  
Evidence grade: A  
Technical sensitivity: scheduler  
Conditions:
  model: LLM serving server
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use as conceptual scheduler loop, not exact SGLang implementation spec.
