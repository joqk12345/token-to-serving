# Source Card: llmsys-26-inference-vs-training

Source ID: llmsys-26-inference-vs-training  
Title: Inference at Scale: Opportunities and Challenges  
Author/issuer: Vikram Sharma Mailthody / CMU LLM Systems  
Date: 2025  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-26-dynamo-vikram_mailthody-0ddbeb69382d5c168b6d4636a82185d0.pdf`  
Pages / slides / sections: AI Inference section  
Claim supported: LLM inference differs from training because serving consists of many smaller online jobs with rapid scale-up and scale-down requirements.  
Exact quote: "Several small jobs"  
Paraphrase: The lecture contrasts training's coordinated long-running job structure with inference's online workload of many smaller requests.  
Evidence grade: A  
Technical sensitivity: serving architecture  
Conditions:
  model: LLM inference generally
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use qualitatively to motivate distributed serving.
