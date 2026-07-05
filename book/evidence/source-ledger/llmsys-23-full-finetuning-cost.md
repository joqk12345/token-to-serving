# Source Card: llmsys-23-full-finetuning-cost

Source ID: llmsys-23-full-finetuning-cost  
Title: Parameter Efficient Fine-Tuning for LLM  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-23-peft-791e97c5520e2ea7491153b45a923ac7.pdf`  
Pages / slides / sections: Slide 4  
Claim supported: Full-parameter fine-tuning updates all model parameters and requires large GPU memory for weights, gradients, optimizer states, and activations.  
Exact quote: "Update all model parameters"  
Paraphrase: Full fine-tuning carries training-state memory for the whole model, not only the base weights.  
Evidence grade: A  
Technical sensitivity: memory accounting  
Conditions:
  model: LLM fine-tuning
  hardware:
  batch size:
  sequence length:
  precision: half-precision example in lecture
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Avoid LLaMA memory numbers unless assumptions are carried.
