# Source Card: llmsys-23-peft-definition

Source ID: llmsys-23-peft-definition  
Title: Parameter Efficient Fine-Tuning for LLM  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-23-peft-791e97c5520e2ea7491153b45a923ac7.pdf`  
Pages / slides / sections: Slide 5  
Claim supported: PEFT updates a small subset or low-rank set of parameters instead of all model parameters.  
Exact quote: "Only update a small subset (or low-rank) of parameters"  
Paraphrase: PEFT reduces trainable state by freezing most pretrained parameters and training limited adaptation parameters.  
Evidence grade: A  
Technical sensitivity: fine-tuning  
Conditions:
  model: LLM fine-tuning
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Quality equivalence requires task/model evidence.
