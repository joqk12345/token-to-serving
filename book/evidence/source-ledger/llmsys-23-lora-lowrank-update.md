# Source Card: llmsys-23-lora-lowrank-update

Source ID: llmsys-23-lora-lowrank-update  
Title: Parameter Efficient Fine-Tuning for LLM  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-23-peft-791e97c5520e2ea7491153b45a923ac7.pdf`  
Pages / slides / sections: Slides 12-14  
Claim supported: LoRA freezes pretrained weights and trains a low-rank update `A B` added to the base weight matrix.  
Exact quote: "Train a low-rank update"  
Paraphrase: LoRA constrains adaptation to a low-rank parameterization, reducing trainable parameter count.  
Evidence grade: A  
Technical sensitivity: fine-tuning algorithm  
Conditions:
  model: Transformer weight matrices
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Placement across attention/FFN/embedding matrices can vary.
