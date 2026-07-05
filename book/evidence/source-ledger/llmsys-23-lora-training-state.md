# Source Card: llmsys-23-lora-training-state

Source ID: llmsys-23-lora-training-state  
Title: Parameter Efficient Fine-Tuning for LLM  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-23-peft-791e97c5520e2ea7491153b45a923ac7.pdf`  
Pages / slides / sections: Slides 16-18  
Claim supported: LoRA fixes the original pretrained weight matrix and stores/trains adapter weights, adapter gradients, adapter optimizer states, and activations.  
Exact quote: "No need to store original parameter states"  
Paraphrase: LoRA's memory savings come from not maintaining optimizer/gradient state for frozen base parameters.  
Evidence grade: A  
Technical sensitivity: memory accounting  
Conditions:
  model: LoRA/CIAT fine-tuning
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Avoid memory numbers without rank/model/precision/optimizer assumptions.
