# Source Card: llmsys-23-qlora-quantized-lora

Source ID: llmsys-23-qlora-quantized-lora  
Title: Parameter Efficient Fine-Tuning for LLM  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-23-peft-791e97c5520e2ea7491153b45a923ac7.pdf`  
Pages / slides / sections: Slide 6  
Claim supported: QLoRA combines parameter-efficient fine-tuning with quantization.  
Exact quote: "Quantization + Low-rank training"  
Paraphrase: QLoRA keeps the base model quantized while training low-rank adaptation parameters.  
Evidence grade: A  
Technical sensitivity: quantized training  
Conditions:
  model: QLoRA fine-tuning
  hardware:
  batch size:
  sequence length:
  precision: lecture mentions 4-bit weight example
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use QLoRA paper for NF4/double-quantization/paged-optimizer details.
