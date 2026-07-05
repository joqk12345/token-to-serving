# Source Card: dettmers-2023-qlora

Source ID: dettmers-2023-qlora  
Title: QLoRA: Efficient Finetuning of Quantized LLMs  
Author/issuer: Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer  
Date: 2023  
Source type: paper  
File path / URL: `https://arxiv.org/abs/2305.14314`  
Pages / slides / sections: Abstract and method framing  
Claim supported: QLoRA backpropagates gradients through a frozen 4-bit quantized pretrained language model into LoRA adapters and introduces NF4, double quantization, and paged optimizers.  
Exact quote: "frozen, 4-bit quantized pretrained language model"  
Paraphrase: QLoRA combines quantized base-model storage with low-rank trainable adapters to reduce fine-tuning memory.  
Evidence grade: A  
Technical sensitivity: quantized training | benchmark  
Conditions:
  model: quantized pretrained language model plus LoRA adapters
  hardware:
  batch size:
  sequence length:
  precision: 4-bit base model in paper framing
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Do not quote single-GPU/model-size/performance claims without experiment setup.
