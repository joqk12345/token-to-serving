# Source Card: hu-2021-lora

Source ID: hu-2021-lora  
Title: LoRA: Low-Rank Adaptation of Large Language Models  
Author/issuer: Edward J. Hu et al.  
Date: 2021  
Source type: paper  
File path / URL: `https://arxiv.org/abs/2106.09685`  
Pages / slides / sections: Abstract and method framing  
Claim supported: LoRA freezes pretrained model weights and injects trainable rank-decomposition matrices into Transformer layers.  
Exact quote: "freezes the pre-trained model weights"  
Paraphrase: LoRA reduces trainable parameter and optimizer-state requirements by learning low-rank adaptation matrices rather than updating the full base model.  
Evidence grade: A  
Technical sensitivity: fine-tuning | benchmark  
Conditions:
  model: Transformer language models
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Do not quote parameter-reduction or memory-reduction ratios without setup.
