# Source Note: llmsys-23 PEFT

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-23-peft-791e97c5520e2ea7491153b45a923ac7.pdf`

## Scope

This source covers parameter-efficient fine-tuning, full fine-tuning memory pressure, PEFT categories, LoRA/CIAT low-rank adaptation, LoRA inference and task switching, LoRA backward computation, LoRA memory state, and QLoRA as quantization plus low-rank training.

## Key Claims

- Full-parameter fine-tuning updates all model parameters and requires large GPU memory.
- PEFT updates a small subset or low-rank set of parameters.
- PEFT approaches include selective, reparameterization, and additive methods.
- LoRA freezes pretrained weights and trains a low-rank update to a weight matrix.
- LoRA/CIAT can be expressed as `W' = W0 + A B` with low rank `r << d`.
- During LoRA training, the original pretrained weight matrix is fixed and gradients are needed for adapter matrices.
- LoRA training stores original parameters, adapter weights, adapter gradients, adapter optimizer states, and activations rather than optimizer state for all frozen parameters.
- QLoRA combines quantization with low-rank training.

## Chapter 11 Use

- Use PEFT as the adaptation half of the chapter.
- Explain LoRA as changing which parameters are trainable, not the base model's forward architecture in full.
- Treat memory numbers in slides as examples only if all assumptions are carried; otherwise avoid them.
- Use QLoRA to connect quantization and PEFT, without reproducing benchmark claims.

## Do Not Use As

- A source for universal quality equivalence between LoRA and full fine-tuning.
- A benchmark source without model, rank, optimizer, precision, task, and hardware context.
- A full current survey of PEFT methods.

## Candidate Source Cards

- `llmsys-23-full-finetuning-cost`
- `llmsys-23-peft-definition`
- `llmsys-23-peft-categories`
- `llmsys-23-lora-lowrank-update`
- `llmsys-23-lora-training-state`
- `llmsys-23-qlora-quantized-lora`

Owner: Technical Researcher  
Purpose: Chapter 11 PEFT source extraction  
Evidence grade: A for course framing; LoRA/QLoRA papers needed for publication-level claims  
Assumptions: Chapter 11 covers LoRA/QLoRA mechanisms, not a complete PEFT survey  
Open questions: Whether adapter/prompt tuning should be sidebar-only  
Handoff: Book Architect for Chapter 11 brief
