---
status: brief
chapter: 11
slug: 11-quantization-and-peft
title: Quantization and Parameter-Efficient Adaptation
primary_sources:
  - llmsys-19-quantization-da7a2abad092c802b03672ce1cc7bee9.pdf
  - llmsys-20-quantization2-ba573d7e5d82e68027bbd3a92c3cd819.pdf
  - llmsys-23-peft-791e97c5520e2ea7491153b45a923ac7.pdf
papers:
  - https://arxiv.org/abs/2208.07339
  - https://arxiv.org/abs/2210.17323
  - https://arxiv.org/abs/2106.09685
  - https://arxiv.org/abs/2305.14314
secondary_sources: []
reader_level: engineer or graduate student who has read Chapters 1-10
technical_depth: intermediate
---

# Quantization and Parameter-Efficient Adaptation

## Chapter Thesis

Compression and adaptation change the model-system contract. Quantization changes how tensors are represented and moved. PEFT changes which parameters are trainable and which optimizer state must exist. Both reduce memory pressure, but both can move cost into accuracy risk, calibration, kernels, adapter placement, and deployment complexity.

## Reader Problem

The reader has seen how training systems scale the original model through data parallelism, model parallelism, ZeRO, and MoE. The next problem is different: what if the base model is already trained, but serving it or adapting it is too expensive? The reader needs to understand why lower precision and low-rank adaptation are not just smaller files, but system design choices with numerical, memory, and runtime consequences.

## System Bottleneck

Primary bottlenecks: parameter memory, activation precision, quantization error, outlier features, calibration cost, kernel support, memory bandwidth, full fine-tuning optimizer state, trainable parameter count, adapter placement, and multi-adapter deployment.

Secondary bottlenecks: bitwidth, scale/zero-point granularity, per-tensor versus per-channel choices, calibration data, Hessian/curvature approximations, LoRA rank, target modules, optimizer choice, task quality, and hardware support for low-bit arithmetic.

## Source Map

| Claim                                                                                                              | Source card                             | Evidence grade | Notes                                                     |
| ------------------------------------------------------------------------------------------------------------------ | --------------------------------------- | -------------- | --------------------------------------------------------- |
| Quantization uses low-bit precision for parameters and layer outputs, reducing memory with possible accuracy loss. | `llmsys-19-quantization-purpose`        | A              | Opening definition.                                       |
| Direct quantization can lose information through precision loss, clipping/range mismatch, and rounding error.      | `llmsys-19-direct-quantization-errors`  | A              | Main misconception correction.                            |
| Absmax and zero-point quantization are basic scale-based mechanisms.                                               | `llmsys-19-absmax-zeropoint`            | A              | Use simple equations only.                                |
| Quantization methods include training-time and post-training approaches.                                           | `llmsys-19-quantization-approaches`     | A              | Taxonomy, not exhaustive survey.                          |
| Layer-wise quantization can minimize full-precision versus quantized layer output difference.                      | `llmsys-19-layerwise-objective`         | A              | Bridge to GPTQ.                                           |
| ZeroQuant uses layer-by-layer knowledge distillation.                                                              | `llmsys-19-zeroquant`                   | A              | Example only; avoid runtime claims.                       |
| LLM.int8 keeps outlier features in higher precision while quantizing most work to 8-bit.                           | `llmsys-19-llmint8-outliers`            | A              | Pair with paper card.                                     |
| LLM.int8 uses vector-wise quantization and mixed-precision outlier handling.                                       | `dettmers-2022-llmint8`                 | A              | Avoid performance numbers.                                |
| GPTQ targets PTQ for large generative Transformers using layer-wise weight quantization.                           | `llmsys-20-gptq-goal`                   | A              | Avoid unconditional superiority claims.                   |
| GPTQ quantizes column blocks and compensates not-yet-quantized weights.                                            | `llmsys-20-gptq-blockwise-compensation` | A              | Core mechanism.                                           |
| GPTQ uses inverse-Hessian/Cholesky precomputation from layer inputs.                                               | `llmsys-20-gptq-hessian-cholesky`       | A              | Keep qualitative unless notation reviewed.                |
| GPTQ batches/lazily applies updates for practical efficiency.                                                      | `llmsys-20-gptq-lazy-updates`           | A              | Systems angle.                                            |
| GPTQ paper uses approximate second-order information for one-shot weight quantization.                             | `frantar-2022-gptq`                     | A              | No speed/bitwidth claims without setup.                   |
| Full fine-tuning updates all model parameters and needs weights, gradients, optimizer states, and activations.     | `llmsys-23-full-finetuning-cost`        | A              | Connect to Ch. 10 memory ledger.                          |
| PEFT updates a small subset or low-rank set of parameters.                                                         | `llmsys-23-peft-definition`             | A              | PEFT definition.                                          |
| PEFT methods include selective, reparameterization, and additive methods.                                          | `llmsys-23-peft-categories`             | A              | Keep LoRA in broader context.                             |
| LoRA freezes pretrained weights and trains a low-rank update.                                                      | `llmsys-23-lora-lowrank-update`         | A              | Main mechanism.                                           |
| LoRA training stores adapter weights/gradients/states rather than optimizer state for all frozen base parameters.  | `llmsys-23-lora-training-state`         | A              | Memory mechanism.                                         |
| LoRA paper freezes pretrained weights and injects trainable rank-decomposition matrices.                           | `hu-2021-lora`                          | A              | Avoid ratio/quality claims without setup.                 |
| QLoRA combines quantization with low-rank training.                                                                | `llmsys-23-qlora-quantized-lora`        | A              | Bridge between halves.                                    |
| QLoRA backpropagates into LoRA adapters through a frozen quantized base model.                                     | `dettmers-2023-qlora`                   | A              | Mention NF4/double quantization only if sourced in draft. |

## Explanation Arc

1. Open from Chapter 10's memory ledger: after training-scale systems, adaptation and deployment still face memory pressure.
2. Define quantization as changing tensor representation, not changing the learned function intentionally.
3. Explain quantization error: rounding, clipping/range mismatch, and lost precision.
4. Introduce absmax and zero-point quantization as simple scale/offset mechanisms.
5. Explain PTQ versus training-time quantization; keep taxonomy compact.
6. Explain layer-wise objectives as matching full-precision and quantized layer outputs.
7. Use LLM.int8 to show outliers force mixed-precision paths.
8. Use GPTQ to show quantization can compensate errors using calibration-derived second-order information.
9. Transition to adaptation: full fine-tuning carries full gradients and optimizer states.
10. Define PEFT and its categories.
11. Explain LoRA as frozen base weights plus trainable low-rank update.
12. Explain LoRA memory savings through trainable-state reduction, not magic compression.
13. Explain QLoRA as quantized base model plus LoRA-style adapters.
14. Close with tradeoff: smaller/adaptable models require numerical care, kernel support, calibration, and adapter management.

## Required Figures

| Figure ID                      | Purpose                                                                         | Form                  | Source                                                            |
| ------------------------------ | ------------------------------------------------------------------------------- | --------------------- | ----------------------------------------------------------------- |
| `fig-11-quantization-map`      | Show float values mapped to low-bit buckets with scale and optional zero point. | Number line           | `llmsys-19-absmax-zeropoint`                                      |
| `fig-11-quantization-error`    | Distinguish rounding error, clipping, and range mismatch.                       | Three small plots     | `llmsys-19-direct-quantization-errors`                            |
| `fig-11-llmint8-outlier-path`  | Show normal 8-bit path plus higher-precision outlier path.                      | Dataflow              | `llmsys-19-llmint8-outliers`                                      |
| `fig-11-gptq-compensation`     | Show quantize column, compute error, update remaining columns.                  | Matrix block diagram  | `llmsys-20-gptq-blockwise-compensation`                           |
| `fig-11-full-ft-vs-lora-state` | Compare full fine-tuning state with LoRA trainable state.                       | Memory ledger         | `llmsys-23-full-finetuning-cost`, `llmsys-23-lora-training-state` |
| `fig-11-lora-update`           | Show `W' = W0 + A B` with low-rank matrices.                                    | Matrix factor diagram | `llmsys-23-lora-lowrank-update`                                   |
| `fig-11-qlora-stack`           | Show quantized frozen base plus trainable adapters.                             | Layer stack           | `llmsys-23-qlora-quantized-lora`                                  |

## Main Sections

### Compression and Adaptation Reopen the Memory Ledger

Bridge from Chapter 10. Training memory was about state residency; now the problem is how to store, serve, and adapt a trained model without carrying unnecessary precision or trainable state.

### Quantization: Change the Representation

Define quantization and its system promise: lower memory, lower bandwidth, sometimes faster kernels if hardware/runtime supports them. Pair every benefit with accuracy and implementation risk.

### Scale, Zero Point, and Error

Explain absmax and zero-point mechanisms. Keep equations simple and include dequantization conceptually. Explain rounding, clipping, and range mismatch.

### PTQ, QAT, and Layer-Wise Calibration

Explain the practical distinction between training-time quantization and post-training quantization. Use layer-wise output matching as the core calibration idea.

### Outliers and LLM.int8

Explain why outlier features make blanket low-bit conversion risky. Use LLM.int8 as the example of mixed precision: most values low precision, outliers higher precision.

### GPTQ: Quantize, Measure Error, Compensate

Explain GPTQ qualitatively. Avoid deep Hessian derivation in main prose; use a small mechanism diagram. If notation appears, state that it is layer-local and calibration-input dependent.

### Full Fine-Tuning Carries Full Training State

Reintroduce weights, gradients, optimizer states, and activations. Avoid slide memory numbers unless all assumptions are carried.

### PEFT: Change What Is Trainable

Define selective, reparameterization, and additive methods. State that this chapter focuses on LoRA because it is the clearest system mechanism.

### LoRA: Low-Rank Updates

Explain `W' = W0 + A B`, frozen base weights, low rank `r`, and trainable adapter matrices. Tie memory savings to optimizer/gradient state only for trainable matrices.

### QLoRA: Quantized Base, Trainable Adapters

Explain the combined idea: base model stored in low precision while low-rank adapters are trained. Mention QLoRA paper mechanisms only at high level unless adding a deeper source pass.

### Deployment Consequences

Discuss adapter switching, kernel support, mixed precision paths, and serving memory. Keep Part V serving details out of scope.

## Technical Checks

- Do not say quantization always speeds up inference; require kernel/hardware/memory-bandwidth conditions.
- Do not imply lower bitwidth preserves quality automatically.
- Do not quote memory or speed numbers without model, hardware, bitwidth, batch, sequence length, and software conditions.
- Do not present GPTQ as universally superior to other PTQ methods.
- Distinguish base parameters, trainable adapter parameters, gradients, optimizer states, and activations.
- Do not imply LoRA always matches full fine-tuning quality.
- Keep QLoRA details limited unless NF4/double quantization/paged optimizer are explicitly sourced in draft.

## Sidebar Decisions

- ZeroQuant: brief example only unless the chapter needs a PTQ/KD sidebar.
- LLM.int8: main quantization example for outliers.
- GPTQ: main advanced quantization mechanism.
- Adapter/prompt tuning: mention as PEFT categories, but keep LoRA central.
- QLoRA: use as bridge; deeper details optional.

## Open Questions

- Should the draft include simple absmax/zero-point equations, or keep them as figure annotations?
- How much GPTQ Hessian notation belongs in a systems chapter?
- Should QLoRA's NF4/double quantization/paged optimizer details be sourced and explained now or deferred?
- Should adapter switching at inference be discussed here or in serving chapters?

## Handoff

Owner: Book Architect  
Purpose: Chapter 11 brief from quantization, GPTQ, LoRA, and QLoRA source extraction  
Evidence grade: A for course lecture claims and primary papers; no benchmark numbers used  
Assumptions: Chapter 11 focuses on mechanisms and systems tradeoffs, not a complete compression/PEFT survey  
Open questions: GPTQ notation depth and QLoRA detail scope  
Handoff: Systems Explainer for Chapter 11 draft
