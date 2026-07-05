# Technical Review: Chapter 11 — Quantization and Parameter-Efficient Adaptation

Date: 2026-07-05  
Reviewer: Codex  
Chapter: `book/chapters/11-quantization-and-peft.md`

## Verdict

Cleared for red-team review.

The chapter is technically coherent at draft level. It uses A-grade sources: the course quantization, GPTQ, and PEFT lectures plus primary papers for LLM.int8, GPTQ, LoRA, and QLoRA. It avoids benchmark numbers and does not make unconditional speed, memory, or quality claims.

## Checks

### Chapter Boundary

- The chapter follows naturally from Chapter 10's memory-ledger framing.
- Serving consequences are mentioned only as runtime-contract implications; detailed serving architecture remains in Part V.
- The chapter does not attempt a full survey of all quantization or PEFT methods.

### Quantization Basics

- Quantization is framed as representation change, not an accuracy-preserving compression guarantee.
- The draft explains scale and zero point using simple equations.
- Error mechanisms are separated into rounding, clipping, and range mismatch.
- The draft states that performance depends on hardware, kernels, operation, and memory behavior.

### PTQ and Calibration

- Post-training quantization is described as behavior preservation using calibration data and local adjustments.
- The layer-wise objective is scoped to linear layers and calibration inputs.
- The draft correctly warns that local layer matching does not eliminate accumulated network-level error.

### LLM.int8 and Outliers

- The chapter uses LLM.int8 as an example of mixed precision for outlier features.
- It does not reproduce memory or performance results.
- It correctly distinguishes the common low-precision path from the higher-precision outlier path.

### GPTQ

- GPTQ is explained through quantize-error-compensate rather than a full Hessian derivation.
- The draft uses approximate second-order information only at mechanism level.
- It avoids bitwidth, runtime, and speed claims.
- It frames lazy updates and custom kernels as systems considerations without claiming guaranteed inference benefit.

### PEFT and LoRA

- Full fine-tuning is correctly tied to full gradients and optimizer state.
- PEFT is defined as reducing trainable state, not as guaranteed quality equivalence.
- LoRA is explained as frozen base weight plus low-rank trainable update.
- The draft distinguishes stored base weights from trainable adapter weights, gradients, and optimizer state.
- It does not imply activation memory disappears under LoRA.

### QLoRA

- QLoRA is correctly framed as a quantized frozen base model plus trainable low-rank adapters.
- NF4, double quantization, and paged optimizers are mentioned as paper mechanisms but not explained in depth without a deeper source pass.
- No single-GPU, model-size, or quality claims are used.

## Risks Remaining

| Risk | Blocking? | Notes |
|---|---:|---|
| Simple quantization equations may hide granularity choices. | No | Draft names implementation variation; later figures can show per-tensor/per-channel choices. |
| GPTQ Hessian explanation is qualitative. | No | Appropriate for systems chapter; avoids notation risk. |
| QLoRA details are shallow. | No | Explicitly scoped; deeper NF4/paged-optimizer treatment can be added later. |
| Adapter deployment section is short. | No | Serving depth belongs in Part V. |
| No benchmark numbers. | No | Correct under evidence rules. |

## Required Fixes

None.

## Red-Team Prompts

- Does the chapter imply lower bitwidth automatically improves runtime?
- Does the quantization section carry enough warning about calibration data and accumulated error?
- Does the GPTQ section make second-order information sound exact rather than approximate?
- Does LoRA appear to reduce base model memory, or only trainable optimizer/gradient state?
- Does QLoRA need deeper NF4/double-quantization explanation before ready status?
- Does deployment discussion overlap too much with serving chapters?

Owner: Technical Reviewer  
Purpose: Chapter 11 technical review  
Evidence grade: A for course lectures and primary papers; no benchmark numbers used  
Assumptions: Review evaluates draft-level correctness; final copy may add figures or deeper QLoRA source cards later  
Open questions: Whether to add NF4/double-quantization/paged-optimizer source details in a later revision  
Handoff: Red Team reviewer for adversarial critique
