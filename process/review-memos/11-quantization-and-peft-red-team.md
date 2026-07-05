# Red Team: Chapter 11 — Quantization and Parameter-Efficient Adaptation

Date: 2026-07-05  
Reviewer: Codex  
Chapter: `book/chapters/11-quantization-and-peft.md`

## Verdict

Cleared for ready promotion.

The chapter treats quantization and PEFT as systems tradeoffs rather than free compression. It avoids benchmark claims, keeps accuracy and runtime caveats visible, and distinguishes base-model storage from trainable adapter state.

## Attacks and Outcomes

### Attack 1: The chapter may imply lower bitwidth automatically improves runtime.

Outcome: Addressed.

The draft explicitly says lower precision does not automatically mean faster inference and lists conditions: dequantization cost, hardware low-bit support, kernel implementation, and whether memory movement is the bottleneck.

### Attack 2: Quantization may appear to preserve quality by default.

Outcome: Addressed.

The draft explains rounding, clipping, range mismatch, calibration dependence, outlier handling, and accumulated layer error. It does not claim quality preservation without conditions.

### Attack 3: GPTQ second-order information may sound exact.

Outcome: Addressed.

The draft uses the GPTQ paper's approximate second-order framing and avoids a full Hessian derivation. It describes curvature-like information as a calibration-dependent mechanism, not an exact global model property.

### Attack 4: LoRA may be read as reducing base-model memory rather than trainable state.

Outcome: Addressed.

The draft states that the base matrix still participates in forward and backward passes, that LoRA does not make the base model disappear, and that its main training-memory effect is reducing trainable parameter state.

### Attack 5: QLoRA details may be too shallow.

Outcome: Non-blocking.

The draft states the structural QLoRA idea and cites the paper mechanisms: frozen 4-bit base model, LoRA adapters, NF4, double quantization, and paged optimizers. It explicitly leaves detailed NF4/paged-optimizer explanation out of scope pending a deeper source pass.

### Attack 6: Deployment discussion may overlap with serving chapters.

Outcome: Non-blocking.

The deployment section is short and framed as runtime-contract context. It explicitly leaves serving architecture to Part V.

### Attack 7: PEFT may sound quality-equivalent to full fine-tuning.

Outcome: Addressed.

The draft states that PEFT can constrain adaptation capacity and that quality depends on task, adapter rank, placement, optimizer behavior, and runtime details. It avoids quality equivalence claims.

## Required Fixes

None.

## Recommended Non-Blocking Follow-Up

- Add a figure for rounding versus clipping versus range mismatch.
- Add a GPTQ matrix diagram showing column quantization, error calculation, and lazy update.
- Add a memory-ledger figure contrasting full fine-tuning with LoRA.
- Add a deeper QLoRA sidebar only after sourcing NF4, double quantization, and paged optimizers in detail.

Owner: Red Team Reviewer  
Purpose: Chapter 11 adversarial review  
Evidence grade: A for reviewed source map; no new factual claims introduced beyond existing cards  
Assumptions: Red-team review evaluates readiness under current claim scope, not final copyediting or figure production  
Open questions: None blocking  
Handoff: Principal author may promote Chapter 11 to ready; next production focus can move to Chapter 12 source extraction
