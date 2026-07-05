# Technical Review: Chapter 15 — Model-Algorithm-System Co-Design

Date: 2026-07-05  
Reviewer: Codex  
Chapter: `book/chapters/15-llm-system-codesign.md`

## Verdict

Cleared for red-team review.

The draft works as a synthesis chapter. It returns to the next-token objective, uses Lecture 1's co-design framing, and then draws on already sourced mechanisms from prior chapters to show how bottlenecks move across model architecture, algorithms, runtimes, memory systems, distributed training, compression, and serving.

## Checks

### Chapter Boundary

- The chapter does not introduce a new subsystem.
- It synthesizes Chapters 1-14 and keeps the focus on design method.
- It avoids motivational conclusion language and stays mechanism-oriented.

### Evidence Use

- Core chapter framing is supported by `llmsys-01` cards.
- Synthesis examples cite prior chapter cards for FlashAttention, DDP, model parallelism, ZeRO, quantization, LoRA/QLoRA, PagedAttention, KV-aware routing, and disaggregation.
- No new benchmark numbers are introduced.
- The STATUS source count was corrected to reflect 23 cited source cards.

### Co-Design Framing

- The draft defines co-design as joint reasoning across model architecture, algorithm, software/runtime, and hardware.
- It correctly emphasizes that local optimizations can move bottlenecks.
- It gives a practical bottleneck-ownership method rather than a vague conclusion.

### Synthesis Examples

- Long context connects attention acceleration, KV-cache management, PagedAttention, routing, and KV transfer.
- Large training connects DDP, tensor/pipeline parallelism, ZeRO, memory residency, and communication.
- Cheap adaptation/serving connects quantization, LoRA/QLoRA, numerical risk, runtime support, and deployment state.
- Goodput-oriented serving connects TTFT/TPOT, prefill/decode characteristics, disaggregation, and KV-transfer cost.

### Technical Discipline

- The draft does not claim any optimization is condition-free.
- It repeatedly asks for workload, target, bottleneck, owner layer, intervention, new bottleneck, and evidence.
- It avoids future predictions and current product claims.

## Risks Remaining

| Risk | Blocking? | Notes |
|---|---:|---|
| The chapter could benefit from a chapter-to-bottleneck table. | No | Useful polish, not required for readiness. |
| The source map in the brief lists only Lecture 1 cards. | No | Draft citations and STATUS reflect prior-source reuse; brief can be expanded later if desired. |
| Some synthesis claims rely on prior chapter context. | No | Expected for final synthesis chapter. |
| Heterogeneous hardware/kTransformers sidebar is omitted. | No | Can be added later with source support. |

## Required Fixes

None.

## Red-Team Prompts

- Does the chapter become too abstract, losing the concrete systems mechanisms from earlier chapters?
- Does it imply co-design is a generic slogan rather than a decision method?
- Does it introduce new unsupported claims while summarizing previous chapters?
- Does it overstate decoder-only dominance or modern architecture stability?
- Does the checklist sufficiently enforce conditions-travel-with-numbers discipline?

Owner: Technical Reviewer  
Purpose: Chapter 15 technical review  
Evidence grade: A for course framing and cited prior chapter cards; no benchmark numbers used  
Assumptions: Review evaluates synthesis coherence and claim discipline, not final copyediting or figure production  
Open questions: Whether to add a chapter-to-bottleneck table in a later revision  
Handoff: Red Team reviewer for adversarial critique
