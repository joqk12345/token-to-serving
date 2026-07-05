# Technical Review: Chapter 7 — Deep Learning Frameworks, JAX, XLA, and TPU

Date: 2026-07-05  
Reviewer: Codex  
Chapter: `book/chapters/07-dl-frameworks-and-compilers.md`

## Verdict

Cleared for red-team review.

The chapter is technically coherent at draft level. Its main claims are supported by A-grade lecture PDFs and initial official JAX/OpenXLA/Pallas documentation cards. The draft avoids unconditioned speedup claims and does not use raw benchmark numbers.

## Changes Made During Review

- Fixed citation marker escaping from `\[CITE: ...]` to `[CITE: ...]`.
- Fixed opening training-step pseudocode. The original form used `grad(loss)(params)`, which could be read as applying `grad` to a value rather than a function. It now uses a `loss_fn` with `value_and_grad(loss_fn)`.
- Fixed escaped `CUDA C\++` to `CUDA C++`.

## Checks

### Equations and Pseudocode

- Reverse-mode autodiff toy example is correct for `y = (a * b) + c` under upstream seed `g_y = 1`.
- Finite-difference formula is acceptable as a gradient-checking example.
- Opening training-step pseudocode is now conceptually correct as JAX-like pseudocode.

### Compiler Pipeline

- The path `Python/JAX tracing → JAXpr → StableHLO → HLO → optimized executable` is acceptable as a pedagogical path for this chapter.
- The draft correctly scopes backend-specific details and avoids implying that every framework follows exactly this stack.
- StableHLO and XLA claims are backed by official OpenXLA documentation cards.

### Dynamic Shape / Static Shape Wording

- The draft includes the necessary caveat that modern compiler stacks may support dynamic shape or shape polymorphism.
- The safe statement is retained: more shape/layout information enables stronger memory planning and specialization.

### Hardware Claims

- TPU/MXU/systolic-array discussion is qualitative and tied to the lecture source.
- The draft does not include hardware-generation-specific numbers.
- Recommendation: add narrower official TPU/backend documentation before final readiness if the chapter later adds detailed TPU microarchitecture claims.

### Pallas Claims

- Pallas is correctly described as experimental based on official JAX docs.
- `grid`, `BlockSpec`, VMEM/HBM, pipelining, output aliasing, and tile-size sensitivity are grounded in lecture cards.
- Recommendation: add narrower official Pallas `BlockSpec` and TPU backend cards before final readiness if the draft becomes API-specific.

### Performance Claims

- No speedup, throughput, bandwidth, or latency number is used in chapter prose.
- Tile-size and fusion claims are conditional and workload-dependent.
- The draft explicitly avoids quoting Pallas speedup numbers without full conditions.

## Risks Remaining

| Risk | Blocking? | Notes |
|---|---:|---|
| JAXpr explanation is simplified. | No | Acceptable for chapter level; red team may ask for a small JAXpr example. |
| StableHLO compatibility wording is general. | No | If expanded, use exact official wording. |
| TPU backend details rely mainly on lecture. | No | Current claims are qualitative; detailed claims need official/source evidence. |
| Pallas API may change. | No | Draft explicitly says experimental. |
| Sharding section previews Chapters 8–10. | No | Scope is controlled and not a substitute for later chapters. |

## Red-Team Prompts

- Does the chapter over-sell JAX/XLA/TPU as the default LLM systems stack?
- Does the JAX tracing explanation hide too many sharp edges?
- Does the compiler pipeline imply too linear a flow?
- Are Pallas and Splash Attention introduced without stealing Chapter 6's attention-algorithm role?
- Does the chapter separate framework, compiler, runtime, and custom-kernel responsibilities?

Owner: Technical Reviewer  
Purpose: Chapter 7 technical review  
Evidence grade: A for cited lecture PDFs and official JAX/OpenXLA/Pallas docs; narrower official/source cards recommended for final API or microarchitecture specificity  
Assumptions: Technical review evaluates draft-level correctness, not final publication readiness  
Open questions: Whether to add exact JAXpr/Pallas BlockSpec examples before red-team or after red-team feedback  
Handoff: Red Team reviewer for adversarial critique
