# Red Team: Chapter 7 — Deep Learning Frameworks, JAX, XLA, and TPU

Date: 2026-07-05  
Reviewer: Codex  
Chapter: `book/chapters/07-dl-frameworks-and-compilers.md`

## Verdict

Cleared for ready promotion.

The chapter survives adversarial review at the current book-production standard. It has no unsupported benchmark claims, no unconditioned speedup claims, no chapter-level claim resting only on weak evidence, and no obvious formula/pseudocode blocker after the technical-review fix.

## Attacks and Outcomes

### Attack 1: The chapter might over-sell JAX/XLA/TPU as the default LLM systems stack.

Outcome: Addressed.

The draft now explicitly says JAX/XLA/TPU are used as a visible case study, not as a recommendation that all LLM systems should use that stack. This is enough for the current chapter scope.

### Attack 2: The compiler pipeline might imply a rigid linear path.

Outcome: Addressed.

The draft now states that real implementations may include caches, frontend-specific paths, runtime dispatch layers, and backend-specific shortcuts. The pipeline is framed as a teaching path for abstractions, not a strict implementation promise.

### Attack 3: The JAX tracing explanation may hide sharp edges.

Outcome: Non-blocking.

The chapter does mention traceable control-flow constraints and the staged-programming tradeoff. It does not teach all JAX sharp edges, but that is acceptable for a systems chapter whose goal is framework/compiler architecture, not a JAX manual.

Recommended future polish: add one small JAXpr example if the chapter expands.

### Attack 4: TPU claims may be too hardware-specific for lecture-only evidence.

Outcome: Non-blocking.

The chapter keeps TPU discussion qualitative and explicitly warns against treating it as a full hardware specification. It does not include generation-specific throughput, memory, or latency numbers.

Recommended future polish: add official TPU/backend source cards if the chapter later adds microarchitecture detail.

### Attack 5: Pallas may be unstable API territory.

Outcome: Addressed.

The chapter cites official Pallas docs and explicitly labels Pallas experimental. The Pallas section is used for systems concepts—grid, blocks, memory movement, pipelining—not as a stable API tutorial.

### Attack 6: Splash Attention may steal Chapter 6's attention-algorithm role.

Outcome: Non-blocking.

The Splash Attention section is short and explicitly says Chapter 6 owns the attention algorithm. Here it is used only as a boundary case for compiler/kernel abstraction.

### Attack 7: The chapter may blur framework, compiler, runtime, and custom-kernel responsibilities.

Outcome: Non-blocking.

The chapter distinguishes graph/autodiff, JAX transformations, StableHLO/HLO compiler IR, XLA backend execution, runtime compilation/caching, sharding/collectives, and Pallas custom kernels. The separation is enough for draft readiness.

## Required Fixes

None.

## Recommended Non-Blocking Follow-Up

- Add a concrete JAXpr output example if the chapter needs a more grounded tracing section.
- Add narrower official/source cards for Pallas `BlockSpec` and TPU backend behavior if later revisions add API or microarchitecture specifics.
- Consider a small figure for `Python function → JAXpr → StableHLO/HLO → executable → Pallas escape hatch`.

Owner: Red Team Reviewer  
Purpose: Chapter 7 adversarial review  
Evidence grade: A for reviewed source map; no new factual claims introduced beyond existing cards  
Assumptions: Red-team review evaluates whether the draft can advance to ready, not whether it is final copyedited manuscript text  
Open questions: None blocking  
Handoff: Principal author may promote Chapter 7 to ready; next production focus can move to Chapter 8 source extraction
