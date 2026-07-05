# Red Team: Chapter 1 — Why LLMs Are Systems Problems

Date: 2026-07-05  
Reviewer: Codex  
Chapter: `book/chapters/01-why-llm-systems.md`

## Verdict

Cleared for ready promotion.

The chapter succeeds as an opening frame. It defines LLM systems as the full training/adaptation/optimization/serving stack and avoids unsupported history, scale, or capability claims.

## Attacks and Outcomes

### Attack 1: The chapter may overclaim LLM capabilities.

Outcome: Addressed.

The draft names common interface capabilities but explicitly avoids claiming perfect task performance or human-like understanding.

### Attack 2: "Systems" may become too broad.

Outcome: Addressed.

The draft grounds systems in concrete bottlenecks: compute, memory, bandwidth, communication, scheduling, abstraction boundaries, kernels, training, and serving.

### Attack 3: The resource question may imply a single GPU-count answer.

Outcome: Addressed.

The draft says the answer is not one number but a resource model including parameters, gradients, optimizer states, activations, sequence length, precision, interconnect, checkpointing, and failures.

### Attack 4: Co-design may sound motivational.

Outcome: Addressed.

The draft gives concrete examples where an algorithm, quantization method, parallel strategy, or batching policy moves bottlenecks across layers.

## Required Fixes

None.

## Recommended Non-Blocking Follow-Up

- Add a historical model-scale sidebar only if supported by explicit primary/source cards.
- Add opening-stack figure artwork after final layout decisions.

Owner: Red Team Reviewer  
Purpose: Chapter 1 adversarial review  
Evidence grade: A for reviewed source map; no new factual claims introduced beyond existing cards  
Assumptions: Red-team review evaluates readiness under current scope, not final copyediting or figure production  
Open questions: None blocking  
Handoff: Principal author may promote Chapter 1 to ready
