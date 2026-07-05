# Technical Review: Chapter 1 — Why LLMs Are Systems Problems

Date: 2026-07-05  
Reviewer: Codex  
Chapter: `book/chapters/01-why-llm-systems.md`

## Verdict

Cleared for red-team review.

The chapter is technically coherent as the opening chapter. It frames LLMs as systems workloads without drifting into a model-history survey or unsupported scale claims. The chosen opening example is a user prompt unfolding into the infrastructure stack, which resolves the earlier opening-example question for the current draft.

## Checks

### Scope

- The chapter maps the book's system stack rather than overexplaining later mechanisms.
- It introduces next-token probability, Transformer workload shape, resource bottlenecks, abstraction levels, and co-design.
- It does not introduce historical model-scale numbers or trend claims that would require additional primary cards.

### Evidence

- Core claims are anchored to `llmsys-01` cards.
- No benchmark, latency, throughput, or model-scale numbers are used.
- Chapter-level claims do not rely on weak sources.

### Technical Framing

- "LLM systems" is defined as the full system required to train, adapt, optimize, and serve LLMs.
- The chapter distinguishes model capability from infrastructure.
- It correctly lists compute, memory, bandwidth, communication, and scheduling as bottleneck categories.
- It introduces co-design as a constraint rather than a slogan.

## Risks Remaining

| Risk | Blocking? | Notes |
|---|---:|---|
| No historical model-scale sidebar. | No | Deliberately omitted under sidebar plan. |
| Opening example is generic. | No | Appropriate for current draft and now resolved in STATUS. |
| No equations beyond next-token probability. | No | Appropriate for Chapter 1; Chapter 2 deepens computation. |

## Required Fixes

None.

## Red-Team Prompts

- Does the chapter overclaim what LLMs can do?
- Does it define systems too broadly to be useful?
- Does the resource list imply a single answer to "how many GPUs"?
- Does the co-design section become motivational rather than mechanistic?

Owner: Technical Reviewer  
Purpose: Chapter 1 technical review  
Evidence grade: A for course lecture framing; no benchmark or historical scale numbers used  
Assumptions: Review evaluates opening-frame correctness, not final copyediting or figure production  
Open questions: None blocking  
Handoff: Red Team reviewer for adversarial critique
