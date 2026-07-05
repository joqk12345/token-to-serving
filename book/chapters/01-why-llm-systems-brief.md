---
status: brief
chapter: 1
slug: 01-why-llm-systems
title: Why LLMs Are Systems Problems
primary_sources:
  - llmsys-01-intro-14e74a426e4a7e3ed485a026e1f65b70.pdf
secondary_sources: []
reader_level: engineer or graduate student with basic ML background
technical_depth: introductory but precise
---

# Why LLMs Are Systems Problems — Brief

## Chapter Thesis

LLMs look like general-purpose language interfaces, but their capabilities are made possible and limited by system design: tensor computation, memory movement, GPU kernels, distributed training, compression, scheduling, and serving abstractions.

## Reader Problem

The reader may know that LLMs generate text by predicting tokens, but not yet see why that objective becomes a full-stack systems problem once the model, dataset, context length, and serving demand scale.

## System Bottleneck

This chapter introduces all major bottleneck classes without solving them yet:

- compute;
- memory capacity;
- memory bandwidth;
- inter-device communication;
- data movement;
- scheduling;
- abstraction design.

## Source Map

| Claim                                                                     | Source card                        | Evidence grade | Notes                                                  |
| ------------------------------------------------------------------------- | ---------------------------------- | -------------- | ------------------------------------------------------ |
| LMs predict next-token probabilities.                                     | `llmsys-01-next-token-probability` | A              | Use as math anchor.                                    |
| Decoder-only autoregressive models dominate modern LLM architecture.      | `llmsys-01-decoder-only`           | A              | Keep model examples generic unless separately sourced. |
| Pretraining uses cross-entropy next-token loss.                           | `llmsys-01-training-objective`     | A              | Detailed formula can move to Ch. 2.                    |
| LLM systems span distributed systems, frameworks, operators, and kernels. | `llmsys-01-system-challenges`      | A              | Use as map of the book.                                |
| Scaling requires model-algorithm-system co-design.                        | `llmsys-01-codesign`               | A              | Install as recurring motif.                            |

## Explanation Arc

1. Start with visible capabilities: translation, summarization, code, math, tool use.
2. Ask what must happen for a model to generate one token.
3. Introduce next-token probability as the simplest mathematical contract.
4. Show the contract becomes a tensor program.
5. Show the tensor program becomes a hardware and runtime problem.
6. Map the book's layers: tokens, Transformer, GPU, framework, distributed training, compression, serving.
7. End with co-design: no layer can be optimized in isolation.

## Required Figures

| Figure ID                        | Purpose                                                      | Form          | Source                             |
| -------------------------------- | ------------------------------------------------------------ | ------------- | ---------------------------------- |
| `fig-01-visible-vs-hidden-stack` | Contrast LLM user capabilities with system layers.           | Stack diagram | `llmsys-01-system-challenges`      |
| `fig-01-token-probability-chain` | Show sequence probability as chained next-token predictions. | Token chain   | `llmsys-01-next-token-probability` |
| `fig-01-abstraction-levels`      | Show the abstraction ladder from model behavior to kernels and hardware. | Layered architecture diagram | `llmsys-01-system-challenges` |
| `fig-01-codesign-loop`           | Show model, algorithm, software, hardware feedback loop.     | Loop diagram  | `llmsys-01-codesign`               |

## Technical Checks

- Formula correctness: defer full derivation to Ch. 2, but do not confuse words and tokens.
- Complexity / memory accounting: only introduce categories; no detailed numbers unless sourced.
- Hardware assumptions: none yet.
- Benchmark conditions: avoid benchmark claims in this chapter.
- Terminology consistency: define LLM systems, token, decoder-only, pretraining, inference.

## Open Questions

- Should this chapter include an external source for GPT-3's 300B-token pretraining claim?
- Should the opening example use a concrete prompt, a production serving request, or a training-step walkthrough?
- How much CUDA terminology should appear before the GPU chapters?

## Handoff

Owner: Book Architect  
Purpose: Establish Chapter 1 structure and evidence envelope  
Evidence grade: A for course-framing claims  
Assumptions: Chapter 1 should be a map, not a deep technical derivation  
Open questions: Need principal decision on opening example  
Handoff: Systems Explainer for draft after source cards are reviewed
