---
status: brief
chapter: 15
slug: 15-llm-system-codesign
title: Model-Algorithm-System Co-Design
primary_sources:
  - llmsys-01-intro-14e74a426e4a7e3ed485a026e1f65b70.pdf
papers: []
secondary_sources: []
reader_level: engineer or graduate student who has read Chapters 1-14
technical_depth: synthesis
---

# Model-Algorithm-System Co-Design

## Chapter Thesis

The durable skill in LLM systems is reasoning across model architecture, algorithm, software runtime, and hardware constraints at the same time. Scaling an LLM system is not only "make the model larger" or "make kernels faster"; it is a co-design loop across objective, architecture, memory, communication, compression, scheduling, and product constraints.

## Reader Problem

The reader has moved from next-token probability through Transformers, GPUs, kernels, distributed training, compression, inference, KV cache, and disaggregated serving. The final chapter should not introduce a new subsystem. It should teach the reader how to combine the previous chapters into a design method: identify the bottleneck, locate the abstraction level where it can be changed, and reason about the downstream tradeoffs.

## System Bottleneck

Primary bottlenecks: cross-layer tradeoff reasoning, abstraction boundaries, compute, memory, communication, data movement, scheduling, compression, scaling up, scaling down, and deciding which layer owns a bottleneck.

Secondary bottlenecks: training/serving mismatch, workload variability, hardware availability, model architecture choice, software optimization, framework/runtime support, evaluation methodology, and product integration.

## Source Map

| Claim                                                                                                                                                       | Source card                        | Evidence grade | Notes                                        |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- | -------------- | -------------------------------------------- |
| Language modeling begins with next-token probability conditioned on prompt and prior tokens.                                                                | `llmsys-01-next-token-probability` | A              | Return to mathematical root.                 |
| Decoder-only autoregressive models are presented as the most popular modern LLM architecture choice.                                                        | `llmsys-01-decoder-only`           | A              | Connect architecture to system consequences. |
| Pretraining commonly uses cross-entropy for next-token prediction over raw text.                                                                            | `llmsys-01-training-objective`     | A              | Objective-level anchor.                      |
| LLM systems must address compute, memory, data movement, communication, abstractions, distributed systems, frameworks, operators, kernels, and compression. | `llmsys-01-system-challenges`      | A              | Main synthesis map.                          |
| LLM system design requires model-algorithm-system co-design across architecture, algorithms, software optimization, and hardware acceleration.              | `llmsys-01-codesign`               | A              | Chapter thesis.                              |

## Explanation Arc

1. Open by returning to next-token prediction and showing how the simple objective expands into the whole system stack.
2. Revisit the book's progression: objective -> Transformer computation -> GPU kernels -> distributed training -> memory/compression -> serving.
3. Define co-design as choosing across layers, not optimizing each layer independently.
4. Show the "bottleneck ownership" method: compute, memory, communication, scheduling, reliability, or abstraction.
5. Present a design checklist for diagnosing an LLM system problem.
6. Use three synthesis examples:
   - long context: attention algorithm, KV cache, memory hierarchy, serving scheduler;
   - large training run: model partitioning, optimizer state, communication overlap, framework/runtime;
   - cheap adaptation/serving: quantization, LoRA/QLoRA, kernel support, adapter deployment.
7. Explain scaling up versus scaling down as different co-design directions.
8. Close with what to remember: every number travels with conditions; every abstraction hides a cost; every optimization moves a bottleneck.

## Required Figures

| Figure ID                              | Purpose                                                                                    | Form                 | Source                        |
| -------------------------------------- | ------------------------------------------------------------------------------------------ | -------------------- | ----------------------------- |
| `fig-15-codesign-loop`                 | Show model architecture, algorithm, software/runtime, and hardware acceleration as a loop. | Four-node loop       | `llmsys-01-codesign`          |
| `fig-15-stack-recap`                   | Recap the book stack from objective to serving infrastructure.                             | Layered stack        | `llmsys-01-system-challenges` |
| `fig-15-bottleneck-routing`            | Show how a bottleneck routes to compute/memory/communication/scheduling/abstraction fixes. | Decision flow        | `llmsys-01-system-challenges` |
| `fig-15-optimization-moves-bottleneck` | Show one optimization shifting pressure to another resource.                               | Cause/effect diagram | Synthesis from Chapters 7-14  |

## Main Sections

### The Simple Objective Becomes a System

Return to next-token prediction and show how objective, architecture, kernels, training, and serving are linked.

### Co-Design Means Joint Constraints

Define model-algorithm-system co-design. Explain why optimizing architecture, algorithm, software, or hardware in isolation can move rather than remove bottlenecks.

### Bottleneck Ownership

Give the reader a method: identify whether the bottleneck is compute, memory, bandwidth, communication, scheduling, reliability, or abstraction. Then ask which layer has the cheapest safe intervention.

### Synthesis Example: Long Context

Connect attention complexity, FlashAttention, KV cache, PagedAttention, memory tiers, and serving routing.

### Synthesis Example: Large Training

Connect DDP, model parallelism, pipeline/tensor parallelism, ZeRO, MoE, communication, optimizer state, and frameworks.

### Synthesis Example: Cheap Adaptation and Serving

Connect quantization, PEFT, LoRA/QLoRA, kernel support, memory bandwidth, and deployment complexity.

### Scaling Up and Scaling Down

Scaling up asks how to train/serve larger models and longer contexts. Scaling down asks how to fit cost, latency, power, and hardware constraints. Both require co-design.

### The Engineering Checklist

Provide a final checklist:

```text
What is the workload?
What is the SLO or training target?
Where is the bottleneck?
Which layer owns the cheapest intervention?
What assumptions travel with each number?
What new bottleneck does the intervention create?
What evidence supports the claim?
```

## Technical Checks

- Do not introduce new performance numbers.
- Do not predict future system trends beyond sourced course framing.
- Do not turn the chapter into a motivational conclusion.
- Keep claims tied to mechanisms already developed in earlier chapters.
- Avoid presenting an optimization as correct without workload and system conditions.
- Make clear that co-design is a reasoning method, not a named product or fixed architecture.

## Sidebar Decisions

- kTransformers/heterogeneous hardware: possible sidebar if the draft wants a final co-design example.
- Figures from earlier chapters: reuse concepts, not screenshots.
- Part I unresolved opening question remains separate; do not retrofit Chapter 1 here.

## Open Questions

- Should this chapter include a table mapping each previous chapter to its dominant bottleneck?
- Should kTransformers appear here as a heterogeneous-serving co-design sidebar?
- Should Chapter 15 include exercises/checklists for readers designing their own system?
- Does the final chapter need additional primary sources beyond Lecture 1, or can it synthesize prior chapter sources?

## Handoff

Owner: Book Architect  
Purpose: Chapter 15 brief from system/co-design source extraction and prior chapter synthesis  
Evidence grade: A for course framing; synthesis relies on already sourced Chapters 1-14  
Assumptions: Chapter 15 is a synthesis chapter, not a new technical subsystem  
Open questions: Whether to add a chapter-to-bottleneck summary table and kTransformers sidebar  
Handoff: Systems Explainer for Chapter 15 draft
