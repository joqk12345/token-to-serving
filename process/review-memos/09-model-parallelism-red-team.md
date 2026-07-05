# Red Team: Chapter 9 — Model Parallelism

Date: 2026-07-05  
Reviewer: Codex  
Chapter: `book/chapters/09-model-parallelism.md`

## Verdict

Cleared for ready promotion.

The chapter explains model parallelism as partitioning plus scheduling, not as a simple memory-fit trick. It avoids unsupported performance claims, avoids schedule-specific formulas without assumptions, and preserves the important communication caveats for pipeline, tensor, attention, and embedding parallelism.

## Attacks and Outcomes

### Attack 1: The chapter may imply model parallelism is mainly a memory fix.

Outcome: Addressed.

The opening and closing repeatedly frame model parallelism as a communication and scheduling problem. Pipeline bubbles, activation storage, boundary communication, collectives, and topology all appear as first-class constraints.

### Attack 2: The GPipe timeline may hide bubble overhead.

Outcome: Non-blocking.

The timeline is simplified, but the text immediately names warmup and drain bubbles and explicitly avoids formula claims. A future figure can show filled, idle, warmup, and drain cells more visually, but the current draft is not misleading.

### Attack 3: 1F1B may confuse readers without a full timeline.

Outcome: Non-blocking.

The chapter describes 1F1B as warmup, alternating forward/backward where dependencies allow, and drain. That is enough for draft-level conceptual treatment. A final-layout figure should be added later because the schedule is easier to see than to describe.

### Attack 4: The FFN tensor-parallel example may hide the collective.

Outcome: Addressed.

The draft states that the final output requires combining partial contributions and calls that combination a reduction across the tensor-parallel group. It also explains why the expanded intermediate can remain partition-local.

### Attack 5: Attention head parallelism may be read as communication-free attention.

Outcome: Addressed.

The draft explicitly limits the no-all-reduce statement to head-local work and warns that output projection or layout transitions may require reduction, all-gather, or another communication step.

### Attack 6: Embedding parallelism may be too thin.

Outcome: Non-blocking.

The chapter intentionally treats embeddings as a special case and avoids universal claims. This is the right scope for Chapter 9. If later chapters discuss vocabulary-parallel output heads or loss fusion, those details can be expanded with implementation-specific source cards.

### Attack 7: Composition may understate topology.

Outcome: Addressed.

The composition section names hardware topology, interconnect bandwidth, model shape, batch size, sequence length, precision, and runtime support as inputs to the parallelism choice. It also states that tensor parallelism wants fast collectives and pipeline parallelism wants balanced stages and enough micro-batches.

## Required Fixes

None.

## Recommended Non-Blocking Follow-Up

- Add a pipeline schedule figure showing naive, GPipe-style micro-batching, and 1F1B with warmup/drain cells.
- Add a tensor-parallel FFN figure showing column split, row split, and final reduction.
- Add a small sidebar explaining that production parallelism plans are topology-sensitive rather than purely model-architecture decisions.
- Add bubble or activation-memory formulas only if the chapter carries explicit schedule and tensor-shape assumptions.

Owner: Red Team Reviewer  
Purpose: Chapter 9 adversarial review  
Evidence grade: A for reviewed source map; no new factual claims introduced beyond existing cards  
Assumptions: Red-team review evaluates readiness under current claim scope, not final copyediting or figure production  
Open questions: None blocking  
Handoff: Principal author may promote Chapter 9 to ready; next production focus can move to Chapter 10 source extraction
