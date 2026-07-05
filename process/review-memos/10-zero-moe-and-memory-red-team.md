# Red Team: Chapter 10 — ZeRO, MoE, and Training Memory

Date: 2026-07-05  
Reviewer: Codex  
Chapter: `book/chapters/10-zero-moe-and-memory.md`

## Verdict

Cleared for ready promotion.

The chapter handles ZeRO and MoE as two different memory-system interventions: ZeRO partitions replicated training state, while MoE sparsifies activated expert capacity. It avoids unsupported performance claims, carries assumptions for formulas, and keeps communication tradeoffs visible.

## Attacks and Outcomes

### Attack 1: The ZeRO formula box may look more precise than the assumptions support.

Outcome: Addressed.

The draft defines `N`, `M`, and `K`, identifies the accounting as simplified lecture framing, and states exclusions: activations, temporary buffers, fragmentation, framework overhead, sequence-length effects, micro-batch size, and communication staging. The text also says the formula is not total training memory.

### Attack 2: ZeRO-3 may sound like free memory reduction.

Outcome: Addressed.

The draft states that ZeRO-3 reduces memory at the cost of additional parameter transfer and frames parameter residency as a schedule. It does not claim parameter communication is hidden or free.

### Attack 3: The MoE section may overstate sparse activation by ignoring shared dense layers.

Outcome: Addressed.

The draft distinguishes total parameters from activated parameters per token and explicitly includes shared model components in the activated path. It also describes shared experts separately from routed experts.

### Attack 4: Router load balancing may sound like only a model-quality issue.

Outcome: Addressed.

The draft gives the router both a model role and a systems role. It connects routing decisions to expert specialization, device utilization, communication volume, and queueing at expert owners.

### Attack 5: Expert all-to-all may be confused with DDP all-reduce.

Outcome: Non-blocking.

The chapter explicitly contrasts MoE token movement with DDP all-reduce. A diagram is still recommended because token dispatch/return is easier to see visually, but the prose distinction is sufficient for ready status.

### Attack 6: MoE inference preview may leak into Part V.

Outcome: Non-blocking.

The inference section is short and explicitly says serving depth belongs to Part V. Its purpose is to prevent the misconception that sparse activation removes memory bandwidth, routing, communication, and kernel costs.

### Attack 7: ZeRO may be blurred with model parallelism.

Outcome: Addressed.

The draft repeatedly distinguishes ZeRO state partitioning from Chapter 9's pipeline/tensor model parallelism. It states that ZeRO is not another way to split layers, but a way to split training state that DDP would otherwise replicate.

## Required Fixes

None.

## Recommended Non-Blocking Follow-Up

- Add a ZeRO stage diagram showing optimizer-state, gradient, and parameter ownership across workers.
- Add a timeline for ZeRO-3 parameter gather/use/release.
- Add an expert-parallel all-to-all diagram contrasting token dispatch with DDP all-reduce.
- Add FSDP, ZeRO++, or offload comparisons only after adding official docs or primary source cards.

Owner: Red Team Reviewer  
Purpose: Chapter 10 adversarial review  
Evidence grade: A for reviewed source map; no new factual claims introduced beyond existing cards  
Assumptions: Red-team review evaluates readiness under current claim scope, not final copyediting or figure production  
Open questions: None blocking  
Handoff: Principal author may promote Chapter 10 to ready; next production focus can move to Chapter 11 source extraction
