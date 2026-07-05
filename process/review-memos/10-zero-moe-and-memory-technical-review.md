# Technical Review: Chapter 10 — ZeRO, MoE, and Training Memory

Date: 2026-07-05  
Reviewer: Codex  
Chapter: `book/chapters/10-zero-moe-and-memory.md`

## Verdict

Cleared for red-team review.

The chapter is technically coherent at draft level. It uses A-grade sources: the course ZeRO and MoE lectures plus primary papers for ZeRO, Switch Transformer, GShard, DeepSpeed-MoE, and DeepSeekMoE. It avoids benchmark numbers and carries explicit assumptions for the ZeRO formula box.

## Checks

### Chapter Boundary

- The chapter follows naturally from Chapter 9: model parallelism partitions computation, while Chapter 10 focuses on resident training state and sparse expert activation.
- Serving detail is kept as a preview only; vLLM, KV cache, and serving scheduling remain in Part V.
- FSDP, ZeRO++, CPU/NVMe offload, and DeepSeek-V3 specifics are not introduced without source cards.

### Memory Ledger

- The opening correctly broadens memory beyond parameters to include gradients, optimizer state, activations, temporary buffers, fragmentation, and implementation overhead.
- The draft avoids total-memory numbers without model, precision, sequence length, batch/micro-batch, optimizer, and hardware context.
- The resident-memory expression is qualitative and appropriately framed as a ledger rather than a complete formula.

### ZeRO Stage Semantics

- ZeRO is described as partitioning data-parallel training state, not as compression or layer partitioning.
- ZeRO-1 is scoped to optimizer-state partitioning.
- ZeRO-2 is scoped to gradient partitioning and correctly distinguishes temporary gradient computation from retained gradient shards.
- ZeRO-3 is scoped to parameter partitioning and parameter availability during forward/backward.
- The draft states the ZeRO-3 communication tradeoff instead of presenting parameter partitioning as free memory savings.

### Formula Box

- The formula box explicitly defines `N`, `M`, and `K`.
- The draft states that formulas are under the lecture's simplified mixed-precision-style accounting.
- The draft excludes activations, temporary buffers, fragmentation, framework overhead, sequence-length effects, micro-batch size, and communication staging from the formula.
- This is acceptable for draft-level explanation.

### MoE Mechanism

- MoE is correctly introduced as replacing dense FFNs with multiple expert FFNs and a router.
- The draft distinguishes total parameters from activated parameters per token.
- Switch-style top-1 routing is used only as a simple example and is backed by lecture plus paper cards.
- The draft does not claim MoE is universally faster, cheaper, or higher quality than dense models.

### Expert Parallelism and Communication

- Expert parallelism is framed as expert placement plus token dispatch/return.
- All-to-all communication is treated as a central bottleneck, not an implementation detail.
- The chapter lists conditions that affect all-to-all cost and avoids topology-free latency claims.
- GShard and DeepSpeed-MoE are used for systems framing without reproducing benchmark numbers.

### Load Balancing and Shared Experts

- Router load balancing is framed as both model behavior and systems load control.
- Routing collapse is named, but formulas are not reproduced without notation review.
- Shared/fine-grained experts are handled at mechanism level and backed by DeepSeekMoE paper card.
- No DeepSeek-V3 architecture numbers are used.

## Risks Remaining

| Risk | Blocking? | Notes |
|---|---:|---|
| ZeRO formulas may still invite over-interpretation. | No | The draft's assumptions and exclusions are explicit. |
| MoE inference preview may overlap with serving chapters. | No | Current treatment is short and scoped to training-chapter context. |
| No FSDP comparison. | No | Correctly omitted until official docs/source cards are added. |
| ZeRO++ and offload are thin. | No | Correctly left out or deferred because no source extraction was done. |
| No figure yet for all-to-all or ZeRO stages. | No | Figure specs exist in the brief; final layout can add them. |

## Required Fixes

None.

## Red-Team Prompts

- Does the ZeRO formula box make the chapter look more precise than its assumptions support?
- Does the ZeRO-3 section imply parameter communication is easy to hide?
- Does the MoE section overstate sparse activation by ignoring shared dense layers?
- Does the router-as-scheduler framing adequately connect learning load balance to hardware load balance?
- Does expert parallelism need a diagram to avoid confusing all-to-all with all-reduce?
- Does the chapter sufficiently distinguish ZeRO state sharding from model parallelism?

Owner: Technical Reviewer  
Purpose: Chapter 10 technical review  
Evidence grade: A for course lectures and primary papers; no benchmark numbers used  
Assumptions: Review evaluates draft-level correctness; final copy may add diagrams or additional official docs later  
Open questions: Whether to add FSDP, ZeRO++, or offload source cards in a later revision  
Handoff: Red Team reviewer for adversarial critique
