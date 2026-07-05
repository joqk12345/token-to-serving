# Technical Review: Chapter 9 — Model Parallelism

Date: 2026-07-05  
Reviewer: Codex  
Chapter: `book/chapters/09-model-parallelism.md`

## Verdict

Cleared for red-team review.

The chapter is technically coherent at draft level. It uses A-grade sources: the course model-parallel lecture, GPipe, PipeDream, Megatron-LM, and official PyTorch pipeline-parallelism documentation. It avoids benchmark numbers and avoids pipeline bubble or activation-memory formulas whose notation would require schedule-specific assumptions.

## Checks

### Chapter Boundary

- The opening correctly extends Chapter 8: DDP/data parallelism replicates the model, while model parallelism partitions model computation.
- ZeRO, optimizer-state partitioning, and MoE are not pulled into the main explanation; they remain Chapter 10 material.
- The chapter frames model parallelism as both a memory strategy and a scheduling/communication strategy, which matches the course source.

### Pipeline Parallelism

- Layer-wise pipeline partitioning is introduced at the right level.
- The naive pipeline idle-device problem is stated qualitatively and supported by the lecture.
- GPipe-style micro-batching is described as batch splitting to fill pipeline stages; no GPipe benchmark numbers are used.
- Pipeline bubbles are described qualitatively. No unsupported bubble formula is included.
- Boundary activation and activation-gradient communication are named explicitly.

### Activation Memory and Checkpointing

- The chapter correctly links in-flight micro-batches to activation storage pressure.
- Checkpointing/rematerialization is described as a memory-for-compute tradeoff.
- The draft avoids asymptotic checkpointing claims because no separate checkpointing paper card is present.

### 1F1B, PipeDream, and Interleaving

- 1F1B is framed as a schedule that starts backward earlier where dependencies allow.
- PipeDream is used as a representative pipelined execution source, not as a license to reproduce benchmark claims.
- Interleaving/chunking is presented as a schedule and balance technique, with implementation complexity acknowledged.
- Megatron-LM is used to support composition of tensor, pipeline, and data parallelism and interleaving, without numerical throughput claims.

### PyTorch Pipeline API

- The PyTorch pipeline API is treated as a runtime contract example, not as a stable universal abstraction.
- The official-docs alpha-status caveat is preserved.
- `PipelineStage` and schedule names are used only at documentation-summary level.

### Tensor Parallelism

- Matrix partitioning is explained through column and row splits.
- The draft correctly ties communication to the chosen partition: all-gather-like communication for replicated full outputs and reduction for partial sums.
- The FFN section uses the standard complementary split pattern: column split for the expansion projection, row split for the contraction projection, and reduction to combine final partial outputs.

### Attention and Embeddings

- Attention-head parallelism is carefully scoped to head-local computation.
- The chapter explicitly avoids claiming the entire attention block is communication-free.
- Embedding parallelism is limited to a cautionary sidebar-level treatment, which is appropriate given implementation sensitivity.

## Risks Remaining

| Risk | Blocking? | Notes |
|---|---:|---|
| No pipeline bubble formula. | No | Omitted intentionally; formulas require schedule assumptions. |
| No activation-memory formula. | No | Omitted intentionally; memory depends on schedule, micro-batch count, checkpointing, tensor shapes, and framework behavior. |
| PyTorch API may evolve. | No | Draft includes alpha caveat and avoids version-specific implementation internals. |
| Tensor-parallel collectives are simplified. | No | Adequate for draft; figures can later make layout transitions more precise. |
| Embedding parallelism is thin. | No | Appropriate for this chapter; deeper treatment can be deferred or placed in an advanced sidebar. |

## Required Fixes

None.

## Red-Team Prompts

- Does the chapter make model parallelism sound mainly like a memory fix rather than a scheduling problem?
- Does the GPipe timeline make bubbles clear enough without a formula?
- Does 1F1B need a diagram to avoid confusing the reader about warmup and drain?
- Does the FFN tensor-parallel example distinguish local intermediate slices from the final reduction?
- Does the attention section sufficiently prevent the misconception that attention tensor parallelism has no communication anywhere?
- Should the composition section state more explicitly that topology can dominate the choice of tensor versus pipeline degree?

Owner: Technical Reviewer  
Purpose: Chapter 9 technical review  
Evidence grade: A for course lecture, GPipe/PipeDream/Megatron papers, and PyTorch official pipeline docs  
Assumptions: Review evaluates draft-level correctness; later revisions may add diagrams or formulas with explicit assumptions  
Open questions: Whether to add pipeline bubble and activation-memory formulas in a later revision  
Handoff: Red Team reviewer for adversarial critique
