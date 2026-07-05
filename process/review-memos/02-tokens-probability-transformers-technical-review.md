# Technical Review: Chapter 2 — From Next-Token Probability to Transformer Computation

Date: 2026-07-05  
Reviewer: Codex  
Chapter: `book/chapters/02-tokens-probability-transformers.md`

## Verdict

Cleared for red-team review.

The chapter is technically coherent at introductory-to-intermediate level. The earlier evidence gap is closed: canonical Transformer, T5, LLaMA, RoPE, and related architecture claims now have primary-paper source cards. The chapter makes Transformer computation legible without moving into GPU-kernel detail.

## Checks

### Source Coverage

- Original Transformer paper supports attention, encoder-decoder framing, and Transformer block structure.
- T5 paper supports text-to-text encoder-decoder framing.
- LLaMA paper supports decoder-only architecture, RoPE use, pre-normalization, and SwiGLU mention.
- RoFormer paper supports RoPE.
- All cited source-ledger cards exist.

### Architecture and Notation

- Next-token probability is connected to autoregressive computation without overloading word/token terminology.
- Embedding and positional information are introduced at the right abstraction level.
- Attention is explained through Q/K/V, softmax weighting, and multi-head structure.
- Causal masking is connected to autoregressive factorization.
- FFN, residuals, and normalization are described as part of the Transformer block.

### Chapter Boundary

- Kernel-level cost is named but deferred to later chapters.
- Serving implications of decode and KV cache are previewed but not taught in detail.
- RoPE and SwiGLU remain short inline examples, consistent with the sidebar plan.

## Risks Remaining

| Risk | Blocking? | Notes |
|---|---:|---|
| T5 versus decoder-only contrast could be a table. | No | Copyedit/layout issue. |
| RoPE/SwiGLU may be dense for new readers. | No | They are brief and source-backed. |
| No full attention equations. | No | Appropriate for this chapter's level. |

## Required Fixes

None.

## Red-Team Prompts

- Does the chapter imply all modern LLMs are decoder-only?
- Does it blur training parallelism with inference sequentiality?
- Does it treat attention as the entire Transformer and understate FFN cost?
- Does it make RoPE/SwiGLU sound mandatory rather than examples?

Owner: Technical Reviewer  
Purpose: Chapter 2 technical review  
Evidence grade: A for course lecture claims and primary architecture papers; no benchmark numbers used  
Assumptions: Review evaluates conceptual and architectural correctness, not final math notation depth  
Open questions: Whether to add a compact architecture comparison table in layout pass  
Handoff: Red Team reviewer for adversarial critique
