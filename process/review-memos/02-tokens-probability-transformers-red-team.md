# Red Team: Chapter 2 — From Next-Token Probability to Transformer Computation

Date: 2026-07-05  
Reviewer: Codex  
Chapter: `book/chapters/02-tokens-probability-transformers.md`

## Verdict

Cleared for ready promotion.

The chapter explains Transformer computation at the right level for Part I. It is anchored to primary architecture papers and avoids kernel-level overreach.

## Attacks and Outcomes

### Attack 1: The chapter may imply all modern LLMs are decoder-only.

Outcome: Addressed.

The draft says decoder-only models are the book's main reasoning case and dominant serving shape, while explicitly acknowledging encoder-only and encoder-decoder families.

### Attack 2: Training parallelism may be confused with inference sequentiality.

Outcome: Addressed.

The draft states that training can process many masked positions in parallel, while interactive generation usually advances step by step.

### Attack 3: Attention may be over-centered relative to FFN.

Outcome: Addressed.

The draft says a chapter that treats attention as the entire Transformer will mislead the reader and notes that FFN layers are computationally heavy.

### Attack 4: RoPE and SwiGLU may sound mandatory.

Outcome: Addressed.

Both are presented as modern examples from LLaMA/RoFormer, not as universal Transformer requirements.

## Required Fixes

None.

## Recommended Non-Blocking Follow-Up

- Consider a compact table contrasting encoder-only, encoder-decoder, and decoder-only architectures.
- Add figures for Q/K/V attention, causal mask, and Transformer block as systems object.

Owner: Red Team Reviewer  
Purpose: Chapter 2 adversarial review  
Evidence grade: A for reviewed source map; no benchmark numbers used  
Assumptions: Red-team review evaluates conceptual correctness and scope, not final figure production  
Open questions: None blocking  
Handoff: Principal author may promote Chapter 2 to ready
