# Technical Review: Chapter 3 — Tokenization, Context, and Decoding

Date: 2026-07-05  
Reviewer: Codex  
Chapter: `book/chapters/03-tokenization-context-decoding.md`

## Verdict

Cleared for red-team review.

The chapter is technically coherent at introductory-to-intermediate level. The earlier evidence gaps are closed with primary cards for BPE, SentencePiece, speculative decoding/sampling, EAGLE, and modern tokenizer context. EAGLE is now explicitly marked as an advanced sidebar rather than part of the main decoding path.

## Checks

### Tokenization

- The chapter correctly distinguishes words, characters, and subword units.
- BPE is framed as a frequent-pair merge process and anchored to Sennrich et al.
- Vocabulary size is described as a systems choice affecting sequence length and embedding/output tables.
- SentencePiece and LLaMA support the practical-tokenization discussion.

### Context and Resource Model

- Context length is correctly framed as a token budget, not a word budget.
- Training and inference consequences are separated: activation/attention cost versus prefill/KV-cache cost.
- No tokenizer fairness or multilingual quantitative claims are made without source conditions.

### Decoding

- Greedy, sampling, and beam search are presented as tradeoffs, not universal recommendations.
- Autoregressive serial dependency is correctly identified as the key serving constraint.
- Speculative decoding is described as draft-model proposal plus target-model validation, with exact target distribution support attributed to the primary paper.
- Draft length examples are treated as workload-dependent.

### Advanced Material

- EAGLE is explicitly marked as an advanced sidebar.
- Tokenizer-free models are not introduced into the main text.
- Serving architecture is deferred to Chapters 12-14.

## Risks Remaining

| Risk | Blocking? | Notes |
|---|---:|---|
| EAGLE sidebar may still be optional. | No | It is short and source-backed. |
| Tokenizer-free models omitted. | No | Matches sidebar plan. |
| Beam search details are compact. | No | Appropriate for chapter scope. |

## Required Fixes

None.

## Red-Team Prompts

- Does BPE sound linguistically meaningful rather than corpus-statistical?
- Does the chapter imply larger vocabulary is always better?
- Does speculative decoding sound guaranteed to reduce latency?
- Does EAGLE distract from the main tokenization/decoding path?
- Does context length get confused with word count?

Owner: Technical Reviewer  
Purpose: Chapter 3 technical review  
Evidence grade: A for course lecture claims and primary tokenization/decoding papers; no benchmark numbers used  
Assumptions: Review evaluates mechanism correctness and chapter scope, not final copyediting  
Open questions: Whether to keep EAGLE sidebar in final layout  
Handoff: Red Team reviewer for adversarial critique
