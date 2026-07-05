# Red Team: Chapter 3 — Tokenization, Context, and Decoding

Date: 2026-07-05  
Reviewer: Codex  
Chapter: `book/chapters/03-tokenization-context-decoding.md`

## Verdict

Cleared for ready promotion.

The chapter keeps tokenization and decoding at the boundary of the Transformer and maintains the systems thread: sequence length, vocabulary size, context budget, decode seriality, and latency tradeoffs.

## Attacks and Outcomes

### Attack 1: BPE may sound linguistically meaningful.

Outcome: Addressed.

The draft explicitly says BPE tokens are artifacts of corpus statistics and merge rules, not guaranteed linguistic units.

### Attack 2: Larger vocabulary may sound always better.

Outcome: Addressed.

The draft describes vocabulary size as a tradeoff: larger vocabularies can reduce sequence length but increase embedding/output structures.

### Attack 3: Speculative decoding may sound guaranteed to reduce latency.

Outcome: Addressed.

The draft says benefit depends on draft/target alignment and that rejected draft work may not help.

### Attack 4: EAGLE may distract from the main path.

Outcome: Addressed.

EAGLE is now labeled as an advanced sidebar and is kept short.

### Attack 5: Context length may be confused with word count.

Outcome: Addressed.

The draft states that context length is a token budget and the same paragraph can cost different token counts across tokenizers/languages/domains.

## Required Fixes

None.

## Recommended Non-Blocking Follow-Up

- Decide in final layout whether to keep or cut the EAGLE sidebar.
- Add figures for BPE merge intuition, context budget, and autoregressive decode timeline.

Owner: Red Team Reviewer  
Purpose: Chapter 3 adversarial review  
Evidence grade: A for reviewed source map; no benchmark numbers used  
Assumptions: Red-team review evaluates mechanism-level correctness and scope, not final copyediting  
Open questions: None blocking  
Handoff: Principal author may promote Chapter 3 to ready
