# Part I Technical Review — Chapters 1-3

Date: 2026-07-04  
Scope: `book/chapters/01-why-llm-systems.md`, `02-tokens-probability-transformers.md`, `03-tokenization-context-decoding.md`

## Verdict

Part I is structurally sound as a first draft. It correctly establishes the book's progression:

```text
LLM as system workload -> Transformer computation -> tokenization/decoding interfaces
```

Do not promote these chapters to `technical-review` cleared yet. They need primary-paper source cards before the evidence bar is strong enough for publication-quality technical nonfiction.

## Findings

### Medium — Chapter 2 relies too much on lecture cards for canonical architecture claims

Chapter 2 explains embeddings, attention, masking, FFN, residuals, layer norm, T5, LLaMA, RoPE, and SwiGLU mostly through course lecture cards. That is acceptable for draft, but the final chapter should anchor canonical claims to original papers:

- Vaswani et al. 2017 for Transformer and attention.
- Raffel et al. 2020 for T5.
- Touvron et al. 2023 for LLaMA.
- Su et al. 2021 for RoPE.

Recommendation: create source-ledger cards before technical review signoff.

### Medium — Chapter 3 speculative decoding section needs a primary-source split

The current draft intentionally hedges that speculative decoding variants differ. That is good. The source card still uses only the lecture, and the lecture's acceptance-rule description is not enough to support a general explanation of speculative decoding algorithms.

Recommendation: add at least one primary speculative decoding paper card, and separate:

- general speculative decoding idea;
- top-k validation as lecture simplification or specific variant;
- EAGLE as separate advanced method.

### Medium — Chapter 3 BPE explanation should get original BPE-for-NMT source

The BPE section is clear and useful, but the final wording should be checked against Sennrich et al. 2016. SentencePiece should be added if the chapter discusses modern tokenizer practice beyond BPE mechanics.

Recommendation: add Sennrich et al. 2016 as required; Kudo & Richardson 2018 as recommended.

### Low — Chapter 1 is doing the right amount of framing

Chapter 1 avoids overclaiming and does not overexplain later details. It could use one external source only if we decide to make historical model-scale claims, such as GPT-3 token count, scaling laws, or model-size trend lines.

Recommendation: keep Chapter 1 mostly course-led unless adding a short model-scale sidebar.

### Low — Citation parser hygiene issue fixed

Several inline markers were written as `\[CITE:` rather than `[CITE:`. This was fixed in Chapter 1 and Chapter 3.

## Recommended Next Source Cards

Priority order:

1. `vaswani-2017-attention-is-all-you-need`
2. `sennrich-2016-bpe-rare-words`
3. `kudo-richardson-2018-sentencepiece`
4. `raffel-2020-t5`
5. `touvron-2023-llama`
6. `su-2021-roformer-rope`
7. `brown-2020-gpt3`
8. `leviathan-2023-speculative-decoding` or another primary speculative decoding source
9. `li-2024-eagle` / EAGLE primary paper

Update 2026-07-04: required source cards were added for Transformer, BPE, SentencePiece, T5, LLaMA, RoPE, speculative decoding, speculative sampling, EAGLE, BLT, and one Hugging Face 2026-07-03 long-context/hybrid-attention candidate. See `book/evidence/recommended-papers-part-i.md`.

## Draft Promotion Recommendation

Keep all three chapters at `draft`.

Next stage should be `source-card expansion`, not prose rewriting. After source cards are added, run a second technical review focused on:

- formulas and notation;
- architecture claims;
- tokenization algorithm wording;
- speculative decoding caveats;
- cite-card coverage.

Owner: Technical Reviewer  
Purpose: First technical review of Part I drafts  
Evidence grade: B for review judgment, based on current source ledger and course PDFs  
Assumptions: The book aims for publication-quality technical nonfiction, not course-note summary  
Open questions: Whether to browse/download missing primary papers now or only list them first  
Handoff: Researcher to create source cards from recommended papers
