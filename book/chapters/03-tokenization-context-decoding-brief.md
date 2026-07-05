---
status: brief
chapter: 3
slug: 03-tokenization-context-decoding
title: Tokenization, Context, and Decoding
primary_sources:
  - llmsys-08-tokenization-594dd043d7a87d8dcc91b7e7585a0e34.pdf
  - llmsys-09-decoding-cac2cd9402765ff5e6c24f7baffd321c.pdf
secondary_sources: []
reader_level: engineer or graduate student with basic ML background
technical_depth: introductory-to-intermediate
---

# Tokenization, Context, and Decoding — Brief

## Chapter Thesis

The interface around the Transformer matters: tokenization determines sequence length and vocabulary behavior, while decoding turns token probabilities into text under quality, diversity, and latency constraints.

## Reader Problem

The reader may think tokens are a preprocessing detail and decoding is simply "pick the next word." This chapter shows that both choices shape model behavior, system cost, and serving latency.

## System Bottleneck

Primary bottlenecks:

- sequence length created by tokenizer choices;
- embedding/output vocabulary size;
- serial autoregressive generation;
- search and sampling cost;
- validation/generation tradeoffs in speculative decoding.

## Source Map

| Claim                                                                                                                      | Source card                               | Evidence grade | Notes                                |
| -------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | -------------- | ------------------------------------ |
| Tokenization choices trade off vocabulary size, OOV behavior, sequence length, and semantic grouping.                      | `llmsys-08-tokenization-tradeoffs`        | A              | Opening anchor.                      |
| BPE iteratively merges frequent adjacent token pairs.                                                                      | `llmsys-08-bpe-algorithm`                 | A              | Main algorithm explanation.          |
| Practical tokenization includes corpus filtering, code, numbers, multilingual allocation, and tokenizer-free alternatives. | `llmsys-08-practical-tokenization`        | A              | Use selectively.                     |
| Beam search keeps top-k partial sequences.                                                                                 | `llmsys-09-beam-search`                   | A              | Decoding method anchor.              |
| Autoregressive decoding is slow because generation is sequential.                                                          | `llmsys-09-autoregressive-decode-latency` | A              | Bridge to serving chapters.          |
| Speculative decoding uses a draft model and target model validation.                                                       | `llmsys-09-speculative-decoding`          | A              | Advanced decoding acceleration hook. |

## Explanation Arc

1. Start with a sentence that tokenizes differently under word, character, and subword schemes.
2. Show why word-level tokenization creates unknown-token and vocabulary-size problems.
3. Show why character-level tokenization lengthens the sequence.
4. Introduce BPE as the compromise: frequent chunks become tokens, rare words decompose.
5. Connect tokenizer choice to embedding table size, output softmax, context length, and multilingual behavior.
6. Switch from input representation to output generation.
7. Explain greedy decoding, sampling, and beam search as different ways to use next-token probabilities.
8. Show why autoregressive decoding is serial.
9. Introduce speculative decoding as a way to validate multiple proposed tokens.
10. Close with the systems hook: serving chapters will treat decoding as a scheduler and memory-management problem.

## Required Figures

| Figure ID                     | Purpose                                                 | Form           | Source                             |
| ----------------------------- | ------------------------------------------------------- | -------------- | ---------------------------------- |
| `fig-03-tokenizer-comparison` | Compare word/char/subword tokenization on one sentence. | Table          | `llmsys-08-tokenization-tradeoffs` |
| `fig-03-bpe-loop`             | Show BPE merge loop.                                    | Step diagram   | `llmsys-08-bpe-algorithm`          |
| `fig-03-decode-methods`       | Compare greedy, sampling, beam.                         | Search diagram | `llmsys-09-beam-search`            |
| `fig-03-speculative-pipeline` | Show draft model and target model validation.           | Pipeline       | `llmsys-09-speculative-decoding`   |
| `fig-03-eagle-feature-loop`   | Show EAGLE as feature-level speculative decoding in an advanced sidebar. | Feature prediction loop | `li-2024-eagle`, `llmsys-09-speculative-decoding` |

## Technical Checks

- Formula correctness: verify BPE algorithm wording against Sennrich et al. before final publication.
- Complexity / memory accounting: avoid hard numerical claims in this chapter; use qualitative links to sequence length and vocab size.
- Hardware assumptions: none.
- Benchmark conditions: no latency numbers unless sourced with setup.
- Terminology consistency: define token, vocabulary, OOV, BPE, prefill/decode if mentioned.

## Open Questions

- Should VOLT be a main section or sidebar?
- Should speculative decoding be introduced here or saved entirely for Chapter 12?
- Should multilingual tokenizer issues get one concise subsection or a later appendix?

## Handoff

Owner: Book Architect  
Purpose: Establish Chapter 3 structure and evidence envelope  
Evidence grade: A for course-framing claims; primary papers needed for detailed BPE, SentencePiece, speculative decoding, and EAGLE variants  
Assumptions: Chapter 3 should bridge representation to generation without becoming an inference-serving chapter  
Open questions: Need decision on how much speculative decoding belongs here  
Handoff: Systems Explainer for draft after algorithm wording checks
