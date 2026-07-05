# Source Notes — llmsys-09-decoding

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-09-decoding-cac2cd9402765ff5e6c24f7baffd321c.pdf`

## High-Level Role

This lecture supports Chapter 3 and later inference chapters. It introduces greedy decoding, sampling, beam search, speculative decoding, and EAGLE.

## Core Claims

| Claim | Evidence grade | Source card |
|---|---|---|
| Exhaustive sequence search is too expensive, with naive complexity exponential in vocabulary size and sequence length. | A | `llmsys-09-sequence-decoding-cost` |
| Greedy decoding chooses the highest-probability next token at each step. | A | `llmsys-09-greedy-decoding` |
| Beam search keeps the top-k partial sequences and expands them step by step as an approximate search. | A | `llmsys-09-beam-search` |
| Autoregressive LLM decoding is slow because tokens are generated sequentially. | A | `llmsys-09-autoregressive-decode-latency` |
| Speculative decoding uses a smaller draft model to propose tokens and a larger target model to validate them in parallel. | A | `llmsys-09-speculative-decoding` |
| EAGLE predicts final-layer features with a small model and uses tree attention for efficient speculative decoding. | A | `llmsys-09-eagle` |

## Chapter Use

Chapter 3 should introduce decoding as the bridge from model probabilities to generated text. Serving-specific latency and batching should be deferred to Chapters 12-14.

## Figure Candidates

| Figure ID | Purpose | Form |
|---|---|---|
| `fig-03-greedy-vs-beam` | Compare local choice and beam search. | Search tree |
| `fig-03-speculative-decoding` | Show draft model proposal and target validation. | Two-model pipeline |
| `fig-03-decode-sequentiality` | Show why autoregressive decode is serial. | Timeline |

## Risks / Checks

- The speculative decoding acceptance rule in the lecture uses top-k validation; other formulations use exact distribution correction. Avoid claiming a single universal acceptance algorithm without paper-specific support.
- Keep EAGLE as an advanced sidebar unless the chapter grows too long.
- Do not merge decoding quality choices with serving throughput choices too early.

Owner: Technical Researcher  
Purpose: Source extraction for Chapter 3 and serving chapters  
Evidence grade: A for course-framing claims  
Assumptions: Chapter 3 covers decoding concepts; later chapters cover serving systems  
Open questions: Which speculative decoding paper should be the primary source card  
Handoff: Book Architect for Chapter 3 brief

