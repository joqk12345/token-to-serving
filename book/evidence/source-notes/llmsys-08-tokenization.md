# Source Notes — llmsys-08-tokenization

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-08-tokenization-594dd043d7a87d8dcc91b7e7585a0e34.pdf`

## High-Level Role

This lecture supports Chapter 3. It explains why tokenization is a systems and modeling choice, not just preprocessing.

## Core Claims

| Claim | Evidence grade | Source card |
|---|---|---|
| Tokenization splits text into model units that index an embedding table. | A | `llmsys-08-tokenization-definition` |
| Word-level tokenization is simple but creates OOV and vocabulary-size tradeoffs. | A | `llmsys-08-word-tokenization-tradeoff` |
| Character-level tokenization avoids OOV but increases sequence length and weakens semantic grouping. | A | `llmsys-08-character-tokenization-tradeoff` |
| BPE starts from characters and iteratively merges frequent token pairs until a target vocabulary size is reached. | A | `llmsys-08-bpe-algorithm` |
| Practical LLM tokenization must handle deduplication/filtering, code, numbers, multilingual text, and vocabulary sharing. | A | `llmsys-08-practical-tokenization` |
| Tokenizer-free approaches such as Byte Latent Transformer replace fixed tokenization with byte/patch modeling. | A | `llmsys-08-tokenizer-free` |

## Chapter Use

Chapter 3 should connect tokenization to sequence length, embedding size, multilingual behavior, and inference cost. It should avoid treating tokenizer choice as a mere NLP preprocessing detail.

## Figure Candidates

| Figure ID | Purpose | Form |
|---|---|---|
| `fig-03-tokenization-tradeoff` | Compare word, character, and subword tokenization. | Table + small example |
| `fig-03-bpe-merge-loop` | Show iterative BPE merging. | Step diagram |
| `fig-03-vocab-seqlen-memory` | Show vocabulary size vs sequence length vs embedding/output cost. | Tradeoff triangle |

## Risks / Checks

- Verify LLaMA 3.1 vocabulary numbers before using them as fact; the slide cites external web material.
- Treat VOLT as a research method, not standard LLM practice unless separately supported.
- Do not imply BPE tokens are always semantically meaningful.

Owner: Technical Researcher  
Purpose: Source extraction for Chapter 3  
Evidence grade: A for course-framing claims  
Assumptions: Chapter 3 will connect tokenizer decisions to system costs  
Open questions: Whether to include VOLT as main text or sidebar  
Handoff: Book Architect for Chapter 3 brief

