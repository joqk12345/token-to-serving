# Part I Technical Review — Revision 2

Date: 2026-07-04  
Scope: `book/chapters/01-why-llm-systems.md`, `02-tokens-probability-transformers.md`, `03-tokenization-context-decoding.md`

## Verdict

Part I is now anchored to the right primary papers and reads as book prose rather than lecture notes.

The chapters should still remain at `draft` for one more pass, but the remaining work is now editorial and structural, not evidentiary:

- tighten a few transition sentences;
- decide whether EAGLE and tokenizer-free material stay in the main body or move to sidebars;
- decide whether Chapter 1 needs a short historical model-scale sidebar;
- prepare chapter-level copyedit and figure planning.

## What Changed Since Rev 1

The following primary sources are now in the source ledger and usable as anchors:

- Vaswani et al. 2017
- Sennrich et al. 2016
- Kudo & Richardson 2018
- Raffel et al. 2020
- Touvron et al. 2023
- Su et al. 2021
- Leviathan et al. 2022
- Chen et al. 2023
- Li et al. 2024

That closes the biggest evidence gap from the first pass.

## Chapter-Level Notes

### Chapter 1

The chapter now reads cleanly and does not depend on lecture voice. It introduces the systems frame and resource thinking well.

Open risk:

- if the book later needs a numerical model-scale history, Chapter 1 will need one more source card for that sidebar.

### Chapter 2

The Transformer chapter is now properly grounded in the canonical Transformer paper, with T5, LLaMA, and RoPE anchored to primary papers. The main remaining question is presentation density:

- whether RoPE and SwiGLU should remain inline or move to a small sidebar;
- whether the T5 and decoder-only contrast stays in the main line or becomes a compact comparison table.

### Chapter 3

The tokenization and decoding chapter now has the correct core anchors:

- BPE and open-vocabulary motivation;
- SentencePiece;
- speculative decoding;
- EAGLE.

The only substantive judgment left is chapter shape:

- EAGLE may be too advanced for the main line if the chapter is meant to stay introductory-to-intermediate;
- tokenizer-free models may be better as a short sidebar than as a primary thread.

## Recommendation

Move from source-card expansion to chapter shaping.

The next action should be one of:

1. write a short sidebar plan for Chapter 2/3;
2. create figure specs for Chapters 1-3;
3. draft Chapter 4 sources and brief.

Owner: Technical Reviewer  
Purpose: Second review after adding primary-paper source cards  
Evidence grade: B  
Assumptions: The aim is publication-quality technical nonfiction with readable chapter flow  
Open questions: Sidebar placement for RoPE, SwiGLU, EAGLE, and tokenizer-free material  
Handoff: Book architect for shape/figure decisions

