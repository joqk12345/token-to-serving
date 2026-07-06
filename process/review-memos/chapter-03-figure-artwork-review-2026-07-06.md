# Chapter 3 Figure Artwork Review

Date: 2026-07-06

Scope: `book/figures/artwork/ch03/*.svg`

## Verdict

The five Chapter 3 diagrams are cleared as first-pass book artwork:

- `fig-03-tokenizer-comparison`
- `fig-03-bpe-loop`
- `fig-03-decode-methods`
- `fig-03-speculative-pipeline`
- `fig-03-eagle-feature-loop`

The assets complete the Part I figure set using the established 1200 by 760 canvas, type hierarchy, line weights, and semantic palette. Each SVG includes an accessible title and description.

## Checks

| Check | Result | Notes |
| --- | --- | --- |
| XML validity | pass | All five files pass `xmllint --noout`. |
| Technical overclaim scan | pass | No benchmark, fixed speedup, or universal decoding claim was introduced. |
| Terminology | pass | Token, vocabulary, OOV, BPE, greedy, sampling, beam search, draft model, target model, and tree attention match Chapter 3. |
| Mechanism | pass | BPE pair counting, decoding selection, speculative validation, and EAGLE feature prediction remain distinct. |
| Layout | pass | Text and shapes remain within the SVG view box; rendered thumbnails confirm hierarchy and dataflow. |
| Accessibility | pass | Each file has `role="img"` plus linked `title` and `desc` elements. |

## Review Notes

- The tokenizer comparison labels its boundaries as illustrative and includes punctuation, a contraction, and a rare synthetic word.
- The BPE example uses corpus-local illustrative counts and shows vocabulary growth without implying linguistic segmentation.
- The decoding comparison presents probability values as illustrative and distinguishes deterministic, stochastic, and multi-candidate selection.
- The speculative pipeline keeps acceptance rules generic and makes draft-target agreement a condition rather than a guaranteed benefit.
- The EAGLE figure is explicitly marked as an advanced sidebar and separates feature prediction from target-model validation.

Owner: Diagram Producer  
Purpose: First-pass Chapter 3 artwork and rendered-artwork review  
Evidence grade: A for source-aligned conceptual diagrams; illustrative examples are labeled  
Assumptions: SVG is the canonical editable format and final typography may be adjusted during publication layout  
Open questions: Whether the publication pipeline requires PDF, EPS, or outlined-font exports  
Handoff: Chapter 4 diagram production using the same visual system
