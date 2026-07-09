# Chapter 11 Figure Artwork Review

Date: 2026-07-08

Scope: `book/figures/artwork/ch11/*.svg`

## Verdict

The seven Chapter 11 diagrams are cleared as first-pass book artwork:

- `fig-11-quantization-map`
- `fig-11-quantization-error`
- `fig-11-llmint8-outlier-path`
- `fig-11-gptq-compensation`
- `fig-11-full-ft-vs-lora-state`
- `fig-11-lora-update`
- `fig-11-qlora-stack`

The assets complete the Part III figure set using the established 1200 by 760 canvas, type hierarchy, line weights, and semantic palette. Each SVG includes an accessible title and description.

## Checks

| Check | Result | Notes |
| --- | --- | --- |
| XML validity | pass | All seven files pass `xmllint --noout`. |
| Technical overclaim scan | pass | No accuracy, memory-saving, throughput, or universal low-bit speed claim was introduced. |
| Terminology | pass | Scale, zero point, clipping, outlier, GPTQ, LoRA, rank, adapter, and QLoRA match Chapter 11. |
| Formula | pass | The LoRA figure uses Chapter 11 orientation `W′ = W₀ + A B`; GPTQ avoids unreviewed Hessian formulas. |
| Scope boundaries | pass | Zero point is optional, LLM.int8 is method-specific, and QLoRA precision choices are implementation-dependent. |
| Layout | pass | Quick Look rendered all seven SVGs without parser or thumbnail failures. |
| Accessibility | pass | Each file has `role="img"` plus linked `title` and `desc` elements. |

## Review Notes

- The quantization map separates integer codes from scale metadata and optional zero point.
- The error figure keeps rounding, clipping, and range mismatch qualitative.
- The LLM.int8 diagram explicitly limits mixed outlier routing to LLM.int8-style execution.
- GPTQ shows blockwise compensation without importing an unreviewed second-order formula.
- The state ledger retains frozen base weights and activations in the LoRA path.
- The LoRA factor diagram preserves the chapter's matrix orientation and low-rank bottleneck.
- The QLoRA stack separates compact frozen-base storage from trainable adapter state without fixing bitwidth or optimizer choices.

Owner: Diagram Producer  
Purpose: First-pass Chapter 11 artwork and rendered-artwork review  
Evidence grade: A for source-aligned conceptual diagrams; no benchmark claims  
Assumptions: SVG is the canonical editable format and final typography may be adjusted during publication layout  
Open questions: Whether a later revision adds a deeper NF4, double-quantization, and paged-optimizer figure set  
Handoff: Chapter 12 diagram production using the same visual system
