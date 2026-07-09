# Book Production Status

Project: **Large Language Model Systems: From Tokens to Serving Infrastructure**

This file is the resumability ledger. Update it after each chapter stage clears.

**Last updated:** 2026-07-09

## Current Focus

- Build source ledger from `downloads/llmsystem2026spring/source_pdfs/`.
- Confirm the spine in `book/spine.yml`.
- Book-level consistency audit completed after all 15 chapters reached ready.
- Figure production is complete. All 15 chapters have eighty-eight artwork-reviewed editable SVG diagrams.
- Reference-generation pipeline is complete: 219 cited source cards resolve to 63 deduplicated works in `book/REFERENCES.md`; workspace validation checks citation resolution and generated-file freshness.
- Index-generation pipeline is complete: `book/INDEX.md` is derived from chapter headings and the terminology registry, and workspace validation checks that it stays current.
- Book-export pipeline is complete: `book/export/book-source.md` and `book/export/book.html` are assembled with pandoc, include the references and index, and pass freshness checks.
- Final copyedit pass is complete: chapter terminology, citation resolution, handoff blocks, unresolved markers, duplicate words, and mechanical whitespace checks pass.
- Next focus: publisher-specific citation styling or adding a PDF/print backend if one becomes available.

## Chapter Status

Legend: `-` not started, `>` in progress, `x` cleared, `!` gated.

| #  | Slug                                         | Sources | Brief | Draft | Technical Review | Red Team | State  |
| -- | -------------------------------------------- | ------: | :---: | :---: | :--------------: | :------: | ------ |
| 1  | `01-why-llm-systems`                         |       5 |   x   |   x   |         x        |     x    | ready |
| 2  | `02-tokens-probability-transformers`         |      11 |   x   |   x   |         x        |     x    | ready |
| 3  | `03-tokenization-context-decoding`           |      12 |   x   |   x   |         x        |     x    | ready |
| 4  | `04-gpu-programming-model`                   |      11 |   x   |   x   |         x        |     x    | ready |
| 5  | `05-kernels-memory-transformer-blocks`       |      13 |   x   |   x   |         x        |     x    | ready |
| 6  | `06-flashattention-transformer-acceleration` |      11 |   x   |   x   |         x        |     x    | ready |
| 7  | `07-dl-frameworks-and-compilers`             |      25 |   x   |   x   |         x        |     x    | ready |
| 8  | `08-distributed-training-ddp`                |      14 |   x   |   x   |         x        |     x    | ready |
| 9  | `09-model-parallelism`                       |      17 |   x   |   x   |         x        |     x    | ready |
| 10 | `10-zero-moe-and-memory`                     |      20 |   x   |   x   |         x        |     x    | ready |
| 11 | `11-quantization-and-peft`                   |      21 |   x   |   x   |         x        |     x    | ready |
| 12 | `12-inference-cost-model`                    |      17 |   x   |   x   |         x        |     x    | ready |
| 13 | `13-kv-cache-vllm-pagedattention`            |      17 |   x   |   x   |         x        |     x    | ready |
| 14 | `14-serving-scheduling-and-disaggregation`   |      27 |   x   |   x   |         x        |     x    | ready |
| 15 | `15-llm-system-codesign`                     |      23 |   x   |   x   |         x        |     x    | ready |

## Open Technical Questions

| Question                                                                                  | Chapter | Blocking? | Owner              | Notes                                                                                      |
| ----------------------------------------------------------------------------------------- | ------- | --------: | ------------------ | ------------------------------------------------------------------------------------------ |
| Should runnable CUDA examples live in an appendix or examples directory?                  | 4-6     |        no | Principal author   | Chapter 4 draft uses minimal snippets; runnable code can be added later if needed.         |
| What opening example should Chapter 1 use?                                                | 1       |        no | Principal author   | Resolved for current draft: use a user prompt that unfolds into the infrastructure stack. |
| Should JAX/XLA/TPU be a standalone chapter or part of frameworks/compilers?               | 7       |        no | Book architect     | Current spine folds it into Ch. 7.                                                         |
| Would narrower JAXpr, Pallas BlockSpec, and TPU backend docs/source cards improve later polish? | 7       |        no | Technical researcher | Chapter 7 is ready under current claim scope; add narrow cards only if later revisions add precise internals. |
| Should Chapter 8 include ring all-reduce communication-volume formulas? | 8       |        no | Technical reviewer | Official/Paper cards are added; byte-count formulas should be added only with careful per-rank/topology conditions. |
| Should Chapter 9 include pipeline bubble or memory formulas? | 9       |        no | Technical reviewer | Primary paper/docs cards are added; formulas should be included only with explicit assumptions. |
| How much paper reproduction detail should be included for vLLM?                           | 13      |        no | Technical reviewer | vLLM/PagedAttention source-ledger pass is complete; add experiment-specific cards only if benchmark numbers are used. |
| Should Chapter 10 include ZeRO memory formulas or leave them as a boxed derivation?        | 10      |        no | Technical reviewer | Formula cards are added; include only with `N`, `M`, `K`, precision, and exclusion assumptions. |

## Terminology Decisions

| Term           | Decision                                                                                     | Notes                              |
| -------------- | -------------------------------------------------------------------------------------------- | ---------------------------------- |
| LLM systems    | Use for the full stack: model objective, kernels, training, serving, and deployment runtime. | Avoid reducing it to serving only. |
| token          | Define once as model-side discrete unit, then distinguish from word/byte/subword.            | Ch. 2-3.                           |
| KV cache       | Use uppercase KV; define key/value tensors at first use.                                     | Ch. 12-14.                         |
| prefill/decode | Use lowercase technical terms after first definition.                                        | Ch. 12-14.                         |
