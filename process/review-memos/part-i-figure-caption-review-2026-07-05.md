# Part I Figure Caption Review

Date: 2026-07-05

Scope: `book/figures/part-i-figure-specs.md`

## Verdict

Part I figure specs are ready for first-pass diagram production after the 2026-07-05 reconciliation pass. The captions below are conceptual, source-aligned, and avoid unsupported benchmark or performance claims.

Do not add model-accuracy, latency, token-count, or speedup values to Part I captions unless a figure-specific source card and condition block are added.

## Chapter 1 Captions

### `fig-01-visible-vs-hidden-stack`

Caption: A user sees a prompt and response, but the system path underneath includes tokenization, embedding lookup, Transformer computation, runtime scheduling, memory movement, and accelerator execution.

Source anchors: `llmsys-01-system-challenges`, `llmsys-01-next-token-probability`

Risk: Keep this as an orientation stack. Do not imply every deployment exposes exactly the same layer boundaries.

### `fig-01-token-probability-chain`

Caption: A language model assigns sequence probability by multiplying conditional next-token probabilities, where each token is predicted from the previous context.

Source anchors: `llmsys-01-next-token-probability`, `llmsys-01-training-objective`

Risk: Avoid using words and tokens interchangeably. If the example sentence is displayed as text, show token boundaries explicitly.

### `fig-01-abstraction-levels`

Caption: LLM systems can be viewed as an abstraction ladder: product behavior depends on frameworks, operators, kernels, runtimes, and hardware, and the active bottleneck may appear at any layer.

Source anchors: `llmsys-01-system-challenges`

Risk: Do not turn this into an organization chart. The figure is about engineering abstraction and bottleneck location.

### `fig-01-codesign-loop`

Caption: Model architecture, algorithms, software systems, and hardware form a feedback loop: changes in one layer can expose or relieve bottlenecks in another.

Source anchors: `llmsys-01-codesign`

Risk: Keep the loop symmetric. Do not imply hardware, algorithms, or models always dominate the design process.

## Chapter 2 Captions

### `fig-02-token-to-logits`

Caption: Token IDs are mapped to embeddings, transformed by repeated Transformer blocks, and projected to logits over the model vocabulary.

Source anchors: `vaswani-2017-attention-is-all-you-need`, `llmsys-06-transformer-components`

Risk: Keep this as the decoder-side forward path used in the chapter. If encoder or encoder-decoder variants appear, separate them from this main path.

### `fig-02-qkv-attention`

Caption: Self-attention projects input vectors into queries, keys, and values; query-key scores become weights that mix value vectors into an output representation.

Source anchors: `vaswani-2017-attention-is-all-you-need`, `llmsys-06-transformer-components`

Risk: Use one attention head in the main artwork. Multi-head structure can be noted without overcrowding the figure.

### `fig-02-causal-mask`

Caption: A causal mask prevents each position from attending to future positions, preserving the left-to-right dependency used for autoregressive generation.

Source anchors: `vaswani-2017-attention-is-all-you-need`, `llmsys-06-masked-self-attention`

Risk: Do not imply masking is the only distinction between all model families; this is specifically the decoder/autoregressive mechanism.

### `fig-02-model-family-map`

Caption: Encoder-only, encoder-decoder, and decoder-only models differ in how they condition on input tokens and produce output tokens.

Source anchors: `llmsys-01-decoder-only`, `llmsys-07-t5-text-to-text`, `llmsys-07-llama-architecture`

Risk: Keep examples secondary. This figure is a model-family map, not a product or model leaderboard.

## Chapter 3 Captions

### `fig-03-tokenizer-comparison`

Caption: Word, character, and subword tokenization split the same text differently, trading off vocabulary size, sequence length, and handling of rare or unseen strings.

Source anchors: `sennrich-2016-bpe-rare-words`, `llmsys-08-tokenization-tradeoffs`

Risk: Use one illustrative sentence. Do not infer universal token counts from the example.

### `fig-03-bpe-loop`

Caption: Byte-pair encoding repeatedly merges frequent adjacent symbols, growing a vocabulary of reusable subword units.

Source anchors: `sennrich-2016-bpe-rare-words`, `llmsys-08-bpe-algorithm`

Risk: Keep the merge sequence small. The figure should show the mechanism, not a production tokenizer training run.

### `fig-03-decode-methods`

Caption: Greedy decoding, sampling, and beam search use the same next-token distribution differently: selecting the highest-probability token, drawing stochastically, or tracking multiple partial sequences.

Source anchors: `llmsys-09-beam-search`, `llmsys-09-autoregressive-decode-latency`

Risk: Avoid quality claims such as "better" or "more creative" without conditions. The figure should show search behavior.

### `fig-03-speculative-pipeline`

Caption: Speculative decoding uses a draft model to propose tokens and a target model to validate them, changing the shape of autoregressive work while preserving target-model validation.

Source anchors: `leviathan-2022-speculative-decoding`, `chen-2023-speculative-sampling`

Risk: Do not claim a universal speedup. Any performance statement needs model, hardware, sequence, and acceptance-rate context.

### `fig-03-eagle-feature-loop`

Caption: EAGLE extends speculative decoding by predicting features that help propose candidate continuations before target-model validation.

Source anchors: `li-2024-eagle`, `llmsys-09-speculative-decoding`

Risk: Mark this as an advanced sidebar. Do not make it the primary decoding mechanism for the chapter.

## Remaining Review Notes

- Captions are cleared for draft diagrams.
- If rendered artwork includes equations, rerun formula review on the exact text.
- If rendered artwork includes a tokenizer example, verify that all displayed token boundaries are illustrative and not presented as tokenizer-universal.
- Keep Part I visual language aligned with Chapter 15 recap figures.

Owner: Technical Reviewer  
Purpose: Part I figure-caption review  
Evidence grade: A for review process; captions inherit their source anchors  
Assumptions: Part I figures are conceptual orientation diagrams and not benchmark artifacts  
Open questions: Whether `fig-03-eagle-feature-loop` should be visually labeled as "advanced sidebar" in the figure title or only in surrounding prose  
Handoff: Illustrator / diagram producer
