# Part I Figure Specs

Scope: Chapters 1-3

These specs are written for the first-pass book illustrations. The goal is not decoration. Each figure must carry a reader-level idea that the prose can refer back to cleanly.

## Chapter 1

### `fig-01-visible-vs-hidden-stack`

- Chapter: 1
- Purpose: Show the gap between user-visible LLM behavior and the hidden system stack.
- Core message: A prompt/response interface sits on top of tokenization, embeddings, Transformer blocks, runtime scheduling, and memory movement.
- Visual form: Vertical stack diagram.
- Layout: Top band for user tasks, middle bands for token/model/runtime layers, bottom band for hardware and memory.
- Labels: translation, summarization, code, reasoning, tokenization, embeddings, Transformer blocks, batching, KV cache, GPU memory, interconnect.
- Source anchors: `llmsys-01-system-challenges`, `llmsys-01-next-token-probability`
- Production note: Keep the stack strictly ordered from visible to hidden. Do not use decorative arrows that obscure the bottleneck path.

### `fig-01-token-probability-chain`

- Chapter: 1
- Purpose: Show next-token probability as a chain rule over a sequence.
- Core message: One sentence probability factors into conditional next-token probabilities.
- Visual form: Token chain / left-to-right probability flow.
- Layout: A short sentence with tokens above a product expansion or linked arrows.
- Labels: prompt, previous tokens, next token, conditional probability, sequence probability.
- Source anchors: `llmsys-01-next-token-probability`, `llmsys-01-training-objective`
- Production note: Keep it short and mathematically honest. This figure should explain the factorization, not prove it.

### `fig-01-abstraction-levels`

- Chapter: 1
- Purpose: Show the abstraction ladder from model behavior down to kernels and hardware.
- Core message: Better abstractions hide some complexity but expose the active bottleneck.
- Visual form: Layered architecture diagram.
- Layout: Upper level application/product, middle level training/inference framework, lower level operators/kernels, bottom hardware.
- Labels: product integration, training runtime, inference runtime, operator library, kernel, GPU/TPU.
- Source anchors: `llmsys-01-system-challenges`
- Production note: Distinguish abstraction layers from deployment stack. This is about engineering responsibility, not org chart.

### `fig-01-codesign-loop`

- Chapter: 1
- Purpose: Show model-algorithm-system co-design as a feedback loop.
- Core message: Architecture, algorithm, software, and hardware change each other.
- Visual form: Four-node loop or square with arrows.
- Layout: Model architecture, training/inference algorithm, software runtime, hardware acceleration.
- Labels: model, algorithm, software, hardware, memory, communication, latency.
- Source anchors: `llmsys-01-codesign`
- Production note: Keep the loop symmetric. Do not imply one node is always primary.

## Chapter 2

### `fig-02-token-to-logits`

- Chapter: 2
- Purpose: Show the basic Transformer forward path from tokens to logits.
- Core message: Token IDs become embeddings, pass through repeated blocks, and produce a vocabulary distribution.
- Visual form: Dataflow diagram.
- Layout: token IDs -> embedding table -> repeated Transformer block -> final linear head -> logits.
- Labels: token IDs, embeddings, Transformer block, logits, vocabulary.
- Source anchors: `vaswani-2017-attention-is-all-you-need`, `llmsys-06-transformer-components`
- Production note: Make the final head explicit. Readers often lose the connection between hidden state and vocabulary scores.

### `fig-02-qkv-attention`

- Chapter: 2
- Purpose: Show how query, key, and value projections produce attention weights and mixed output.
- Core message: Attention computes a weighted mixture of value vectors using query-key similarity.
- Visual form: Dataflow plus small matrix.
- Layout: input vectors -> Q/K/V projections -> score matrix -> softmax -> weighted sum -> output.
- Labels: Q, K, V, score matrix, softmax, weighted sum, output.
- Source anchors: `vaswani-2017-attention-is-all-you-need`, `llmsys-06-transformer-components`
- Production note: Keep the math readable. Use one head in the main figure, not all heads at once.

### `fig-02-causal-mask`

- Chapter: 2
- Purpose: Show the decoder mask that blocks access to future tokens.
- Core message: Autoregressive decoding depends on causal masking during training and stepwise generation at inference.
- Visual form: Attention matrix with upper triangle masked.
- Layout: square matrix with diagonal and lower triangle visible, upper triangle shaded/blocked.
- Labels: allowed attention, masked future positions, token positions.
- Source anchors: `vaswani-2017-attention-is-all-you-need`, `llmsys-06-masked-self-attention`
- Production note: The figure should visually explain why the decoder cannot look right. Do not over-label it with equation clutter.

### `fig-02-model-family-map`

- Chapter: 2
- Purpose: Compare encoder-only, encoder-decoder, and decoder-only model families.
- Core message: The three families differ in how they condition and generate sequences.
- Visual form: Three-column taxonomy table or side-by-side schematic.
- Layout: encoder-only, encoder-decoder, decoder-only.
- Labels: masked LM, conditional generation, autoregressive generation, examples.
- Source anchors: `llmsys-01-decoder-only`, `llmsys-07-t5-text-to-text`, `llmsys-07-llama-architecture`
- Production note: This figure should be compact and conceptual, not a model zoo.

## Chapter 3

### `fig-03-tokenizer-comparison`

- Chapter: 3
- Purpose: Compare word-level, character-level, and subword tokenization on one sentence.
- Core message: Tokenization choices trade off vocabulary size, sequence length, and OOV behavior.
- Visual form: Side-by-side table or aligned token strips.
- Layout: one example sentence rendered three ways.
- Labels: word, character, subword, OOV, longer sequence, smaller vocab.
- Source anchors: `sennrich-2016-bpe-rare-words`, `llmsys-08-tokenization-tradeoffs`
- Production note: Use a sentence that contains punctuation, contraction, and at least one rare word.

### `fig-03-bpe-loop`

- Chapter: 3
- Purpose: Show the iterative BPE merge process.
- Core message: Frequent adjacent pairs are merged repeatedly until the vocabulary reaches target size.
- Visual form: Step diagram.
- Layout: start from characters, show pair counts, highlight a merge, repeat into a final token.
- Labels: characters, frequent pair, merge, vocabulary growth.
- Source anchors: `sennrich-2016-bpe-rare-words`, `llmsys-08-bpe-algorithm`
- Production note: Keep the example tiny. Four to six merges is enough.

### `fig-03-decode-methods`

- Chapter: 3
- Purpose: Compare greedy, sampling, and beam search.
- Core message: Decoding is a search/selection strategy, not just a post-processing step.
- Visual form: Three-panel search diagram.
- Layout: one prompt, then three panels for greedy, sampling, beam.
- Labels: highest probability, stochastic choice, top-k partial sequences.
- Source anchors: `llmsys-09-beam-search`, `llmsys-09-autoregressive-decode-latency`
- Production note: Make the tradeoff visible. Greedy should look deterministic, beam should look broader, sampling should look variable.

### `fig-03-speculative-pipeline`

- Chapter: 3
- Purpose: Show draft model proposal followed by target model validation.
- Core message: Speculative decoding changes the work shape by validating multiple proposed tokens at once.
- Visual form: Two-stage pipeline with validation loop.
- Layout: draft model on the left, target model on the right, accept/reject branch back to the next step.
- Labels: draft model, proposed tokens, target validation, accepted tokens, rejected tokens.
- Source anchors: `leviathan-2022-speculative-decoding`, `chen-2023-speculative-sampling`
- Production note: Keep it generic enough to cover exact speculative decoding and speculative sampling without claiming one algorithm is universal.

### `fig-03-eagle-feature-loop`

- Chapter: 3
- Purpose: Show EAGLE as feature-level speculative decoding.
- Core message: EAGLE predicts internal features instead of only next-token candidates.
- Visual form: Feature prediction loop or internal-state pipeline.
- Layout: input tokens -> embedding -> small predictor -> predicted features -> target model validation.
- Labels: embedding, feature prediction, target model, tree attention, validation.
- Source anchors: `li-2024-eagle`, `llmsys-09-speculative-decoding`
- Production note: Treat this as an advanced sidebar figure. Do not make it the chapter's primary decoding figure.

## Cross-Chapter Notes

- Use the same type style and line weight across all Part I figures.
- Prefer one conceptual figure per section, not one figure per paragraph.
- Keep equations inside the prose unless the figure is explicitly mathematical.
- Avoid ornamental icons and stock-like scenes.
- If a figure introduces a term, the surrounding prose must define the term once.

Owner: Book Architect  
Purpose: Part I figure planning  
Evidence grade: A for structural decisions; figure content must be checked against the source cards before drawing  
Assumptions: Figures are intended for a technical nonfiction audience, not a slide deck audience  
Open questions: Whether Part I captions should be reviewed before or after Part III/IV figure specs are drafted  
Handoff: Writer / illustrator for first draft diagrams
