---
status: ready
chapter: 2
slug: 02-tokens-probability-transformers
title: From Next-Token Probability to Transformer Computation
primary_sources:
  - llmsys-06-transformer-14bd7575a2f6c8bac60522354c11d691.pdf
  - llmsys-07-llms-acf5db9438a8d9a86f86d29d9c563c00.pdf
secondary_sources: []
reader_level: engineer or graduate student with basic ML background
technical_depth: introductory-to-intermediate
---

# From Next-Token Probability to Transformer Computation

The previous chapter reduced language modeling to a simple contract:

```text
P(next token | previous tokens)
```

This chapter asks what that contract becomes when implemented.

The short answer is: a repeated tensor program. Tokens are converted to vectors. Those vectors pass through layers that mix information across positions, transform each position independently, normalize intermediate values, and finally produce logits over the vocabulary. During training, the program is run over many examples and optimized with cross-entropy. During inference, the same program is run repeatedly to extend a sequence one token at a time.

The Transformer is the dominant way to organize that computation. The original paper describes it as an attention-based encoder-decoder architecture that removes recurrence and convolution from the core sequence model, which is why it became the canonical baseline for later LLM systems. [CITE: vaswani-2017-attention-is-all-you-need]

## From Sequence Probability to Computation

The Transformer paper begins from sequence-to-sequence learning. In an encoder-decoder system, the model estimates the probability of a target sequence conditioned on a source sequence:

```text
p_theta(y | x) = product over t of p(y_t | x, y_1:t-1; theta)
```

That factorization says the decoder predicts each target token from the source and the previous target tokens. [CITE: llmsys-06-teacher-forcing]

Decoder-only LLMs remove the separate source/target split. The conditioning context is simply the prefix: prompt tokens plus already generated tokens. The same autoregressive idea remains. The model computes a distribution for the next token.

The implementation problem is to turn the prefix into a useful representation before scoring the vocabulary.

## Tokens Become Vectors

A neural network cannot directly multiply the string `"Pittsburgh"` or the token ID `19437`. It needs vectors.

The first step is an embedding lookup. Each token ID indexes a learned row in an embedding table. If the model dimension is `d_model`, each token becomes a vector of length `d_model`.

At this point, the model knows which token appears at each position, but not enough about where it appears. The same token may mean different things at different positions. Transformer models therefore add some form of positional information to token embeddings. The original Transformer used sinusoidal positional encodings; modern LLMs often use relative-position mechanisms such as Rotary Positional Embeddings. The LLaMA paper uses RoPE in its decoder-only architecture, and the RoFormer paper gives the primary RoPE formulation. [CITE: llmsys-07-llama-architecture; su-2021-roformer-rope]

The exact positional method matters, but the system-level point is simpler: after tokenization and embedding, a sequence has become a matrix.

```text
sequence length x model dimension
```

Most of the rest of the model repeatedly transforms this matrix.

## Attention Mixes Positions

A token representation by itself is not enough. The representation for `"bridges"` should depend on whether the prefix is about Pittsburgh, dentistry, networking, or a card game.

Attention is the mechanism that lets each position read from other positions.

For each input vector, the model computes three learned projections:

- query, or `Q`;
- key, or `K`;
- value, or `V`.

The query asks what the current position is looking for. The keys describe what other positions offer. The values carry the information to be mixed. In scaled dot-product attention, query-key scores are normalized with softmax and used as weights over values.

Multi-head attention splits representation space into multiple heads, applies attention per head, concatenates the results, and projects them back. The original Transformer paper is the primary source for that block structure. [CITE: llmsys-06-transformer-components; vaswani-2017-attention-is-all-you-need]

This is not just a modeling trick. It defines the model's central workload. Attention creates matrix multiplications, softmax operations, and memory reads/writes whose cost depends on sequence length, number of heads, and hidden dimension. Later chapters will return to this cost when discussing GPU kernels and FlashAttention.

## The Decoder Cannot Look Right

For an autoregressive decoder, attention needs a rule: position `t` may attend to positions `<= t`, but not to future positions.

Decoder self-attention masks the right side before softmax by assigning those positions negative infinity. The original Transformer paper is the primary anchor for causal masking in the decoder. [CITE: llmsys-06-masked-self-attention; vaswani-2017-attention-is-all-you-need]

The mask enforces the probability factorization. If the model is training on:

```text
Pittsburgh is a city of bridges
```

then the representation used to predict `"bridges"` may see `"Pittsburgh is a city of"`, but the representation used to predict `"city"` must not see `"of bridges"`.

This distinction is easy to miss because training can process many positions in parallel under the mask. The mask blocks information, not computation. During inference, however, the next token is not known yet, so generation usually advances step by step. That gap between training parallelism and inference sequentiality will become central in the serving chapters.

## Feed-Forward Layers Transform Each Position

Attention mixes information across positions. The feed-forward network transforms each position's representation.

The Transformer block combines multi-head attention, feed-forward network, residual connections, and layer normalization. The original paper is the architectural baseline; the source cards are the systems-friendly explanation. [CITE: llmsys-06-transformer-components; vaswani-2017-attention-is-all-you-need]

Conceptually, the feed-forward part is a small neural network applied to each position. In many Transformer models, it expands the hidden dimension, applies a nonlinearity, and projects back. Older descriptions often use ReLU. Modern LLMs commonly use gated activations. The LLaMA paper uses SwiGLU as one of its architectural improvements. [CITE: llmsys-07-llama-architecture; touvron-2023-llama]

For systems work, the feed-forward network matters because it is often a large share of the model's arithmetic. A chapter that treats attention as the entire Transformer will mislead the reader. Attention is structurally distinctive; FFN layers are computationally heavy.

## Residuals and Normalization Keep the Stack Trainable

Large Transformers are deep stacks of repeated blocks. Each block modifies the representation, but it also passes information forward through residual connections. Layer normalization stabilizes intermediate activations.

Residual connection and layer normalization are core Transformer components, and the architecture distinguishes post-norm and pre-norm forms. [CITE: llmsys-06-transformer-components]

Modern decoder-only LLMs commonly use pre-normalization. The LLaMA paper is the primary anchor for that decoder-only variant. [CITE: llmsys-07-llama-architecture; touvron-2023-llama]

This is another example of a recurring pattern in LLM systems: a modeling choice changes the engineering envelope. Normalization placement affects training stability. Training stability affects how deep and large a model can be trained. How large the model becomes affects memory, parallelism, and serving cost.

## Training Uses Known Prefixes

During supervised sequence training, the decoder can be given the ground-truth prefix. [CITE: llmsys-06-teacher-forcing]

For a target sequence:

```text
I like singing and dancing
```

the model can be trained to predict each token using the true earlier tokens, not its own sampled mistakes. The loss sums the log-probability errors over positions.

Decoder-only language-model pretraining uses the same basic idea over raw text: predict the next token at each position from the previous tokens. Chapter 1 introduced this as cross-entropy next-token prediction. [CITE: llmsys-01-training-objective]

The system implication is useful: training can evaluate many token positions in one forward/backward pass. That is why training workloads are usually shaped around batches of sequences and large tensor programs. Inference for interactive generation has a different shape: many small serial extensions of active sequences.

## Encoder-Decoder, Decoder-Only, and Why the Difference Matters

The Transformer family includes more than one architecture.

Encoder-decoder models read an input sequence with an encoder and generate an output sequence with a decoder. T5 uses a standard encoder-decoder Transformer using a unified text-to-text format, and the T5 paper is the primary source for that formulation. [CITE: llmsys-07-t5-text-to-text; raffel-2020-t5]

Decoder-only models use a causal Transformer decoder to model a single sequence from left to right. [CITE: llmsys-01-decoder-only]

For the rest of this book, that architectural distinction controls many system choices.

Encoder-decoder models naturally separate source encoding from target decoding. Decoder-only LLMs combine prompt and generated text into one growing context. In serving, that means the system must manage a context that expands token by token and reuse previous attention state efficiently. That reused state is the KV cache, which becomes a first-class memory object in later chapters.

## A Transformer Block as a Systems Object

It is tempting to describe a Transformer block only as a diagram in a paper:

```text
embedding -> attention -> add/norm -> FFN -> add/norm -> logits
```

For this book, that is only the first layer of understanding. A systems engineer sees additional questions:

- Which matrix multiplications dominate arithmetic?
- Which tensors dominate memory capacity?
- Which operations are bandwidth-bound?
- Which intermediate values must be saved for backward pass?
- Which tensors can be recomputed?
- Which work can be parallelized across devices?
- Which state must persist during decoding?

The Transformer is not merely an architecture. It is a workload template.

That template explains why the next chapters proceed in two directions. Chapter 3 studies the interface around the block: tokenization before it and decoding after it. The GPU chapters then study how the block runs efficiently on hardware. Training chapters study how to distribute the block and its state. Serving chapters study how to run the block repeatedly under latency and memory constraints.

The next-token objective is simple. The Transformer is how that objective becomes computation. LLM systems begin when that computation becomes too large, too stateful, or too latency-sensitive to treat as a single model call.

Owner: Principal Author  
Purpose: Chapter 2 ready draft after source extraction, brief, technical review, and red-team review  
Evidence grade: A for course lecture and primary-paper architecture claims; no benchmark numbers used  
Assumptions: This chapter should make Transformer computation legible and defer kernel-level detail  
Open questions: Whether to turn encoder-decoder versus decoder-only contrast into a compact table during later copyedit  
Handoff: Production can move to book-level consistency audit
