---
status: ready
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

# Tokenization, Context, and Decoding

The Transformer does not read text. It reads token IDs.

It also does not directly write prose. It produces a distribution over the next token. A decoding procedure turns that distribution into an actual sequence.

Those two interfaces, tokenization before the model and decoding after the model, shape almost every system property that follows. Tokenization changes sequence length, vocabulary size, embedding cost, multilingual behavior, and how text maps into model memory. Decoding changes quality, diversity, latency, batching behavior, and how much work must be done one step at a time.

This chapter is about the boundary around the Transformer block.

## Tokens Are Not Words

A tokenizer splits text into units that can index an embedding table. The token IDs are the model's discrete interface. [CITE: llmsys-08-tokenization-tradeoffs]

![Word, character, and subword tokenization split the same text differently, trading off vocabulary size, sequence length, and handling of rare or unseen strings.](../figures/artwork/ch03/fig-03-tokenizer-comparison.svg)

That sounds simple until the word "unit" is taken seriously.

Consider:

```text
Bob's handyman is a do-it-yourself kinda guy, isn't he?
```

Is `Bob's` one word or two? Is `do-it-yourself` one word, three words, or a compound? What about `isn't`? What about languages that do not use spaces between words?

Word boundaries are not a universal technical foundation. Word-level tokenization is easy for some languages and brittle for others. It creates out-of-vocabulary behavior: if a word was not in the vocabulary, the system needs an unknown token or another fallback. A larger vocabulary reduces unknowns but increases the number of learned embedding and output parameters. A smaller vocabulary is cheaper but loses coverage. The subword paper on rare-word translation is the primary source for that open-vocabulary motivation. [CITE: llmsys-08-tokenization-tradeoffs; sennrich-2016-bpe-rare-words]

Character-level tokenization avoids many unknown-token problems. Any word can be spelled as characters. But character sequences are longer. Longer sequences increase the amount of work the Transformer must do. Tokens also become less semantically compact: a single character often carries less meaning than a word or subword.

Subword tokenization is the compromise most modern LLM readers encounter. Frequent patterns can become tokens. Rare words can decompose into smaller units.

## BPE as a Compression Habit

Byte Pair Encoding, or BPE, starts from small units and repeatedly merges frequent adjacent pairs into new tokens until the vocabulary reaches a target size. The Sennrich paper is the primary anchor for this NMT use of subword units. [CITE: llmsys-08-bpe-algorithm; sennrich-2016-bpe-rare-words]

![Byte-pair encoding repeatedly merges frequent adjacent symbols, growing a vocabulary of reusable subword units.](../figures/artwork/ch03/fig-03-bpe-loop.svg)

A small toy example is enough:

```text
cat
catch
rat
rattle
```

If `a` followed by `t` appears often, the tokenizer may merge `a` and `t` into `at`. If `c` followed by `at` appears often, it may merge into `cat`. Over many merges, common chunks become single tokens.

BPE tokens are not guaranteed to be linguistic units. They are artifacts of corpus statistics and merge rules. That is part of their usefulness. The tokenizer does not need a complete theory of morphology. It needs a vocabulary that keeps sequences manageable while still representing rare strings.

The systems consequence is direct:

```text
more tokens in the sequence -> more positions for the model to process
larger vocabulary -> larger embedding/output tables
```

Neither side is free.

## Vocabulary Size Is a Systems Choice

Tokenizer design affects at least four costs.

First, it affects sequence length. If a language, domain, or notation is over-tokenized, a short human string becomes a long model sequence. That increases attention work, KV cache size, and latency.

Second, it affects vocabulary size. A larger vocabulary means larger embedding and output projection structures. It may shorten sequences, but it also increases the dimension of the final token distribution.

Third, it affects data distribution. Tokenization choices can make code, numbers, and multilingual text easier or harder for the model to represent.

Fourth, it affects downstream adaptation. If a tokenizer poorly represents a language or domain, fine-tuning has to work against the representation.

Practical LLM tokenization includes corpus deduplication and filtering, handling code, handling numbers, multilingual vocabulary capacity, vocabulary sharing, and tokenizer-free alternatives. SentencePiece is the primary source for raw-sentence subword training, and the LLaMA paper anchors the modern decoder-only example. [CITE: llmsys-08-practical-tokenization; kudo-richardson-2018-sentencepiece; touvron-2023-llama]

This book will not treat those as preprocessing footnotes. Tokenization defines the first resource model of an LLM request: how many tokens enter the system.

## Context Is a Budget

Once text is tokenized, context length becomes a budget.

A model that accepts `N` tokens does not accept `N` words. It accepts `N` tokenizer outputs. The same paragraph may cost different numbers of tokens in different languages, domains, or tokenizers. Code and math notation may behave differently from ordinary prose. A multilingual system may give some languages a shorter representation than others.

That budget matters during both training and inference.

During training, sequence length affects activation memory and attention computation. During inference, prompt length affects prefill work and KV cache size. During decoding, every generated token extends the context and adds new cache state.

This is why tokenization belongs in a systems book. A tokenizer choice can show up later as a memory problem, a throughput problem, or a fairness problem across languages.

## Decoding Turns Probabilities Into Text

The model produces logits for the next token. Decoding decides what to do with them.

![Greedy decoding, sampling, and beam search use the same next-token distribution differently: selecting the highest-probability token, drawing stochastically, or tracking multiple partial sequences.](../figures/artwork/ch03/fig-03-decode-methods.svg)

Sequence decoding asks for:

```text
argmax_y P(y | x)
```

The exhaustive search over all possible sequences is too expensive. With vocabulary size `V` and sequence length `N`, the naive search space grows exponentially. The speculative decoding paper is the primary source for that acceleration strategy. [CITE: llmsys-09-autoregressive-decode-latency; leviathan-2022-speculative-decoding]

Practical systems use approximate procedures.

Greedy decoding chooses the highest-probability next token at each step. It is simple and cheap, but each local decision is final. Sampling draws from the model's distribution, often with additional controls. Beam search keeps multiple partial candidates and expands them step by step. Beam search keeps the top `k` partial sequences at each step and expands them with one more forward generation. [CITE: llmsys-09-beam-search]

Each method expresses a different tradeoff:

- greedy decoding favors determinism and low overhead;
- sampling favors diversity;
- beam search favors approximate sequence-level search at higher compute cost.

These are not only quality choices. They affect runtime behavior.

## Autoregression Is Serial

The main system constraint in decoding is that the model usually cannot generate token `t + 1` until token `t` has been selected.

LLM autoregressive decoding is slow because generation proceeds one token at a time, and producing `N` tokens normally requires `N` forward passes. [CITE: llmsys-09-autoregressive-decode-latency; leviathan-2022-speculative-decoding]

This serial dependency separates LLM inference from many throughput-oriented neural network workloads. A vision classifier can process an image and finish. A language model serving request may remain active for hundreds or thousands of decode steps. Each step is small compared with the full prompt prefill, but it is latency-sensitive and stateful.

That is why later serving chapters will distinguish prefill from decode. Prefill processes the input context. Decode extends it.

## Speculative Decoding Changes the Work Shape

Speculative decoding attacks the serial bottleneck by adding another model.

![Speculative decoding uses a draft model to propose tokens and a target model to validate them, changing the shape of autoregressive work while preserving target-model validation.](../figures/artwork/ch03/fig-03-speculative-pipeline.svg)

The basic idea is to use a smaller draft model to propose several tokens, then use the larger target model to validate them. The primary speculative decoding paper keeps the target distribution exact. [CITE: llmsys-09-speculative-decoding; leviathan-2022-speculative-decoding]

The intuition is:

```text
small model proposes several likely next tokens
large model checks those proposed tokens
accepted tokens advance the sequence faster
rejected tokens force correction
```

The benefit depends on alignment between draft and target models. If draft tokens are often accepted, the system can reduce latency. If they are often rejected, the extra draft work may not help. Common draft lengths such as `N = 4` or `N = 8` should be treated as workload-dependent rather than universal. [CITE: llmsys-09-speculative-decoding; chen-2023-speculative-sampling]

For this chapter, speculative decoding is mainly a preview. The deeper serving question is how to schedule many active requests, each with different prompt lengths, decode lengths, cache states, and latency targets.

## Advanced Sidebar: EAGLE as a Hint of the Direction

EAGLE predicts final-layer features rather than directly predicting next tokens with a separate small language model. It uses the original model's embedding and language-model head, a small Transformer layer, and tree attention for efficient implementation. [CITE: llmsys-09-speculative-decoding; li-2024-eagle]

![EAGLE extends speculative decoding by predicting features that help propose candidate continuations before target-model validation.](../figures/artwork/ch03/fig-03-eagle-feature-loop.svg)

The details are advanced enough to defer. The important lesson is that decoding acceleration is not only about choosing a different search algorithm. It can change the internal representation being predicted, the attention mask used for validation, and the shape of the work presented to the hardware.

Decoding is part of system design.

## The Boundary Becomes the Workload

Tokenization and decoding sit on opposite sides of the Transformer, but they meet in the same resource model.

Tokenization determines how long the input is. Decoding determines how long generation runs. Together they shape:

- how much prefill work is needed;
- how large the KV cache becomes;
- how many decode steps remain active;
- how requests batch together;
- how much memory is consumed per user;
- how output quality trades against latency.

The model's next-token distribution is only the center of the system. The full LLM workload includes everything required to feed that distribution and turn it into text.

The next part of the book moves down a layer. Once text has become token matrices and decoding has created a workload, the question becomes how that workload runs on hardware. The next chapters enter the GPU.

Owner: Principal Author  
Purpose: Chapter 3 ready draft after source extraction, brief, technical review, and red-team review  
Evidence grade: A for course lecture and primary-paper tokenization/decoding claims; no benchmark numbers used  
Assumptions: This chapter introduces decoding concepts and defers serving architecture to Chapters 12-14  
Open questions: Whether tokenizer-free models belong as a later sidebar  
Handoff: Production can move to book-level consistency audit
