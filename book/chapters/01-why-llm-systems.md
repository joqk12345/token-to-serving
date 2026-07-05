---
status: ready
chapter: 1
slug: 01-why-llm-systems
title: Why LLMs Are Systems Problems
primary_sources:
  - llmsys-01-intro-14e74a426e4a7e3ed485a026e1f65b70.pdf
secondary_sources: []
reader_level: engineer or graduate student with basic ML background
technical_depth: introductory but precise
---

# Why LLMs Are Systems Problems

A user types a prompt. The model answers in fluent sentences. At the surface, the system looks like a language interface: translate this, summarize that, write a function, solve a word problem, polish an email.

Underneath that interface is a long chain of engineering decisions. Text must become tokens. Tokens must become vectors. Vectors must move through repeated Transformer blocks. Those blocks must run as matrix multiplications, reductions, normalizations, nonlinearities, and memory transfers. During training, the system must move parameters, gradients, activations, and optimizer states through many devices. During inference, it must keep latency low while serving many users whose requests have different prompt lengths, output lengths, and arrival times.

That stack is what this book means by **LLM systems**: the full system required to train, adapt, optimize, and serve large language models.

LLM systems work asks for three things: understanding the techniques behind modern LLM systems, implementing core components such as fast CUDA kernels, scalable training systems, and efficient inference, and identifying new systems challenges created by LLM scale. [CITE: llmsys-01-system-challenges]

The important word is **systems**. A large language model is not only a model. It is also a workload.

## The Simple Contract

The mathematical contract begins simply. A language model assigns a probability to the next token given the prompt and the previous tokens:

```text
P(next token | prompt, previous tokens)
```

This appears in next-word prediction form:

```text
P(next word y_t | Prompt x, previous words y_1:t-1)
```

In a modern LLM book, it is more precise to say token rather than word, because the model normally reads and writes subword, byte, or byte-pair units rather than human words. The conditional structure is the same. The model sees a prefix and scores possible continuations. [CITE: llmsys-01-next-token-probability]

For a sequence, the model repeats the same idea. The probability of a sentence can be written as a product of conditional probabilities: first token, second token given the first, third token given the first two, and so on. Training pushes the model to assign high probability to the observed next token in real text. The common loss is cross-entropy for next-token prediction. [CITE: llmsys-01-training-objective]

This simple contract explains why one model can appear to do many things. Translation, summarization, code generation, question answering, and style transfer can all be framed as: given this input sequence, produce a useful output sequence.

It does not explain why the system is hard.

## The Contract Becomes a Tensor Program

The next-token contract must be implemented as computation.

At each step, tokens index an embedding table. The resulting vectors pass through many network layers. In a Transformer-based model, those layers include attention, linear projections, nonlinear activation functions, normalization, softmax operations, and residual paths. At the bottom, these become a small set of repeated operator patterns:

- matrix and tensor multiplication;
- elementwise maps;
- reductions such as sums and averages;
- normalization;
- softmax;
- memory movement.

These are the common computation layers and low-level operators that show up in language models. [CITE: llmsys-01-system-challenges]

Once the model is small, these details are implementation. Once the model is large, they become the main problem. The same next-token objective may require thousands of GPUs to train, careful partitioning to fit in memory, specialized kernels to keep hardware busy, and a serving runtime that can batch requests without breaking latency targets.

The model is a probability distribution. The system is the machinery that makes that distribution usable.

## The Dominant Shape: Decoder-Only Autoregression

Language models come in several architectural families. Encoder-only models, such as BERT-style masked language models, are built for representation and prediction over masked positions. Encoder-decoder models read an input sequence with an encoder and generate an output sequence with a decoder. Decoder-only models generate autoregressively, conditioning each next token on the previous tokens.

Decoder-only causal language models are the most common architecture choice for modern LLMs. [CITE: llmsys-01-decoder-only]

That choice matters for systems. Autoregressive generation has a serial dependency: token `t + 1` depends on token `t`. During training, many positions can be processed in parallel under a causal mask. During inference, generation usually advances one token at a time for each sequence. That difference will later explain why training throughput, inference throughput, and user-visible latency are different system problems.

This book will mostly reason from the decoder-only case because it is the dominant shape of current general-purpose LLM serving. Encoder-decoder and encoder-only models still matter, especially for understanding the Transformer family, but they are not the center of the serving stack.

## Capability Is Not the Same as Infrastructure

Common LLM capabilities include translation, commonsense reasoning, math reasoning, code generation, text rewriting, and image-prompt generation. The point is not that every example is perfect, or that a model "understands" all tasks in a human sense. The point is that a broad set of AI tasks can be exposed through the same token interface.

That interface hides a systems stack.

Suppose a user asks a model to write a small Python function. The visible work is a few lines of code. The hidden work includes:

1. tokenizing the prompt;
2. looking up embeddings;
3. running many Transformer layers over the prompt;
4. storing key/value tensors needed for later attention;
5. sampling or selecting the next token;
6. appending that token to the context;
7. repeating the process until the answer is complete;
8. scheduling the request alongside other requests;
9. moving data through GPU memory fast enough to avoid wasting compute.

The same request can be easy or hard depending on prompt length, output length, batch composition, model size, precision, hardware, and cache state.

That is why a model benchmark is not a serving system, and a serving demo is not a training system.

## The First Bottleneck: Resources

The practical question is simple: how many resources do you need to train a 100B model?

That question is deliberately larger than model architecture. To answer it, an engineer must account for:

- parameters;
- gradients;
- optimizer states;
- activations;
- batch size;
- sequence length;
- precision;
- GPU memory;
- GPU interconnect bandwidth;
- cluster network bandwidth;
- checkpointing;
- failure recovery;
- data loading.

The answer is not one number. It is a resource model.

This is the first habit of LLM systems work: translate model size and workload shape into compute, memory, bandwidth, and time.

## Computation Is Not Enough

A naive performance story says: make the math faster. That story is incomplete.

Making computation fast is not enough. Data transfer takes time. Large models have many parameters. Training moves gradients and optimizer states. Inference moves weights, activations, and KV-cache tensors. Long-context LLMs require large working memory. [CITE: llmsys-01-system-challenges]

This distinction appears throughout the book.

Sometimes the limiting factor is arithmetic throughput: the hardware cannot multiply fast enough. Sometimes the limiting factor is memory bandwidth: the arithmetic units wait for data. Sometimes it is memory capacity: the tensors do not fit. Sometimes it is communication: devices spend too much time exchanging parameters, gradients, activations, or cache blocks. Sometimes it is scheduling: the system has enough total capacity but cannot shape work into efficient batches.

A good LLM systems engineer asks which bottleneck is active before optimizing.

## Abstractions Decide What Engineers Can Build

A useful abstraction hides one concern while still supporting a wide range of applications.

At the upper level, engineers integrate models into product systems and track quality over time. At the middle level, they build training and inference software, runtime systems, and streaming dataflows. At the lower level, they write kernels, compilers, and hardware-specific code. [CITE: llmsys-01-system-challenges]

Each level hides something.

A deep learning framework hides device execution details behind tensors and graphs. A distributed training library hides some communication patterns behind data parallel or model parallel APIs. A serving runtime hides batching, cache allocation, and scheduling behind an inference endpoint. A kernel library hides memory tiling and warp-level execution behind an operator call.

The abstraction is successful only if it hides complexity without hiding the bottleneck that matters.

That is why this book moves up and down the stack. The reader needs the model objective, because it explains the workload. The reader needs Transformer computation, because it explains the operators. The reader needs GPU memory hierarchy, because it explains kernel performance. The reader needs distributed training, because the model no longer fits comfortably on one device. The reader needs serving architecture, because generation is interactive, stateful, and latency-sensitive.

## Co-Design Is the Pattern

LLMs need model-algorithm-system co-design. Model architecture, training and inference algorithms, software optimization, and hardware acceleration must be designed together. [CITE: llmsys-01-codesign]

This is not a slogan. It is a constraint.

An attention variant that reduces asymptotic memory may still perform poorly if it maps badly to GPU memory access. A quantization method that reduces weight memory may not improve latency if decoding is dominated by KV-cache bandwidth. A model-parallel strategy that fits the model may lose the gain through communication overhead. A batching strategy that improves throughput may harm tail latency for interactive users.

Every layer changes the tradeoff surface for the layers above and below it.

The rest of the book is organized around that fact.

Part I builds the shared vocabulary: tokens, probabilities, Transformer computation, tokenization, and decoding. Part II moves inside a single device: GPU programming, kernels, memory hierarchy, and attention acceleration. Part III studies training systems: frameworks, distributed data parallelism, model parallelism, ZeRO, and MoE. Part IV covers adaptation and compression. Part V turns to inference and serving: cost models, KV cache, vLLM, scheduling, disaggregation, and caching. Part VI returns to co-design as the durable engineering skill.

The visible product is language. The actual engineering object is a system that turns probability into service under resource constraints.

Owner: Principal Author  
Purpose: Chapter 1 ready draft after source extraction, brief, technical review, and red-team review  
Evidence grade: A for course-framing claims; no benchmark or historical model-scale numbers used  
Assumptions: The chapter should map the book and introduce bottlenecks without deriving formulas deeply  
Open questions: Whether to add a historical model-scale sidebar in a later revision  
Handoff: Production can move to book-level consistency audit
