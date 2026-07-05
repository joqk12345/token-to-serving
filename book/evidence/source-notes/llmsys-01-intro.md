# Source Notes — llmsys-01-intro

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-01-intro-14e74a426e4a7e3ed485a026e1f65b70.pdf`

Course: 11868/11968 Large Language Model Systems  
Instructor: Lei Li  
Pages/slides: 51

## High-Level Role

This lecture establishes the book's opening frame: LLMs appear as general-purpose language interfaces, but the course is about the system stack needed to train and serve them.

It supports Chapter 1 and the final synthesis chapter.

## Core Claims

| Claim                                                                                                                                          | Evidence grade | Source card                        |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | ---------------------------------- |
| Modern LLMs are generalist, instructible, and increasingly agentic.                                                                            | A              | `llmsys-01-modern-llms`            |
| A language model predicts the probability of the next token conditioned on prompt and previous tokens.                                         | A              | `llmsys-01-next-token-probability` |
| Decoder-only autoregressive models are the most popular architecture choice for modern LLMs.                                                   | A              | `llmsys-01-decoder-only`           |
| Pretraining commonly uses cross-entropy loss for next-token prediction.                                                                        | A              | `llmsys-01-training-objective`     |
| LLM systems are constrained by compute, memory, communication, data movement, and programming abstractions.                                    | A              | `llmsys-01-system-challenges`      |
| LLM system design requires model-algorithm-system co-design across architecture, algorithms, software optimization, and hardware acceleration. | A              | `llmsys-01-codesign`               |

## Useful Chapter Material

### Opening Problem

The lecture contrasts user-facing capabilities, such as translation, summarization, code generation, and math reasoning, with the engineering questions underneath: how many resources are required to train a 100B model, how to build fast kernels, and how to train and serve efficiently.

Chapter use: open Chapter 1 with the contrast between visible capability and hidden infrastructure.

### Mathematical Anchor

The lecture gives the next-token probability formulation:

```text
P(next word y_t | prompt x, previous words y_1:t-1)
```

It then expands sentence probability as a product of conditional token probabilities.

Chapter use: introduce the objective before discussing Transformer computation and system costs.

### Architecture Taxonomy

The lecture distinguishes:

- encoder-only masked language models, such as BERT;
- encoder-decoder models for conditional generation;
- decoder-only autoregressive models, the dominant LLM architecture.

Chapter use: keep this taxonomy in Part I before going deeper into Transformer computation.

### System Stack

The lecture maps computation from high-level model layers to low-level operators:

- multi-head attention;
- layer norm;
- dropout;
- linear layers;
- nonlinear activation;
- softmax;
- matrix/tensor multiplication;
- reductions;
- map operations;
- memory movement.

Chapter use: bridge from abstract LLM behavior to kernels and memory movement.

### System Challenge

The lecture states the central system problem: compute training and inference for larger LLMs on bigger datasets with fewer resources, faster.

It also names abstraction levels:

- distributed / parallel system;
- deep learning frameworks;
- operators on blocks of data;
- CUDA/TPU kernels;
- compression.

Chapter use: Chapter 1's "map of the book."

## Figure Candidates

| Figure ID                        | Purpose                                                         | Form                                       |
| -------------------------------- | --------------------------------------------------------------- | ------------------------------------------ |
| `fig-01-visible-vs-hidden-stack` | Show user-facing LLM tasks above system layers.                 | Stack diagram                              |
| `fig-01-token-probability-chain` | Explain next-token probability and sequence probability.        | Token chain with conditional probabilities |
| `fig-01-abstraction-levels`      | Map model layer to framework to operator to kernel to hardware. | Layered architecture diagram               |
| `fig-01-codesign-loop`           | Show model, algorithm, software, and hardware co-design.        | Four-node loop                             |

## Risks / Checks

- Do not overstate the course's examples as proof of general reasoning ability.
- Keep "generalist, instructible, agentic" as course framing unless supported by additional sources.
- When mentioning GPT-3's 300B tokens, either cite this lecture as course claim or add a primary source.
- Do not turn the introduction into a generic AI history chapter.

Owner: Technical Researcher  
Purpose: Source extraction for Chapter 1 and synthesis chapter  
Evidence grade: A for course framing, with external primary sources needed for historical/public model details  
Assumptions: The course lecture is an authoritative source for this book's teaching spine  
Open questions: Whether to include an external scaling-law source in Chapter 1  
Handoff: Book Architect for `01-why-llm-systems` brief
