---
status: brief
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

# From Next-Token Probability to Transformer Computation — Brief

## Chapter Thesis

The language-modeling objective becomes a repeated tensor computation: embeddings create token representations, attention mixes contextual information, feed-forward layers transform each position, masks enforce autoregressive conditioning, and training optimizes next-token or conditional-token probabilities.

## Reader Problem

After Chapter 1, the reader knows LLMs are systems problems. This chapter must make the model's core computation legible enough that later chapters on kernels, memory, distributed training, and serving have a shared object.

## System Bottleneck

Primary bottleneck class: tensor computation and memory movement inside repeated Transformer blocks.

Secondary bottlenecks introduced but not deeply solved:

- attention matrix size;
- embedding/output vocabulary cost;
- sequential dependency during generation;
- training/inference mismatch.

## Source Map

| Claim                                                                                  | Source card                                                        | Evidence grade | Notes                              |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | -------------- | ---------------------------------- |
| Transformer blocks combine embeddings, MHA, FFN, residual connections, and layer norm. | `llmsys-06-transformer-components`                                 | A              | Main architecture anchor.          |
| Decoder self-attention masks future positions before softmax.                          | `llmsys-06-masked-self-attention`                                  | A              | Bridge to autoregressive decoding. |
| Training uses teacher forcing for conditional generation.                              | `llmsys-06-teacher-forcing`                                        | A              | Contrast with inference later.     |
| LLaMA uses decoder-only Transformer with pre-norm, SwiGLU, and RoPE.                   | `llmsys-07-llama-architecture`                                     | A              | Modern LLM example.                |
| T5 uses encoder-decoder text-to-text format.                                           | `llmsys-07-t5-text-to-text`                                        | A              | Contrast with decoder-only.        |
| Next-token probability and cross-entropy were introduced in Chapter 1 source cards.    | `llmsys-01-next-token-probability`; `llmsys-01-training-objective` | A              | Reuse and deepen.                  |

## Explanation Arc

1. Start with a short sequence: "Pittsburgh is a city of ...".
2. Convert words/tokens into vectors through embeddings.
3. Explain why position must enter the representation.
4. Introduce Q, K, V as learned views of each token representation.
5. Explain attention as content-based mixing, then multi-head attention as multiple mixing subspaces.
6. Add causal masking: why the decoder cannot look right.
7. Add FFN, residual connections, and layer normalization.
8. Explain training with teacher forcing and cross-entropy.
9. Contrast encoder-decoder and decoder-only systems.
10. Close with the systems hook: this elegant block becomes expensive because it repeats over layers, tokens, heads, and batches.

## Required Figures

| Figure ID                 | Purpose                                              | Form                   | Source                                                                                |
| ------------------------- | ---------------------------------------------------- | ---------------------- | ------------------------------------------------------------------------------------- |
| `fig-02-token-to-logits`  | Show token -> embedding -> blocks -> logits.         | Dataflow               | `llmsys-06-transformer-components`                                                    |
| `fig-02-qkv-attention`    | Explain Q/K/V projections and weighted value mixing. | Dataflow + tiny matrix | `llmsys-06-transformer-components`                                                    |
| `fig-02-causal-mask`      | Show future-token masking before softmax.            | Attention matrix       | `llmsys-06-masked-self-attention`                                                     |
| `fig-02-model-family-map` | Compare encoder-only, encoder-decoder, decoder-only. | Table                  | `llmsys-01-decoder-only`; `llmsys-07-t5-text-to-text`; `llmsys-07-llama-architecture` |

## Technical Checks

- Formula correctness: verify scaled dot-product attention notation before final draft.
- Complexity / memory accounting: introduce qualitatively only; detailed attention IO/memory belongs to FlashAttention chapter.
- Hardware assumptions: none in this chapter.
- Benchmark conditions: no performance numbers unless sourced.
- Terminology consistency: use "token" consistently, not "word", after initial examples.

## Open Questions

- Should the chapter include the original Transformer positional encoding formula, or avoid it until a paper card is added?
- Should RoPE/SwiGLU be one paragraph each or a sidebar?
- Should T5 be kept as contrast, or deferred to a footnote/box?

## Handoff

Owner: Book Architect  
Purpose: Establish Chapter 2 structure and evidence envelope  
Evidence grade: A for course-framing claims; original paper cards needed for detailed historical and hyperparameter claims  
Assumptions: Chapter 2 should make the computation understandable, not exhaust Transformer variants  
Open questions: Need decision on how deep to go into RoPE/SwiGLU  
Handoff: Systems Explainer for draft after formula checks
