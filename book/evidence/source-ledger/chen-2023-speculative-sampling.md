Source ID: chen-2023-speculative-sampling
Title: Accelerating Large Language Model Decoding with Speculative Sampling
Author/issuer: Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, John Jumper
Date: 2023-02-02
Source type: paper
File path / URL: https://arxiv.org/abs/2302.01318
Pages / slides / sections: abstract; speculative sampling method
Claim supported: Speculative sampling accelerates Transformer decoding by generating multiple tokens per target-model call using draft continuations and modified rejection sampling that preserves the target distribution.
Exact quote: "generation of multiple tokens from each transformer call"
Paraphrase: The paper presents a related speculative sampling method and reports Chinchilla decoding speedups under a distributed setup.
Evidence grade: A
Technical sensitivity: algorithm
Conditions:
  model: Chinchilla 70B in reported benchmark
  hardware: distributed setup in paper
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: primary source, parallel to Leviathan et al.
Checked by: Codex
Checked date: 2026-07-04
Notes: Use as corroborating source if Chapter 3 discusses distribution-preserving speculative decoding generally.

