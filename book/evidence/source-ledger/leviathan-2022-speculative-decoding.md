Source ID: leviathan-2022-speculative-decoding
Title: Fast Inference from Transformers via Speculative Decoding
Author/issuer: Yaniv Leviathan, Matan Kalman, Yossi Matias
Date: 2022-11-30
Source type: paper
File path / URL: https://arxiv.org/abs/2211.17192
Pages / slides / sections: abstract; algorithm sections
Claim supported: Speculative decoding uses efficient approximation models and parallel verification to accelerate autoregressive decoding while preserving the target model output distribution.
Exact quote: "computing several tokens in parallel"
Paraphrase: The paper introduces speculative decoding for faster exact decoding from large autoregressive Transformer models.
Evidence grade: A
Technical sensitivity: algorithm
Conditions:
  model: T5-XXL in paper experiments
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: primary speculative decoding source
Checked by: Codex
Checked date: 2026-07-04
Notes: Use to replace lecture-only support in Chapter 3. Do not conflate this exact algorithm with the lecture's top-k validation simplification.

