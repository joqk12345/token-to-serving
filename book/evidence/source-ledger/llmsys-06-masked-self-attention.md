Source ID: llmsys-06-masked-self-attention
Title: 11868/11968 Large Language Model Systems, Lecture 6: Transformer
Author/issuer: Lei Li
Date:
Source type: lecture
File path / URL: downloads/llmsystem2026spring/source_pdfs/llmsys-06-transformer-14bd7575a2f6c8bac60522354c11d691.pdf
Pages / slides / sections: slide 14
Claim supported: Decoder self-attention masks future positions before the softmax so the decoder cannot attend to tokens to its right.
Exact quote: "Maskout right side before softmax (-inf)"
Paraphrase: Causal self-attention applies a mask to future positions before softmax, making autoregressive generation consistent with left-to-right conditioning.
Evidence grade: A
Technical sensitivity: formula
Conditions:
  model: decoder Transformer
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: stable teaching formulation
Checked by: Codex
Checked date: 2026-07-04
Notes: Useful bridge to decoding and KV cache later.

