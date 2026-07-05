Source ID: llmsys-09-beam-search
Title: 11868/11968 Large Language Model Systems, Lecture 9: Decoding
Author/issuer: Lei Li
Date:
Source type: lecture
File path / URL: downloads/llmsystem2026spring/source_pdfs/llmsys-09-decoding-cac2cd9402765ff5e6c24f7baffd321c.pdf
Pages / slides / sections: slides 11-17
Claim supported: Beam search approximates sequence decoding by keeping the top-k partial sequences and expanding them step by step.
Exact quote: "at each step, keep k best partial sequences"
Paraphrase: The lecture describes beam search as an approximate dynamic-programming-style search over partial generated sequences.
Evidence grade: A
Technical sensitivity: algorithm
Conditions:
  model:
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: stable teaching formulation
Checked by: Codex
Checked date: 2026-07-04
Notes: Mention length normalization/pruning only if supported by additional cards.

