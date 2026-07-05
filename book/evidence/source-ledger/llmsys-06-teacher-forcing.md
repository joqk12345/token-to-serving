Source ID: llmsys-06-teacher-forcing
Title: 11868/11968 Large Language Model Systems, Lecture 6: Transformer
Author/issuer: Lei Li
Date:
Source type: lecture
File path / URL: downloads/llmsystem2026spring/source_pdfs/llmsys-06-transformer-14bd7575a2f6c8bac60522354c11d691.pdf
Pages / slides / sections: slide 19
Claim supported: Transformer sequence-to-sequence training uses cross-entropy with teacher forcing, conditioning on ground-truth prefixes during training.
Exact quote: "Teacher-forcing during training."
Paraphrase: During training, the decoder is given the ground-truth prefix and optimized with cross-entropy over the next target token.
Evidence grade: A
Technical sensitivity: formula
Conditions:
  model: encoder-decoder Transformer
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: stable teaching formulation
Checked by: Codex
Checked date: 2026-07-04
Notes: Keep distinct from inference, where generated prefixes are used.

