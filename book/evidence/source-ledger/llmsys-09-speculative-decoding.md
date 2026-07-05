Source ID: llmsys-09-speculative-decoding
Title: 11868/11968 Large Language Model Systems, Lecture 9: Decoding
Author/issuer: Lei Li
Date:
Source type: lecture
File path / URL: downloads/llmsystem2026spring/source_pdfs/llmsys-09-decoding-cac2cd9402765ff5e6c24f7baffd321c.pdf
Pages / slides / sections: slides 23-46
Claim supported: Speculative decoding uses a smaller draft model to propose multiple tokens and a larger target model to validate them, reducing latency when validation is cheaper than sequential generation.
Exact quote: "use a small model (draft model) to generate N 'drafty' tokens and then leverage the large model (target model) to validate them"
Paraphrase: The lecture frames speculative decoding as a latency optimization for autoregressive inference that validates several proposed tokens in a target-model pass.
Evidence grade: A
Technical sensitivity: algorithm
Conditions:
  model: draft-target model pair
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: algorithm variants differ; acceptance rules need paper-specific support
Checked by: Codex
Checked date: 2026-07-04
Notes: Avoid universal acceptance-rule claims until a primary paper card is added.

