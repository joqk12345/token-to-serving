Source ID: llmsys-09-autoregressive-decode-latency
Title: 11868/11968 Large Language Model Systems, Lecture 9: Decoding
Author/issuer: Lei Li
Date:
Source type: lecture
File path / URL: downloads/llmsystem2026spring/source_pdfs/llmsys-09-decoding-cac2cd9402765ff5e6c24f7baffd321c.pdf
Pages / slides / sections: slides 23-24, 39-41
Claim supported: Autoregressive LLM decoding is slow because generating N tokens normally requires N sequential forward passes, while speculative validation can evaluate proposed tokens more parallelly.
Exact quote: "need to generate one token at the time in a sequential manner"
Paraphrase: The lecture identifies serial token generation as a key latency source in LLM decoding.
Evidence grade: A
Technical sensitivity: implementation
Conditions:
  model: autoregressive LLM
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: stable high-level formulation
Checked by: Codex
Checked date: 2026-07-04
Notes: The "100s of milliseconds" example needs conditions before use as a numeric claim.

