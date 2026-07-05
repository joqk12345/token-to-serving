Source ID: llmsys-01-next-token-probability
Title: 11868/11968 Large Language Model Systems, Lecture 1: Introduction
Author/issuer: Lei Li
Date: 2026-01-15
Source type: lecture
File path / URL: downloads/llmsystem2026spring/source_pdfs/llmsys-01-intro-14e74a426e4a7e3ed485a026e1f65b70.pdf
Pages / slides / sections: slides 21-22
Claim supported: A language model models the probability of the next token conditioned on the prompt and previous generated tokens; sequence probability factors into conditional next-token probabilities.
Exact quote: "P(next word y_t | Prompt x, previous words y_1:t-1 )"
Paraphrase: The lecture introduces language modeling as conditional next-token prediction and expands a sentence probability as a product of token-level conditional probabilities.
Evidence grade: A
Technical sensitivity: formula
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
Notes: Use for Part I explanation. Avoid overloading "word" versus "token"; the slide says word, but the book should generalize to token after defining tokenizer.

