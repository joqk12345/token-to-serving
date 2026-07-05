Source ID: vaswani-2017-attention-is-all-you-need
Title: Attention Is All You Need
Author/issuer: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
Date: 2017-06-12
Source type: paper
File path / URL: https://arxiv.org/abs/1706.03762
Pages / slides / sections: abstract; model architecture sections
Claim supported: The Transformer is an attention-based encoder-decoder architecture that dispenses with recurrence and convolution, improves parallelizability, and became the canonical architecture anchor for Chapter 2.
Exact quote: "based solely on attention mechanisms, dispensing with recurrence and convolutions entirely"
Paraphrase: The paper introduces the Transformer as a sequence transduction architecture built around attention rather than recurrent or convolutional layers.
Evidence grade: A
Technical sensitivity: architecture
Conditions:
  model: Transformer
  hardware: paper reports eight GPUs for the large WMT experiment; preserve conditions when using training-cost claims
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: canonical source
Checked by: Codex
Checked date: 2026-07-04
Notes: Use for Chapter 2 attention, positional encoding, encoder-decoder architecture, masking, and training-cost claims. Keep course cards as teaching source; this is the primary technical anchor.

