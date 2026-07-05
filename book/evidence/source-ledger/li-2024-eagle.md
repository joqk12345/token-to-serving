Source ID: li-2024-eagle
Title: EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty
Author/issuer: Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang
Date: 2024-01-26
Source type: paper
File path / URL: https://arxiv.org/abs/2401.15077
Pages / slides / sections: abstract; method sections
Claim supported: EAGLE performs speculative sampling by predicting target-model internal features rather than only next-token candidates, improving inference efficiency.
Exact quote: "autoregression at the feature (second-to-top-layer) level"
Paraphrase: The paper introduces EAGLE as a speculative sampling framework based on second-to-top-layer feature prediction.
Evidence grade: A
Technical sensitivity: algorithm
Conditions:
  model: Vicuna, LLaMA2-Chat, Mixtral in paper evaluation
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: primary EAGLE source
Checked by: Codex
Checked date: 2026-07-04
Notes: Use if Chapter 3 keeps the EAGLE preview. EAGLE-2 should get a separate card if discussed.

