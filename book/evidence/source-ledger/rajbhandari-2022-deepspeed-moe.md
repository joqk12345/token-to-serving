# Source Card: rajbhandari-2022-deepspeed-moe

Source ID: rajbhandari-2022-deepspeed-moe  
Title: DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale  
Author/issuer: Samyam Rajbhandari et al.  
Date: 2022  
Source type: paper  
File path / URL: `https://arxiv.org/abs/2201.05596`  
Pages / slides / sections: Abstract and system framing  
Claim supported: DeepSpeed-MoE frames MoE systems as needing both training and inference support because sparse models have unique architecture, communication, and deployment challenges.  
Exact quote: "fast MoE model inference remains challenging"  
Paraphrase: DeepSpeed-MoE is a source for treating MoE as an end-to-end systems problem, including expert parallelism, communication, kernels, and inference.  
Evidence grade: A  
Technical sensitivity: system | benchmark  
Conditions:
  model: MoE Transformer
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Do not quote paper speedup/latency/cost numbers without full setup.
