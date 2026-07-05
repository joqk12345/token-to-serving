# Source Card: rajbhandari-2019-zero

Source ID: rajbhandari-2019-zero  
Title: ZeRO: Memory Optimizations Toward Training Trillion Parameter Models  
Author/issuer: Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He  
Date: 2019  
Source type: paper  
File path / URL: `https://arxiv.org/abs/1910.02054`  
Pages / slides / sections: Abstract and system framing  
Claim supported: ZeRO targets memory redundancy in data/model-parallel training while preserving computational granularity and low communication volume relative to naive alternatives.  
Exact quote: "eliminates memory redundancies"  
Paraphrase: The original ZeRO paper motivates state partitioning as a way to increase trainable model size without forcing the user into purely model-parallel execution.  
Evidence grade: A  
Technical sensitivity: algorithm | benchmark  
Conditions:
  model:
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Do not quote model-size, speedup, or PFLOP claims without experimental setup.
