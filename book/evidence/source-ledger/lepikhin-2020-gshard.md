# Source Card: lepikhin-2020-gshard

Source ID: lepikhin-2020-gshard  
Title: GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding  
Author/issuer: Dmitry Lepikhin et al.  
Date: 2020  
Source type: paper  
File path / URL: `https://arxiv.org/abs/2006.16668`  
Pages / slides / sections: Abstract and automatic sharding framing  
Claim supported: GShard combines conditional computation with automatic sharding to express and run large sparse MoE Transformer models across many accelerators.  
Exact quote: "conditional computation and automatic sharding"  
Paraphrase: GShard is a source for MoE expert parallelism and automatic partitioning as a systems approach to sparse Transformer scaling.  
Evidence grade: A  
Technical sensitivity: distributed training | compiler/runtime  
Conditions:
  model: sparsely-gated MoE Transformer
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Do not quote parameter count, TPU count, or training-time claims without experiment context.
