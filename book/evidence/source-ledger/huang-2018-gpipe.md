# Source Card: huang-2018-gpipe

Source ID: huang-2018-gpipe  
Title: GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism  
Author/issuer: Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Mia Xu Chen, Dehao Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V. Le, Yonghui Wu, Zhifeng Chen  
Date: 2018  
Source type: paper  
File path / URL: `https://arxiv.org/abs/1811.06965`  
Pages / slides / sections: Abstract and pipeline parallelism algorithm  
Claim supported: GPipe partitions a network into layer subsequences across accelerators and uses batch-splitting pipeline parallelism to train models that exceed a single accelerator's memory.  
Exact quote: "batch-splitting pipelining"  
Paraphrase: GPipe frames pipeline parallelism as layer partitioning plus micro-batch pipeline execution for networks expressible as sequences of layers.  
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
Notes: Do not quote speedup/accuracy numbers without paper setup.
