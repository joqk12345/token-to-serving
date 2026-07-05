# Source Card: narayanan-2021-megatron-lm

Source ID: narayanan-2021-megatron-lm  
Title: Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM  
Author/issuer: Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Anand Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, Matei Zaharia  
Date: 2021  
Source type: paper  
File path / URL: `https://arxiv.org/abs/2104.04473`  
Pages / slides / sections: Abstract and parallelism composition discussion  
Claim supported: Tensor, pipeline, and data parallelism can be composed to scale large language model training, and naive use of parallelism can run into scaling issues from communication and waiting.  
Exact quote: "tensor, pipeline, and data parallelism"  
Paraphrase: The paper studies how to combine model-parallel and data-parallel techniques for large language model training and introduces interleaved pipeline scheduling as one optimization.  
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
Notes: Avoid using throughput/MFU numbers without full experimental setup.
