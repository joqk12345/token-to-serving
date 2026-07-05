# Source Card: li-2020-pytorch-ddp

Source ID: li-2020-pytorch-ddp  
Title: PyTorch Distributed: Experiences on Accelerating Data Parallel Training  
Author/issuer: Shen Li, Yanli Zhao, Rohan Varma, Omkar Salpekar, Pieter Noordhuis, Teng Li, Adam Paszke, Jeff Smith, Brian Vaughan, Pritam Damania, Soumith Chintala  
Date: 2020  
Source type: paper  
File path / URL: `https://arxiv.org/abs/2006.15704`  
Pages / slides / sections: Abstract and DDP design/evaluation paper  
Claim supported: PyTorch DDP addresses the non-trivial dependency between computation and communication with techniques including gradient bucketing and overlapping computation with communication.  
Exact quote: "bucketing gradients"  
Paraphrase: The paper frames data parallelism as model replicas producing gradients independently and communicating gradients each iteration, and identifies bucketing and compute/communication overlap as native DDP acceleration techniques.  
Evidence grade: A  
Technical sensitivity: implementation | benchmark  
Conditions:
  model:
  hardware: paper evaluation includes multi-GPU settings; any scaling number must carry setup
  batch size:
  sequence length:
  precision:
  software version: PyTorch v1.5 context in abstract
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Do not quote near-linear scaling without paper setup; use for design motivation and DDP mechanism.
