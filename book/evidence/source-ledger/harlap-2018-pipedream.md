# Source Card: harlap-2018-pipedream

Source ID: harlap-2018-pipedream  
Title: PipeDream: Fast and Efficient Pipeline Parallel DNN Training  
Author/issuer: Aaron Harlap, Deepak Narayanan, Amar Phanishayee, Vivek Seshadri, Nikhil Devanur, Greg Ganger, Phil Gibbons  
Date: 2018  
Source type: paper  
File path / URL: `https://arxiv.org/abs/1806.03377`  
Pages / slides / sections: Abstract and pipeline scheduling system  
Claim supported: PipeDream pipelines forward and backward passes of different inputs across partitioned DNN layers to keep GPUs productive and overlap computation and communication.  
Exact quote: "pipelining execution"  
Paraphrase: The paper presents a pipeline-parallel training system that partitions layers, schedules forward/backward work across inputs, and uses weight versioning for correctness.  
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
Notes: Use for 1F1B/PipeDream context; benchmark claims need setup.
