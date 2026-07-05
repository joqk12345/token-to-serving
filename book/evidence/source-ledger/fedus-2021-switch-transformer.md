# Source Card: fedus-2021-switch-transformer

Source ID: fedus-2021-switch-transformer  
Title: Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity  
Author/issuer: William Fedus, Barret Zoph, Noam Shazeer  
Date: 2021  
Source type: paper  
File path / URL: `https://arxiv.org/abs/2101.03961`  
Pages / slides / sections: Abstract and routing framing  
Claim supported: Switch Transformers use sparse expert selection so different inputs activate different parameters; the paper frames complexity, communication cost, and training instability as adoption barriers addressed by simplified routing.  
Exact quote: "selects different parameters for each incoming example"  
Paraphrase: Switch Transformer is a top-1 sparse MoE design where routing and load/stability issues are part of the system, not incidental details.  
Evidence grade: A  
Technical sensitivity: architecture | training  
Conditions:
  model: Switch Transformer
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Avoid pretraining-speed or trillion-parameter claims unless all setup conditions are carried.
