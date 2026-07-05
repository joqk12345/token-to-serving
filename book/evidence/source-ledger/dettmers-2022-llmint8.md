# Source Card: dettmers-2022-llmint8

Source ID: dettmers-2022-llmint8  
Title: LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale  
Author/issuer: Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer  
Date: 2022  
Source type: paper  
File path / URL: `https://arxiv.org/abs/2208.07339`  
Pages / slides / sections: Abstract and method framing  
Claim supported: LLM.int8 uses vector-wise quantization for most matrix-multiplication features and a mixed-precision path for emergent outlier dimensions.  
Exact quote: "mixed-precision decomposition"  
Paraphrase: LLM.int8 is a source for explaining why LLM quantization must treat activation outliers specially rather than blindly quantizing every value the same way.  
Evidence grade: A  
Technical sensitivity: algorithm | benchmark  
Conditions:
  model: Transformer LLMs
  hardware:
  batch size:
  sequence length:
  precision: INT8 plus higher-precision outlier path
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Do not quote memory or performance numbers without experiment setup.
