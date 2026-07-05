# Source Card: llmsys-17-moe-ffn-router

Source ID: llmsys-17-moe-ffn-router  
Title: System for MOE Models  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-17-MoE-3aa3125f9ccdd4bb7109ef077fbe9260.pdf`  
Pages / slides / sections: Slides 5, 40  
Claim supported: Transformer MoE replaces a single dense FFN with multiple expert FFNs and a routing/gating network that selects one or more experts for each token.  
Exact quote: "Instead of a single dense FFN, using multiple FFNs (experts)"  
Paraphrase: The MoE layer is a sparse Transformer FFN replacement controlled by a router.  
Evidence grade: A  
Technical sensitivity: architecture  
Conditions:
  model: Transformer MoE
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Do not conflate Transformer MoE with classical mixture-of-experts learning.
