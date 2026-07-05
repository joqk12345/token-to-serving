# Source Card: llmsys-24-vllm-parallelism-options

Source ID: llmsys-24-vllm-parallelism-options  
Title: Paged Attention & vLLM for Efficient LLM Inference Engine  
Author/issuer: Woosuk Kwon / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-24-vLLM_woosuk_kwon-b6a0750bb310949461ba5a635a1126eb.pdf`  
Pages / slides / sections: Slides 69-76  
Claim supported: vLLM's serving-parallelism discussion distinguishes data, tensor, expert, context, pipeline, and prefill/decode disaggregation tradeoffs.  
Exact quote: "Things to consider for parallelism"  
Paraphrase: Serving parallelism must consider communication volume, overlap, network heterogeneity, and load imbalance; different parallelism modes improve different bottlenecks.  
Evidence grade: A  
Technical sensitivity: distributed serving | parallelism  
Conditions:
  model: LLM serving
  hardware: multi-GPU/multi-node serving settings
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use lightly in Chapter 13; Chapter 14 owns disaggregation and cross-engine serving architecture.
