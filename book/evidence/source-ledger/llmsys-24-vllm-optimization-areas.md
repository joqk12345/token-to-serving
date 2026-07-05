# Source Card: llmsys-24-vllm-optimization-areas

Source ID: llmsys-24-vllm-optimization-areas  
Title: Paged Attention & vLLM for Efficient LLM Inference Engine  
Author/issuer: Woosuk Kwon / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-24-vLLM_woosuk_kwon-b6a0750bb310949461ba5a635a1126eb.pdf`  
Pages / slides / sections: Slides 47-64  
Claim supported: vLLM optimization areas include minimizing CPU overheads, using efficient GPU kernels, model parallelism, and memory management/caching.  
Exact quote: "Minimizing CPU overheads"  
Paraphrase: The lecture presents vLLM as a full inference-engine stack, where KV-cache paging is one optimization among scheduler, kernel, and parallelism work.  
Evidence grade: A  
Technical sensitivity: serving architecture  
Conditions:
  model: vLLM inference engine
  hardware:
  batch size:
  sequence length:
  precision:
  software version: lecture version as of April 2026
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Keep Chapter 13 centered on KV cache; mention other optimizations only to place PagedAttention inside the engine.
