# Source Card: llmsys-24-preemption-recovery

Source ID: llmsys-24-preemption-recovery  
Title: Paged Attention & vLLM for Efficient LLM Inference Engine  
Author/issuer: Woosuk Kwon / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-24-vLLM_woosuk_kwon-b6a0750bb310949461ba5a635a1126eb.pdf`  
Pages / slides / sections: Slides 34-37  
Claim supported: When physical KV blocks are exhausted, a serving engine can preempt requests by swapping KV cache to CPU memory or deleting and recomputing it later.  
Exact quote: "Swapping"  
Paraphrase: KV-cache pressure creates a scheduling problem: the system may free memory from some requests so other requests can proceed, then recover preempted state by transfer or recomputation.  
Evidence grade: A  
Technical sensitivity: scheduling | memory management  
Conditions:
  model: Transformer serving with bounded GPU KV-cache capacity
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: The lecture names recomputation with FCFS as its strategy; avoid latency claims without the figure setup.
