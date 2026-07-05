# Source Card: llmsys-24-os-virtual-memory-analogy

Source ID: llmsys-24-os-virtual-memory-analogy  
Title: Paged Attention & vLLM for Efficient LLM Inference Engine  
Author/issuer: Woosuk Kwon / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-24-vLLM_woosuk_kwon-b6a0750bb310949461ba5a635a1126eb.pdf`  
Pages / slides / sections: Slide 38  
Claim supported: The lecture compares OS pages to KV blocks and shared pages across processes to shared KV blocks across samples, while noting request-level preemption and recomputation-based recovery.  
Exact quote: "OS pages ↔ KV blocks"  
Paraphrase: OS virtual memory is a useful analogy for PagedAttention, but the serving system's block table and recovery mechanisms are specialized for KV-cache data.  
Evidence grade: A  
Technical sensitivity: analogy | memory management  
Conditions:
  model: PagedAttention-style serving engine
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: State the differences explicitly so readers do not overextend the analogy.
