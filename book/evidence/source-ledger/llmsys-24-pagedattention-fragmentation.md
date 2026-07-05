# Source Card: llmsys-24-pagedattention-fragmentation

Source ID: llmsys-24-pagedattention-fragmentation  
Title: Paged Attention & vLLM for Efficient LLM Inference Engine  
Author/issuer: Woosuk Kwon / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-24-vLLM_woosuk_kwon-b6a0750bb310949461ba5a635a1126eb.pdf`  
Pages / slides / sections: Slide 27  
Claim supported: Under PagedAttention's block allocation model, internal fragmentation is limited to the last block of a sequence and external fragmentation is avoided.  
Exact quote: "Only happens at the last block"  
Paraphrase: Because allocation happens in fixed-size KV blocks, unused capacity can remain only in a sequence's final partially filled block rather than in arbitrary holes between variable-size allocations.  
Evidence grade: A  
Technical sensitivity: memory allocation  
Conditions:
  model: PagedAttention-style KV block allocation
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: If deriving a waste bound, define block size and sequence-length assumptions explicitly.
