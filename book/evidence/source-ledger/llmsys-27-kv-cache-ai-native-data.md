# Source Card: llmsys-27-kv-cache-ai-native-data

Source ID: llmsys-27-kv-cache-ai-native-data  
Title: KV Cache  
Author/issuer: Junchen Jiang / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-27-LMCache_junchenjiang-c21828fe582270cb5e08b4a21a002956.pdf`  
Pages / slides / sections: KV Cache is the new AI-Native Data  
Claim supported: KV cache can be treated as a managed data object, not merely an internal tensor.  
Exact quote: "not mere tensors"  
Paraphrase: The lecture argues that KV cache has data-management properties: it can be stored, moved, reused, compressed, and integrated with storage systems.  
Evidence grade: A  
Technical sensitivity: data management | serving  
Conditions:
  model: LLM serving with KV-cache reuse/offload
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Phrase as a systems framing, not a formal database abstraction.
