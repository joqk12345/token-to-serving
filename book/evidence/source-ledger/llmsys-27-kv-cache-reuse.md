# Source Card: llmsys-27-kv-cache-reuse

Source ID: llmsys-27-kv-cache-reuse  
Title: KV Cache  
Author/issuer: Junchen Jiang / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-27-LMCache_junchenjiang-c21828fe582270cb5e08b4a21a002956.pdf`  
Pages / slides / sections: How KV cache avoids repeated computation  
Claim supported: KV cache avoids repeated computation by storing reusable attention state.  
Exact quote: "avoids repeated computation"  
Paraphrase: The lecture frames KV cache as reusable serving state that can reduce repeated prefill or attention work when requests share context.  
Evidence grade: A  
Technical sensitivity: KV cache | inference  
Conditions:
  model: Transformer inference with reusable KV cache
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use as qualitative support; cache-hit economics require workload conditions.
