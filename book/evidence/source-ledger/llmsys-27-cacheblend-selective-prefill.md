# Source Card: llmsys-27-cacheblend-selective-prefill

Source ID: llmsys-27-cacheblend-selective-prefill  
Title: KV Cache  
Author/issuer: Junchen Jiang / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-27-LMCache_junchenjiang-c21828fe582270cb5e08b4a21a002956.pdf`  
Pages / slides / sections: KV Cache Blending / Full prefill vs. Selective prefill  
Claim supported: CacheBlend-style selective prefill reuses stored KV cache while recomputing selected tokens to recover interactions not captured by direct cache concatenation.  
Exact quote: "Selective prefill"  
Paraphrase: The lecture presents selective prefill as a way to combine reused KV cache with limited recomputation rather than fully prefilling all tokens.  
Evidence grade: A  
Technical sensitivity: KV-cache reuse | algorithm  
Conditions:
  model: RAG or mixed-context prefill reuse setting
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Avoid the slide's speedup number without experiment setup.
