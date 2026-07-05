# Source Card: llmsys-27-lmcache-separated-service

Source ID: llmsys-27-lmcache-separated-service  
Title: KV Cache  
Author/issuer: Junchen Jiang / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-27-LMCache_junchenjiang-c21828fe582270cb5e08b4a21a002956.pdf`  
Pages / slides / sections: LMCache separates KV cache from inference  
Claim supported: LMCache separates KV-cache management from inference engines by running as a separate KV-cache management service.  
Exact quote: "separates KV cache from inference"  
Paraphrase: The lecture contrasts in-process KV libraries with LMCache as a service/process that manages KV cache outside individual vLLM/SGL processes.  
Evidence grade: A  
Technical sensitivity: serving architecture | KV management  
Conditions:
  model: LLM inference engines using external KV cache management
  hardware:
  batch size:
  sequence length:
  precision:
  software version: lecture status as of 2026
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: For current support claims, check LMCache docs/source.
