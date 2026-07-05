# Source Card: llmsys-27-storage-plugin-interface

Source ID: llmsys-27-storage-plugin-interface  
Title: KV Cache  
Author/issuer: Junchen Jiang / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-27-LMCache_junchenjiang-c21828fe582270cb5e08b4a21a002956.pdf`  
Pages / slides / sections: LMCache hooks / storage plugins  
Claim supported: LMCache exposes hooks and storage plugins for get/put KV-cache operations.  
Exact quote: "Get/Put KV cache"  
Paraphrase: The lecture presents LMCache as an integration layer between inference engines and storage/transfer backends.  
Evidence grade: A  
Technical sensitivity: software interface | storage  
Conditions:
  model: LLM serving with external KV-cache storage
  hardware:
  batch size:
  sequence length:
  precision:
  software version: lecture status as of 2026
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Check official docs/source before publishing exact API names.
