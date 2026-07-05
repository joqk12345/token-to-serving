# Source Card: llmsys-27-kv-cache-compression

Source ID: llmsys-27-kv-cache-compression  
Title: KV Cache  
Author/issuer: Junchen Jiang / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-27-LMCache_junchenjiang-c21828fe582270cb5e08b4a21a002956.pdf`  
Pages / slides / sections: KV cache compression workflow  
Claim supported: KV-cache compression is presented as a way to store more KV cache and load/transfer it more efficiently.  
Exact quote: "KV cache compression"  
Paraphrase: The lecture treats compression as a storage and transfer optimization for KV cache, not only as model-parameter compression.  
Evidence grade: A  
Technical sensitivity: compression | storage | transfer  
Conditions:
  model: LLM serving with KV-cache storage/transfer
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: numbers require experiment context  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Do not use compression-ratio or transfer-speed numbers without setup.
