# Source Card: llmsys-28-mooncake-store-integration

Source ID: llmsys-28-mooncake-store-integration  
Title: LLM Serving on Heterogeneous Hardware  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-28-mooncake-kTransformer-1243bfbecbb0c3610bf95eac030acb2a.pdf`  
Pages / slides / sections: Mooncake Store: External Integration  
Claim supported: Mooncake Store is presented as integrating with inference engines through object put/get and batch transfer APIs over local memory, remote memory, SSD, and third-party storage.  
Exact quote: "BatchTransfer API"  
Paraphrase: The lecture shows Mooncake Store as an external KV-cache storage/transfer layer connecting engines to multiple memory and storage resources.  
Evidence grade: A  
Technical sensitivity: storage interface | transfer  
Conditions:
  model: LLM serving with external KV-cache store
  hardware: local memory, remote memory, remote SSD, third-party storage
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Check Mooncake docs/source before publishing exact API behavior.
