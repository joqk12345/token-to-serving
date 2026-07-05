# Source Card: llmsys-26-nixl-transfer-layer

Source ID: llmsys-26-nixl-transfer-layer  
Title: Inference at Scale: Opportunities and Challenges  
Author/issuer: Vikram Sharma Mailthody / CMU LLM Systems  
Date: 2025  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-26-dynamo-vikram_mailthody-0ddbeb69382d5c168b6d4636a82185d0.pdf`  
Pages / slides / sections: NIXL  
Claim supported: NIXL is presented as a cross-node and cross-memory buffer-list transfer layer with northbound API and southbound backend API.  
Exact quote: "cross-node & cross-mem"  
Paraphrase: The lecture uses NIXL as a data-transfer substrate for moving KV and other buffers across memory domains and nodes.  
Evidence grade: A  
Technical sensitivity: data transfer | distributed serving  
Conditions:
  model:
  hardware: cross-node/cross-memory transfer setting
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Add official NIXL documentation/source cards before describing API details beyond the lecture.
