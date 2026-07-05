# Source Card: llmsys-26-memory-tiers-kv-offload

Source ID: llmsys-26-memory-tiers-kv-offload  
Title: Inference at Scale: Opportunities and Challenges  
Author/issuer: Vikram Sharma Mailthody / CMU LLM Systems  
Date: 2025  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-26-dynamo-vikram_mailthody-0ddbeb69382d5c168b6d4636a82185d0.pdf`  
Pages / slides / sections: Memory Management  
Claim supported: Distributed serving can leverage multiple memory/storage tiers such as HBM, host memory, local SSD, and network storage for KV offload.  
Exact quote: "Host memory"  
Paraphrase: The lecture frames KV management as using the datacenter memory hierarchy, not only GPU HBM.  
Evidence grade: A  
Technical sensitivity: memory hierarchy | KV offload  
Conditions:
  model: LLM serving with KV-cache offload
  hardware: GPU HBM, host memory, local SSD, network storage
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Do not use the slide's TTFT improvement number without setup.
