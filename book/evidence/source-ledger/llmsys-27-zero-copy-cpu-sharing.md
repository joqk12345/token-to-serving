# Source Card: llmsys-27-zero-copy-cpu-sharing

Source ID: llmsys-27-zero-copy-cpu-sharing  
Title: KV Cache  
Author/issuer: Junchen Jiang / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-27-LMCache_junchenjiang-c21828fe582270cb5e08b4a21a002956.pdf`  
Pages / slides / sections: Zero-copy CPU sharing  
Claim supported: LMCache is presented as using a multi-process CPU pool and remote KV pool to reduce extra copies when sharing KV cache across GPU workers.  
Exact quote: "0 Extra copies"  
Paraphrase: The lecture illustrates moving KV cache through LMCache-managed CPU memory to avoid redundant CPU-side copies between GPU workers.  
Evidence grade: A  
Technical sensitivity: memory transfer | KV cache  
Conditions:
  model: LLM serving with CPU KV-cache sharing
  hardware: GPU workers plus CPU/remote KV pool
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use as a conceptual data-movement example; implementation details need source/docs.
