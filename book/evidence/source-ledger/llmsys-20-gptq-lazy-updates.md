# Source Card: llmsys-20-gptq-lazy-updates

Source ID: llmsys-20-gptq-lazy-updates  
Title: LLM Quantization - GPTQ  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-20-quantization2-ba573d7e5d82e68027bbd3a92c3cd819.pdf`  
Pages / slides / sections: Slides 14, 20  
Claim supported: GPTQ batches and delays updates to remaining columns to improve practical efficiency.  
Exact quote: "lazy-fashion"  
Paraphrase: GPTQ improves GPU practicality by batching compensation updates rather than updating every future weight immediately.  
Evidence grade: A  
Technical sensitivity: performance mechanism  
Conditions:
  model: Transformer linear weights
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Do not quote speedup numbers without setup.
