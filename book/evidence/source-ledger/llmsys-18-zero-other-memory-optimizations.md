# Source Card: llmsys-18-zero-other-memory-optimizations

Source ID: llmsys-18-zero-other-memory-optimizations  
Title: Memory Optimization in Distributed Training  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2025  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-18-zero-20eb6c8d8c1e7092e1b922abf03d8cdd.pdf`  
Pages / slides / sections: Slides 67-70  
Claim supported: Additional memory optimizations include partitioned activation checkpointing, constant-size buffers, memory defragmentation, memory reuse, and communication reduction.  
Exact quote: "Constant Size Buffers"  
Paraphrase: ZeRO-style systems also manage activations, buffers, fragmentation, and communication rather than only partitioning model state.  
Evidence grade: A  
Technical sensitivity: memory management  
Conditions:
  model: distributed training
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Treat ZeRO++ and LightSeq as pointers unless original source cards are added.
