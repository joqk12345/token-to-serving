# Source Card: llmsys-10-transformer-operator-stack

Source ID: llmsys-10-transformer-operator-stack  
Title: Accelerating Transformer Training and Inference  
Author/issuer: CMU 11868/11968 LLM Systems course staff  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-10-transformer-acc-5ba466406bf7296f86cd244ad0405867.pdf`  
Pages/slides: multi-head attention and feed-forward network, non-GEMM fusion sections  
Claim supported: Transformer blocks include GEMM-heavy components and non-GEMM operators such as bias, dropout, residual, LayerNorm, softmax, and cross entropy.  
Exact quote: "cuBLAS GEMM"; "Custom Elementwise"; "Custom Reduce"  
Paraphrase: The source separates GEMM work from custom elementwise and reduction kernels in a Transformer stack.  
Evidence grade: A  
Technical risk: Low for operator taxonomy.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Use to frame why cuBLAS is necessary but not sufficient.
