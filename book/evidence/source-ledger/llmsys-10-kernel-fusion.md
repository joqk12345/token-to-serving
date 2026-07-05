# Source Card: llmsys-10-kernel-fusion

Source ID: llmsys-10-kernel-fusion  
Title: Accelerating Transformer Training and Inference  
Author/issuer: CMU 11868/11968 LLM Systems course staff  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-10-transformer-acc-5ba466406bf7296f86cd244ad0405867.pdf`  
Pages/slides: kernel fusion and non-GEMM operator sections  
Claim supported: Kernel fusion reduces launch overhead and avoids extra intermediate memory loads and stores.  
Exact quote: "reduce overhead"; "reduce extra memory access"  
Paraphrase: The source contrasts two matrix-add kernels with one fused three-input add kernel, then applies fusion to Transformer operators.  
Evidence grade: A  
Technical risk: Low for concept; performance magnitude is workload-dependent.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Core Chapter 5 Transformer optimization card.
