# Source Card: llmsys-13-pallas-memory-hierarchy

Source ID: llmsys-13-pallas-memory-hierarchy  
Title: Pallas Kernels Splash Attention  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Google slide source  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-13-pallas_splash_attention_srinath_mandalapu-b0bc7990950b84561ff9aa8e1791f727.pdf`  
Pages/slides: Pallas fundamentals and HBM vs VMEM sections  
Claim supported: Pallas exposes TPU memory spaces such as HBM, VMEM, SMEM, and semaphore memory so kernels can explicitly organize data movement.  
Exact quote: "HBM vs. VMEM"  
Paraphrase: The source presents Pallas memory-space enums and explains VMEM as smaller but faster than HBM for kernel-local work.  
Evidence grade: A  
Technical risk: Medium; exact memory-space semantics should be checked against Pallas documentation/source.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use to explain why automatic compiler lowering sometimes needs a kernel-level interface.
