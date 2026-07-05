# Source Card: llmsys-13-splash-attention-sparse-flash

Source ID: llmsys-13-splash-attention-sparse-flash  
Title: Pallas Kernels Splash Attention  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Google slide source  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-13-pallas_splash_attention_srinath_mandalapu-b0bc7990950b84561ff9aa8e1791f727.pdf`  
Pages/slides: Splash Attention sections  
Claim supported: Splash Attention combines FlashAttention-style tiling/fusion with sparse block processing so attention work can skip blocks that are not needed by the mask.  
Exact quote: "Sparse + Flash"  
Paraphrase: The source frames Splash Attention as dense attention transformed into sparse block execution with hardware-aware kernels and tiling.  
Evidence grade: A  
Technical risk: Medium; algorithmic details and implementation claims should be cross-checked against Tokamax source or paper/docs.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Candidate sidebar; avoid duplicating Chapter 6.
