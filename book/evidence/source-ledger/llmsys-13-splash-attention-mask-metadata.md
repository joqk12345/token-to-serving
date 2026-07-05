# Source Card: llmsys-13-splash-attention-mask-metadata

Source ID: llmsys-13-splash-attention-mask-metadata  
Title: Pallas Kernels Splash Attention  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Google slide source  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-13-pallas_splash_attention_srinath_mandalapu-b0bc7990950b84561ff9aa8e1791f727.pdf`  
Pages/slides: Splash Attention mask metadata and Pallas call sections  
Claim supported: Splash Attention uses mask metadata such as active rows/columns and block classification to map sparse attention work to kernel grid coordinates.  
Exact quote: "active_rows/cols"  
Paraphrase: The source describes block filtering and coordinate mapping metadata that lets the kernel locate active Q/KV blocks and avoid unnecessary work.  
Evidence grade: A  
Technical risk: Medium; source-code review needed before presenting exact data structures as stable API.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use if Chapter 7 includes a sparse-kernel case study.
