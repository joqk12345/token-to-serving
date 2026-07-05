# Source Card: llmsys-13-pallas-pipelining

Source ID: llmsys-13-pallas-pipelining  
Title: Pallas Kernels Splash Attention  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Google slide source  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-13-pallas_splash_attention_srinath_mandalapu-b0bc7990950b84561ff9aa8e1791f727.pdf`  
Pages/slides: Pallas pipelining API sections  
Claim supported: Pallas can overlap HBM-to-VMEM data transfer with active computation through software pipelining.  
Exact quote: "overlaps HBM"  
Paraphrase: The source describes pipelined scheduling where the next tile is prefetched from HBM while the current tile is computed in VMEM.  
Evidence grade: A  
Technical risk: Medium; measured benefit depends on tile shape, workload, compiler, and hardware.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use conditions with any performance claim.
