# Source Card: llmsys-13-pallas-vmem-constraint

Source ID: llmsys-13-pallas-vmem-constraint  
Title: Pallas Kernels Splash Attention  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Google slide source  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-13-pallas_splash_attention_srinath_mandalapu-b0bc7990950b84561ff9aa8e1791f727.pdf`  
Pages/slides: VMEM bottleneck and tiling sections  
Claim supported: Pallas kernels must tile LLM-sized tensors because loading whole tensors into VMEM can exceed VMEM capacity.  
Exact quote: "Ran out of memory"  
Paraphrase: The source shows a VMEM resource-exhaustion example for a large tensor and motivates BlockSpec tiling to fit chunks into local memory.  
Evidence grade: A  
Technical risk: Low for concept; exact VMEM capacity is hardware dependent.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Useful concrete bottleneck for Pallas section.
