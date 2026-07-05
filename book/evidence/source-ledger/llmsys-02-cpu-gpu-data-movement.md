# Source Card: llmsys-02-cpu-gpu-data-movement

Source ID: llmsys-02-cpu-gpu-data-movement  
Title: GPU Programming  
Author/issuer: CMU 11868/11968 LLM Systems course staff  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-02-gpu-programming-c64a0141b96a1f384db7f6717ed8e039.pdf`  
Pages/slides: CPU-GPU data movement section  
Claim supported: CPU-GPU transfers are much slower than on-device GPU memory access and therefore shape performance.  
Exact quote: "PCIe"; "GPU Memory"; "System Memory"  
Paraphrase: The source contrasts host-device transfer bandwidth with GPU memory bandwidth to make data movement visible.  
Evidence grade: A  
Technical risk: Medium; exact numbers are device and bus dependent.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Use for the rule that moving data can dominate naive GPU programs.
