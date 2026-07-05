# Source Card: llmsys-13-pallas-output-aliasing

Source ID: llmsys-13-pallas-output-aliasing  
Title: Pallas Kernels Splash Attention  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Google slide source  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-13-pallas_splash_attention_srinath_mandalapu-b0bc7990950b84561ff9aa8e1791f727.pdf`  
Pages/slides: Pallas output aliasing section  
Claim supported: Pallas output aliasing can reuse an input buffer for output, reducing allocation and data movement for in-place-style updates.  
Exact quote: "Input-Output Aliasing"  
Paraphrase: The lecture shows aliasing as a way for a kernel output to overwrite or reuse an existing buffer rather than allocate a separate output buffer.  
Evidence grade: A  
Technical risk: Medium; correctness depends on aliasing constraints and API semantics.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Good example for framework/runtime memory management.
