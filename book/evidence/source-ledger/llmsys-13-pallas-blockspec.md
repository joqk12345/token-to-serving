# Source Card: llmsys-13-pallas-blockspec

Source ID: llmsys-13-pallas-blockspec  
Title: Pallas Kernels Splash Attention  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Google slide source  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-13-pallas_splash_attention_srinath_mandalapu-b0bc7990950b84561ff9aa8e1791f727.pdf`  
Pages/slides: BlockSpec and block computation sections  
Claim supported: `BlockSpec` partitions global tensors into blocks and maps each grid coordinate to the HBM slice copied into VMEM for a kernel invocation.  
Exact quote: "BlockSpec partitions"  
Paraphrase: The lecture explains that the Pallas grid defines iteration space while BlockSpec defines per-program data slices.  
Evidence grade: A  
Technical risk: Low for conceptual explanation; API details need official docs.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Central card for Pallas programming model.
