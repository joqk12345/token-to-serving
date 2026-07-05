# Source Card: llmsys-04-coalesced-access

Source ID: llmsys-04-coalesced-access  
Title: GPU Acceleration  
Author/issuer: CMU 11868/11968 LLM Systems course staff  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-04-gpu-acceleration-48ffa5768ba62c54138f0a71ca2b68b8.pdf`  
Pages/slides: locality, coalesced memory access, and transpose sections  
Claim supported: Consecutive memory accesses in a warp can be coalesced, while strided or scattered accesses can require more memory transactions.  
Exact quote: "Consecutive memory accesses in a warp are coalesced together."  
Paraphrase: The source uses coalesced and uncoalesced access diagrams plus matrix transpose to show why memory layout matters.  
Evidence grade: A  
Technical risk: Low for concept; exact transaction behavior is architecture-specific.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Use for memory-layout section.
