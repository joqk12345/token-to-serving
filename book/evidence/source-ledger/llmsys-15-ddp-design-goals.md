# Source Card: llmsys-15-ddp-design-goals

Source ID: llmsys-15-ddp-design-goals  
Title: Distributed Data Parallel Training  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Lei Li  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-15-ddp-165dbe3873fac21eb8b339e64bcfee28.pdf`  
Pages/slides: Design Goal of DDP section  
Claim supported: DDP design goals include being non-intrusive to local training scripts and interceptive enough to trigger communication algorithms promptly.  
Exact quote: "Non-intrusive"  
Paraphrase: The source cites DDP goals from Li et al., emphasizing minimal user-code change while exposing optimization opportunities to the implementation.  
Evidence grade: A  
Technical risk: Medium; cite Li et al. VLDB 2020 directly if this becomes a central design-history claim.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Good setup for why DDP integrates with autograd.
