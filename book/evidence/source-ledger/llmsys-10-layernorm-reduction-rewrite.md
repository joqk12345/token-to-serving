# Source Card: llmsys-10-layernorm-reduction-rewrite

Source ID: llmsys-10-layernorm-reduction-rewrite  
Title: Accelerating Transformer Training and Inference  
Author/issuer: CMU 11868/11968 LLM Systems course staff  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-10-transformer-acc-5ba466406bf7296f86cd244ad0405867.pdf`  
Pages/slides: rewrite reduction LayerNorm sections  
Claim supported: LayerNorm reductions can be algebraically rewritten to reduce synchronization.  
Exact quote: "One thread synchronization"; "Two thread synchronizati"  
Paraphrase: The source rewrites variance computation using mean of squares minus square of mean to reduce synchronization points.  
Evidence grade: A  
Technical risk: Medium; formula should be reviewed carefully before final publication.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Use for concept; verify notation in technical review.
