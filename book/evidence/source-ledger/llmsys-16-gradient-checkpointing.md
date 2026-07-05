# Source Card: llmsys-16-gradient-checkpointing

Source ID: llmsys-16-gradient-checkpointing  
Title: Model Parallel Training  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Lei Li  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-16-model-parallel-83b41612547620ee0e172caa1ee448ed.pdf`  
Pages/slides: Reduce PP Memory Cost and Gradient Checkpointing sections  
Claim supported: Gradient checkpointing/rematerialization trades extra forward recomputation during backward for reduced activation storage.  
Exact quote: "recomputes"  
Paraphrase: The lecture describes storing fewer activations and recomputing forward segments during backward to reduce memory pressure.  
Evidence grade: A  
Technical risk: Medium; asymptotic memory formulas should be checked against Chen et al. before draft use.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use conceptually unless primary checkpointing paper is extracted.
