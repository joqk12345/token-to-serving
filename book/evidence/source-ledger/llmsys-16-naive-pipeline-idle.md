# Source Card: llmsys-16-naive-pipeline-idle

Source ID: llmsys-16-naive-pipeline-idle  
Title: Model Parallel Training  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Lei Li  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-16-model-parallel-83b41612547620ee0e172caa1ee448ed.pdf`  
Pages/slides: Limitations of Naïve Pipeline Parallel section  
Claim supported: Naive pipeline parallelism can have low utilization because only one device works at a time and computation/communication are not interleaved.  
Exact quote: "only one device is working"  
Paraphrase: The lecture identifies idle devices, no computation/communication interleaving, and high activation memory as limitations of naive pipeline parallelism.  
Evidence grade: A  
Technical risk: Low for conceptual schedule; exact utilization depends on schedule and workload.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use to create the chapter's first bottleneck.
