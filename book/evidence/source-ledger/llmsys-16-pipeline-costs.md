# Source Card: llmsys-16-pipeline-costs

Source ID: llmsys-16-pipeline-costs  
Title: Model Parallel Training  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Lei Li  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-16-model-parallel-83b41612547620ee0e172caa1ee448ed.pdf`  
Pages/slides: Microbatch Pipelining and Limitations of Pipeline Parallel sections  
Claim supported: Pipeline parallelism introduces bubble overhead, communication overhead at partition boundaries, and activation-memory pressure from in-flight micro-batches.  
Exact quote: "Bubble overhead"  
Paraphrase: The lecture lists pipeline bubble overhead, partition-boundary activation transfer, and peak activation-memory concerns.  
Evidence grade: A  
Technical risk: Medium; formulas need review and conditions before use.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Avoid quoting formulas without verifying notation.
