# Source Card: llmsys-16-one-f-one-b

Source ID: llmsys-16-one-f-one-b  
Title: Model Parallel Training  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Lei Li  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-16-model-parallel-83b41612547620ee0e172caa1ee448ed.pdf`  
Pages/slides: Improving Pipeline Parallel with 1F1B flush section  
Claim supported: 1F1B-style schedules start backward as soon as possible to reduce activation memory from in-flight micro-batches.  
Exact quote: "start backward as soon as possible"  
Paraphrase: The lecture presents PipeDream-Flush 1F1B as a pipeline schedule that reduces the number of completed-forward/not-yet-backward micro-batches.  
Evidence grade: A  
Technical risk: Medium; exact in-flight counts depend on schedule assumptions.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use for scheduling intuition; cite PipeDream paper before detailed claims.
