# Source Card: llmsys-16-pytorch-pipelining

Source ID: llmsys-16-pytorch-pipelining  
Title: Model Parallel Training  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Lei Li  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-16-model-parallel-83b41612547620ee0e172caa1ee448ed.pdf`  
Pages/slides: Pipeline Parallelism in PyTorch section  
Claim supported: PyTorch pipeline parallelism is presented as building pipeline stages and using a pipeline schedule, with examples such as `PipelineStage` and `ScheduleGPipe`.  
Exact quote: "PipelineStage"  
Paraphrase: The lecture shows PyTorch distributed pipelining examples that split a model into stages and run a schedule over micro-batches.  
Evidence grade: A  
Technical risk: Medium; official PyTorch docs needed for current API claims.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Keep illustrative unless official docs are added.
