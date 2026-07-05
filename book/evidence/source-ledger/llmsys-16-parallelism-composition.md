# Source Card: llmsys-16-parallelism-composition

Source ID: llmsys-16-parallelism-composition  
Title: Model Parallel Training  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Lei Li  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-16-model-parallel-83b41612547620ee0e172caa1ee448ed.pdf`  
Pages/slides: Combination of Pipeline and Tensor Model Parallelism; Model Parallel + Data Parallel sections  
Claim supported: Tensor parallelism, pipeline parallelism, and data parallelism can be combined; tensor parallelism is often used within a server and pipeline/data parallelism can scale beyond that.  
Exact quote: "Combination of Pipeline and Tensor"  
Paraphrase: The lecture gives takeaways for combining tensor model parallelism, pipeline model parallelism, and data parallelism based on model fit and cluster configuration.  
Evidence grade: A  
Technical risk: Medium; "generally" guidance depends on hardware and model architecture.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use as qualitative composition guidance.
