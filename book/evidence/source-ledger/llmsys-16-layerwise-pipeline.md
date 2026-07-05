# Source Card: llmsys-16-layerwise-pipeline

Source ID: llmsys-16-layerwise-pipeline  
Title: Model Parallel Training  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Lei Li  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-16-model-parallel-83b41612547620ee0e172caa1ee448ed.pdf`  
Pages/slides: Model Parallel Training and Pipeline Parallelism sections  
Claim supported: Pipeline parallelism partitions a model across devices by layers, with each device responsible for forward/backward computation for its assigned layers.  
Exact quote: "distributed across multiple GPUs over layers"  
Paraphrase: The source shows layers placed on different devices and activations/gradients passed between adjacent stages.  
Evidence grade: A  
Technical risk: Low.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Basic pipeline-parallel definition.
