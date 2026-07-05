# Source Card: llmsys-16-pipeline-chunking

Source ID: llmsys-16-pipeline-chunking  
Title: Model Parallel Training  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Lei Li  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-16-model-parallel-83b41612547620ee0e172caa1ee448ed.pdf`  
Pages/slides: Further Improving Pipeline Parallel by Chunking section  
Claim supported: Pipeline stages can be interleaved by chunking model layers so a device owns multiple chunks at different positions in the pipeline.  
Exact quote: "Chunk 1 Layers"  
Paraphrase: The lecture shows layer chunks assigned across devices to create smaller computation chunks and interleaved pipeline stages.  
Evidence grade: A  
Technical risk: Medium; detailed Megatron-LM claims need the Megatron paper.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Optional sidebar card.
