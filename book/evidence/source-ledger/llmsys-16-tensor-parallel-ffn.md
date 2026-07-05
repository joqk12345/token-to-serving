# Source Card: llmsys-16-tensor-parallel-ffn

Source ID: llmsys-16-tensor-parallel-ffn  
Title: Model Parallel Training  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Lei Li  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-16-model-parallel-83b41612547620ee0e172caa1ee448ed.pdf`  
Pages/slides: Tensor Parallelism for FFN sections  
Claim supported: Transformer FFN tensor parallelism can split the first weight matrix by columns and the second by rows, avoiding all-reduce for the intermediate activation while requiring combination for the final output.  
Exact quote: "All-reduce is not needed (for Y)"  
Paraphrase: The lecture shows an FFN split where each GPU computes a shard of the GeLU activation and the second projection sums shard contributions.  
Evidence grade: A  
Technical risk: Medium; notation and communication placement should be technically reviewed.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Central Transformer tensor-parallel example.
