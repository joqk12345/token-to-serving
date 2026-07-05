# Source Card: llmsys-16-tensor-parallel-matmul

Source ID: llmsys-16-tensor-parallel-matmul  
Title: Model Parallel Training  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Lei Li  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-16-model-parallel-83b41612547620ee0e172caa1ee448ed.pdf`  
Pages/slides: Tensor Parallelism section  
Claim supported: Tensor parallelism splits matrix computation across multiple GPUs, producing partial outputs that may need all-gather or reduction depending on the partition.  
Exact quote: "spliting the matrix computation"  
Paraphrase: The lecture illustrates splitting a matrix multiply across two GPUs and gathering partial output chunks.  
Evidence grade: A  
Technical risk: Medium; exact communication depends on row/column partition choice.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Main tensor-parallel concept card.
