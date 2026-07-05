# Source Card: llmsys-16-tensor-parallel-attention

Source ID: llmsys-16-tensor-parallel-attention  
Title: Model Parallel Training  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Lei Li  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-16-model-parallel-83b41612547620ee0e172caa1ee448ed.pdf`  
Pages/slides: Tensor Parallelism for Self-Attention section  
Claim supported: Self-attention tensor parallelism can split weights by columns/heads, allowing head-local computation without all-reduce for the attention heads.  
Exact quote: "Split weights over columns (heads)"  
Paraphrase: The lecture presents attention head partitioning as a tensor-parallel split with no all-reduce needed for the head-local stage.  
Evidence grade: A  
Technical risk: Medium; output projection communication should be handled carefully in draft.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Avoid overclaiming no communication for entire attention block.
