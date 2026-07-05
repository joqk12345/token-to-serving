# Source Card: llmsys-04-naive-matmul-intensity

Source ID: llmsys-04-naive-matmul-intensity  
Title: GPU Acceleration  
Author/issuer: CMU 11868/11968 LLM Systems course staff  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-04-gpu-acceleration-48ffa5768ba62c54138f0a71ca2b68b8.pdf`  
Pages/slides: simple matrix multiplication and peak computation sections  
Claim supported: A naive matrix multiplication kernel can have low arithmetic intensity because each inner-loop operation reads operands from global memory.  
Exact quote: "compute-to-globalmemory-access: 2 / (2 * 4) = 0.25 FLOP/B"  
Paraphrase: The source calculates a simple FLOP-per-byte ratio for naive matrix multiplication to motivate memory-bound behavior.  
Evidence grade: A  
Technical risk: Medium; ratio depends on simplified assumptions. Use as intuition, not a final performance model.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Good candidate for a small boxed derivation.
