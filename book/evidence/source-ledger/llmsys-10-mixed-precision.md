# Source Card: llmsys-10-mixed-precision

Source ID: llmsys-10-mixed-precision  
Title: Accelerating Transformer Training and Inference  
Author/issuer: CMU 11868/11968 LLM Systems course staff  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-10-transformer-acc-5ba466406bf7296f86cd244ad0405867.pdf`  
Pages/slides: mixed-precision calculation and update sections  
Claim supported: Mixed precision can reduce storage and transfer cost while some optimizer updates still need FP32 calculation or state.  
Exact quote: "Forward / Backward could use FP16 or FP8"; "Gradient update in Optimizer ... needs FP32"  
Paraphrase: The source frames mixed precision as both a compute and memory optimization, with FP32 retained for trainer updates.  
Evidence grade: A  
Technical risk: Medium; exact precision choices vary by model, hardware, and training recipe.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Keep claims conditional.
