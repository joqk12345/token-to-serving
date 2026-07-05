# Source Card: llmsys-02-warp-execution

Source ID: llmsys-02-warp-execution  
Title: GPU Programming  
Author/issuer: CMU 11868/11968 LLM Systems course staff  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-02-gpu-programming-c64a0141b96a1f384db7f6717ed8e039.pdf`  
Pages/slides: SIMT, kernel threads, and warp execution sections  
Claim supported: CUDA threads are grouped into blocks and grids, and SMs execute threads in warps.  
Exact quote: "Kernel executed as Grid of Blocks of Threads"; "Each warp contains 32 threads"  
Paraphrase: The source presents SIMT execution, thread blocks, grids, and warps as the execution hierarchy.  
Evidence grade: A  
Technical risk: Low for NVIDIA CUDA; avoid implying all accelerators use identical warp semantics.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Use to explain why thread indexing and branch behavior matter.
