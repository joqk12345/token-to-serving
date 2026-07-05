# Source Card: llmsys-13-pallas-tile-size-tuning

Source ID: llmsys-13-pallas-tile-size-tuning  
Title: Pallas Kernels Splash Attention  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Google slide source  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-13-pallas_splash_attention_srinath_mandalapu-b0bc7990950b84561ff9aa8e1791f727.pdf`  
Pages/slides: tuning Pallas kernels and matmul visualization sections  
Claim supported: Pallas tile size and grid shape affect invocation count, instruction overhead, and measured throughput for a given matmul workload.  
Exact quote: "Tile Size"  
Paraphrase: The lecture reports a tile-size tuning example for a `(4096, 7168) × (7168, 18432)` matmul and emphasizes minimizing instruction overhead while fitting the memory hierarchy.  
Evidence grade: A  
Technical risk: High for numeric speedups; include full shape, device, precision, tile size, and measurement context before using numbers.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Conditions travel with any performance claim.
