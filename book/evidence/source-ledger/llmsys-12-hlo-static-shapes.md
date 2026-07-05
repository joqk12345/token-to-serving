# Source Card: llmsys-12-hlo-static-shapes

Source ID: llmsys-12-hlo-static-shapes  
Title: Introduction to JAX / XLA / TPU  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Google slide source  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-12-Introduction_to_JAX_XLA_TPU-f0450caf9e7e6707c009f7f77997a2be.pdf`  
Pages/slides: HLO and XLA shapes/memory layout sections  
Claim supported: HLO works with array dimensions known at compile time, which lets the compiler determine memory use and layouts ahead of execution.  
Exact quote: "known at compile time"  
Paraphrase: The source ties static shape information to compile-time memory planning and efficient memory layout.  
Evidence grade: A  
Technical risk: Medium; modern dynamic-shape support should be checked before writing broad claims.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: State as course framing and avoid overclaiming all XLA programs are fully static.
