# Source Card: llmsys-12-xla-compilation-pipeline

Source ID: llmsys-12-xla-compilation-pipeline  
Title: Introduction to JAX / XLA / TPU  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Google slide source  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-12-Introduction_to_JAX_XLA_TPU-f0450caf9e7e6707c009f7f77997a2be.pdf`  
Pages/slides: JAX execution lifecycle and TPU compilation pipeline sections  
Claim supported: A JAX `jit` path can be explained as tracing to JAXpr, lowering to StableHLO/HLO, optimizing HLO, lowering to target-specific IR, scheduling, and producing an executable.  
Exact quote: "Jaxpr → StableHLO → HLO"  
Paraphrase: The source lays out a compilation path from traced JAX expressions through hardware-independent and hardware-specific compiler stages to a loaded executable.  
Evidence grade: A  
Technical risk: Medium; backend-specific names such as LLO/VLIW should be qualified to TPU discussion.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Core pipeline card for Chapter 7 figure.
