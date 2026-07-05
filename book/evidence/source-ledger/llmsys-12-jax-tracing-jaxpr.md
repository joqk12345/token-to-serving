# Source Card: llmsys-12-jax-tracing-jaxpr

Source ID: llmsys-12-jax-tracing-jaxpr  
Title: Introduction to JAX / XLA / TPU  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Google slide source  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-12-Introduction_to_JAX_XLA_TPU-f0450caf9e7e6707c009f7f77997a2be.pdf`  
Pages/slides: JAX tracing and JAX lowering sections  
Claim supported: JAX traces Python/JAX functions by abstract execution into a JAXpr made of primitive operations.  
Exact quote: "No actual math"  
Paraphrase: The source explains that tracing captures operation structure into JAXpr, which becomes the graph-like representation the compiler can optimize and lower.  
Evidence grade: A  
Technical risk: Medium; draft should distinguish tracing from ordinary eager execution carefully.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Central for explaining staged programming.
