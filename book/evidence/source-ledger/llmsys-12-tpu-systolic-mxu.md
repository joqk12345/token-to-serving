# Source Card: llmsys-12-tpu-systolic-mxu

Source ID: llmsys-12-tpu-systolic-mxu  
Title: Introduction to JAX / XLA / TPU  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Google slide source  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-12-Introduction_to_JAX_XLA_TPU-f0450caf9e7e6707c009f7f77997a2be.pdf`  
Pages/slides: TPU tensor core and MXU sections  
Claim supported: TPU matrix units use systolic-array-style dataflow where operands move through a grid and matrix multiply work is organized around that hardware structure.  
Exact quote: "systolic array"  
Paraphrase: The lecture uses MXU and systolic-array diagrams to explain why TPU compilation is tightly coupled to tiling, layout, and scheduled data movement.  
Evidence grade: A  
Technical risk: Medium; hardware generation details need official documentation before publication.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use for mechanism, not for unsupported TPU spec claims.
