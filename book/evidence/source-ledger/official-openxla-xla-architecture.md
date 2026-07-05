# Source Card: official-openxla-xla-architecture

Source ID: official-openxla-xla-architecture  
Title: XLA architecture  
Author/issuer: OpenXLA Project  
Date: Last updated 2024-01-10 UTC; accessed 2026-07-05  
Source type: official-docs  
File path / URL: `https://openxla.org/xla/architecture`  
Pages / slides / sections: Objectives and How it works  
Claim supported: XLA compiles StableHLO model graphs into target-optimized executables using target-independent optimization, backend-specific optimization, and target-specific code generation.  
Exact quote: "target-optimized executable"  
Paraphrase: The official XLA architecture page describes objectives including speed, memory use, custom-op reduction, and portability; it outlines a pipeline from StableHLO graph to HLO optimization, backend optimization, and code generation.  
Evidence grade: A  
Technical sensitivity: implementation  
Conditions:
  model:
  hardware: CPU/GPU and alternative backends discussed generally
  batch size:
  sequence length:
  precision:
  software version: XLA architecture page last updated 2024-01-10 UTC
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Use for Chapter 7 compiler pipeline claims; TPU-specific LLO/VLIW claims still need TPU-specific sources.
