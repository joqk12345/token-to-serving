# Source Note: llmsys-12 Introduction to JAX / XLA / TPU

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-12-Introduction_to_JAX_XLA_TPU-f0450caf9e7e6707c009f7f77997a2be.pdf`

## Scope

This source covers the JAX AI stack, JAX transformations, Flax NNX and Optax examples, sharding, TPU architecture, XLA compilation, StableHLO/HLO, TPU memory/layout decisions, attention fusion, and compiler-inserted distributed collectives.

## Key Claims

- JAX exposes NumPy-like array programming plus transformations such as `jit`, `grad`, and `vmap`; those transformations compile, differentiate, and vectorize array programs.
- JAX traces Python/JAX functions into JAXpr, lowers them to StableHLO and HLO, and then uses XLA backend passes to produce target-specific executables.
- StableHLO is presented as a hardware-agnostic intermediate representation that multiple ML frameworks can lower into.
- HLO requires array dimensions known at compile time, which lets the compiler reason about memory layout and allocation.
- XLA optimization includes graph-level simplification, operator fusion, layout/tiling, buffer/copy insertion, and memory-space assignment.
- TPU execution is compiler-scheduled around systolic matrix units, vector/scalar units, memory hierarchy, and VLIW bundles.
- JAX sharding specifications can be lowered into SPMD programs where XLA injects collectives such as all-gather.

## Chapter 7 Use

- Use this as the main source for the “staged array program” explanation: Python function → traced JAXpr → StableHLO/HLO → optimized executable.
- Use TPU material to explain why compiler IR carries shapes, layouts, tiling, memory spaces, and scheduling information.
- Use sharding examples to connect Chapter 7 to later distributed-training chapters without teaching all of DDP/model parallelism yet.
- Use attention-fusion examples cautiously: they motivate compiler fusion, but Chapter 6 already owns FlashAttention and exact attention algorithms.

## Do Not Use As

- A final source for TPU Ironwood hardware specifications without official Google TPU documentation.
- A benchmark source unless every workload, shape, precision, and hardware condition is carried with the number.
- A replacement for formal XLA, StableHLO, or JAX official documentation when exact API/compiler semantics matter.

## Candidate Source Cards

- `llmsys-12-jax-transformations`
- `llmsys-12-jax-tracing-jaxpr`
- `llmsys-12-stablehlo-portability`
- `llmsys-12-hlo-static-shapes`
- `llmsys-12-xla-compilation-pipeline`
- `llmsys-12-xla-optimization-passes`
- `llmsys-12-tpu-systolic-mxu`
- `llmsys-12-xla-attention-fusion`
- `llmsys-12-jax-sharding-spmd`

Owner: Technical Researcher  
Purpose: Chapter 7 source extraction  
Evidence grade: A for course framing; official docs needed for exact public API guarantees  
Assumptions: Chapter 7 will use JAX/XLA/TPU as a compiler-stack case study, not as a claim that all LLM systems should use JAX  
Open questions: Which hardware facts should be cross-checked against official TPU documentation before draft  
Handoff: Book Architect for Chapter 7 brief
