---
status: brief
chapter: 7
slug: 07-dl-frameworks-and-compilers
title: Deep Learning Frameworks, JAX, XLA, and TPU
primary_sources:
  - llmsys-05-dlframework-fa0770d636572de3f7b48ccae0ba8848.pdf
  - llmsys-12-Introduction_to_JAX_XLA_TPU-f0450caf9e7e6707c009f7f77997a2be.pdf
  - llmsys-13-pallas_splash_attention_srinath_mandalapu-b0bc7990950b84561ff9aa8e1791f727.pdf
secondary_sources: []
official_docs:
  - https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html
  - https://openxla.org/stablehlo
  - https://openxla.org/xla/architecture
  - https://docs.jax.dev/en/latest/pallas/index.html
reader_level: engineer or graduate student who has read Chapters 1-6
technical_depth: intermediate-to-advanced
---

# Deep Learning Frameworks, JAX, XLA, and TPU

## Chapter Thesis

Deep learning frameworks are not just convenience APIs. They are staging systems that turn Python-level tensor code into differentiable, optimized, memory-aware, and sometimes distributed accelerator programs.

## Reader Problem

The reader can now reason about Transformer computation, GPU kernels, memory movement, and attention acceleration. The missing bridge is how a model written in Python becomes something an accelerator can actually execute: a graph, a gradient program, an optimized compiler IR, a scheduled device executable, or a custom kernel when the compiler abstraction is too high.

## System Bottleneck

Primary bottlenecks: programmability, graph capture, automatic differentiation, compile-time shape/layout reasoning, memory planning, kernel fusion, data movement across HBM/local memory, and distributed sharding/collective insertion.

Secondary bottlenecks: compiler cache behavior, hardware-specific backend lowering, and the abstraction ceiling where generated kernels are not enough.

## Source Map

| Claim                                                                                                                                              | Source card                                | Evidence grade | Notes                                                              |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ | -------------- | ------------------------------------------------------------------ |
| Frameworks represent tensor programs with computation graphs.                                                                                      | `llmsys-05-computation-graph`              | A              | Establish graph/program framing.                                   |
| Autodiff builds gradient computation from primitive operations.                                                                                    | `llmsys-05-automatic-differentiation`      | A              | Explain mechanism before JAX `grad`.                               |
| Finite differences are useful for gradient checking, not scalable training.                                                                        | `llmsys-05-gradient-checking`              | A              | Misconception sidebar.                                             |
| Framework programming models differ across dynamic, static, and functional styles.                                                                 | `llmsys-05-framework-programming-models`   | A              | Avoid current-version absolutism.                                  |
| TensorFlow-style systems separate symbolic graph construction from optimized device execution.                                                     | `llmsys-05-tensorflow-graph-execution`     | A              | Historical bridge to compiler stacks.                              |
| JAX exposes transformations such as `jit`, `grad`, and `vmap`.                                                                                     | `llmsys-12-jax-transformations`            | A              | Main JAX entry point.                                              |
| JAX official docs describe automatic differentiation, JIT compilation through OpenXLA, and vectorization.                                          | `official-jax-quickstart-transformations`  | A              | Use for API-level transformation framing.                          |
| JAX tracing captures a function into JAXpr primitives.                                                                                             | `llmsys-12-jax-tracing-jaxpr`              | A              | Core staged-programming mechanism.                                 |
| StableHLO is a hardware-agnostic IR boundary.                                                                                                      | `llmsys-12-stablehlo-portability`          | A              | Official docs should confirm final wording.                        |
| StableHLO is an official portability layer between ML frameworks and ML compilers.                                                                 | `official-openxla-stablehlo-portability`   | A              | Includes compatibility guarantee wording.                          |
| HLO/static shape information supports memory/layout planning.                                                                                      | `llmsys-12-hlo-static-shapes`              | A              | Needs dynamic-shape caveat.                                        |
| JAX/XLA compilation can be described as JAXpr → StableHLO/HLO → optimized executable.                                                              | `llmsys-12-xla-compilation-pipeline`       | A              | TPU-specific backend terms must be scoped.                         |
| XLA official architecture describes StableHLO graph compilation into optimized executables through target-independent and backend-specific passes. | `official-openxla-xla-architecture`        | A              | Use for compiler architecture, not TPU-specific microarchitecture. |
| XLA optimization includes fusion, tiling, buffer/copy insertion, and memory-space assignment.                                                      | `llmsys-12-xla-optimization-passes`        | A              | Keep examples conditional.                                         |
| TPU matrix execution is organized around systolic/MXU-style dataflow.                                                                              | `llmsys-12-tpu-systolic-mxu`               | A              | Hardware details need official cross-check.                        |
| Compiler attention fusion can reduce materialization and memory traffic under specific compiled shapes/backends.                                   | `llmsys-12-xla-attention-fusion`           | A              | High technical risk; conditions must travel.                       |
| JAX sharding can lower into SPMD execution with compiler-inserted collectives.                                                                     | `llmsys-12-jax-sharding-spmd`              | A              | Bridge to distributed training chapters.                           |
| Pallas exposes TPU memory spaces and explicit HBM/VMEM movement.                                                                                   | `llmsys-13-pallas-memory-hierarchy`        | A              | API docs/source needed for exact semantics.                        |
| Official Pallas docs describe Pallas as experimental custom GPU/TPU kernel support with fine-grained control.                                      | `official-jax-pallas-experimental`         | A              | Draft must preserve experimental-status caveat.                    |
| `BlockSpec` maps global tensors into block-local kernel inputs.                                                                                    | `llmsys-13-pallas-blockspec`               | A              | Core Pallas mechanism.                                             |
| Pallas pipelining overlaps HBM/VMEM transfers with computation.                                                                                    | `llmsys-13-pallas-pipelining`              | A              | Performance impact is workload-dependent.                          |
| Output aliasing can reduce allocation and data movement.                                                                                           | `llmsys-13-pallas-output-aliasing`         | A              | Requires correctness caveat.                                       |
| LLM-sized tensors must be tiled to fit local memory.                                                                                               | `llmsys-13-pallas-vmem-constraint`         | A              | Concrete bottleneck.                                               |
| Tile size and grid shape materially affect Pallas performance.                                                                                     | `llmsys-13-pallas-tile-size-tuning`        | A              | Do not use speedup without full context.                           |
| Splash Attention combines Flash-style tiling with sparse block execution.                                                                          | `llmsys-13-splash-attention-sparse-flash`  | A              | Sidebar candidate.                                                 |
| Sparse attention kernels need mask metadata to map active blocks to work.                                                                          | `llmsys-13-splash-attention-mask-metadata` | A              | Source-code review before API-level details.                       |

## Explanation Arc

1. Concrete problem: the researcher writes a Transformer training step in Python, but the hardware needs scheduled tensor programs, gradients, buffers, kernels, and communication.
2. Start with frameworks as graph builders: primitive tensor operations form a computation graph.
3. Add autodiff: the framework must transform the forward computation into a backward computation.
4. Contrast programming models: eager/dynamic execution improves programmability; staged/static representations expose optimization.
5. Introduce JAX as a clean case study in staged array programming: `jit`, `grad`, and `vmap` transform Python functions.
6. Walk the compile path: Python/JAX tracing → JAXpr → StableHLO/HLO → optimization passes → backend executable.
7. Explain why shapes, layouts, memory spaces, and fusion matter for accelerator execution.
8. Use TPU as the hardware endpoint: systolic matrix units and compiler scheduling make layout and tiling load-bearing.
9. Show sharding as the distributed extension: logical partitioning becomes SPMD code and collectives.
10. Drop below automatic compilation with Pallas: when memory movement and loop structure dominate, the programmer may need explicit kernel control.
11. Close with the tradeoff: framework abstraction raises productivity, but systems performance depends on where the abstraction boundary sits.

## Required Figures

| Figure ID                            | Purpose                                                                     | Form                                        | Source                                                           |
| ------------------------------------ | --------------------------------------------------------------------------- | ------------------------------------------- | ---------------------------------------------------------------- |
| `fig-07-python-to-device-program`    | Show the full path from Python function to compiled accelerator executable. | Pipeline diagram                            | `llmsys-12-xla-compilation-pipeline`                             |
| `fig-07-forward-to-backward-graph`   | Explain autodiff as a program transformation.                               | Small computation graph with backward edges | `llmsys-05-automatic-differentiation`                            |
| `fig-07-ir-stack`                    | Locate JAXpr, StableHLO, HLO, backend IR, and executable.                   | Layered stack diagram                       | `llmsys-12-jax-tracing-jaxpr`, `llmsys-12-stablehlo-portability` |
| `fig-07-xla-memory-planning`         | Show why shape/layout/fusion choices affect buffers and HBM traffic.        | Dataflow and memory-layout diagram          | `llmsys-12-xla-optimization-passes`                              |
| `fig-07-tpu-systolic-compile-target` | Connect HLO tiling/layout to TPU matrix execution.                          | Systolic-array sketch                       | `llmsys-12-tpu-systolic-mxu`                                     |
| `fig-07-pallas-blockspec-grid`       | Explain grid coordinates, BlockSpec slices, VMEM refs, and HBM tensors.     | Kernel tiling diagram                       | `llmsys-13-pallas-blockspec`                                     |
| `fig-07-pallas-pipeline`             | Show overlapped HBM-to-VMEM transfer and compute.                           | Timeline diagram                            | `llmsys-13-pallas-pipelining`                                    |

## Main Sections

### The Model Is Python; the Machine Wants a Program

Open from a training step: forward pass, loss, backward pass, optimizer update. The system problem is not just expressing math; it is turning that expression into ordered device work with buffers and gradients.

### Computation Graphs and Autodiff

Explain primitive operators, graph edges, forward values, and backward accumulation. Include finite-difference gradient checking only as a validation contrast.

### Programming Models: Dynamic, Static, Functional

Use framework contrasts to explain the tradeoff between debuggability and optimizability. Keep this conceptual; avoid claiming a framework is “always dynamic” or “always static” without version-specific sources.

### JAX as Staged Array Programming

Introduce `jit`, `grad`, and `vmap` as transformations over functions. Emphasize that JAX transformation works because the program is restricted enough to trace into primitives.

### From JAXpr to HLO to an Executable

Walk through tracing, lowering, optimization, backend specialization, executable loading, and caching. This is the chapter’s central mechanism section.

### Why Compiler IR Carries System Information

Explain shapes, dtypes, layouts, tiling, buffer/copy insertion, memory-space assignment, and fusion. Tie this directly to HBM traffic and kernel launch/materialization costs already introduced in Chapters 4–6.

### TPU as the Compiler’s Target

Use TPU/MXU/systolic-array material to show why compiler scheduling is not abstract bookkeeping. The executable must feed matrix units, vector units, memory movement, and interconnect in a coordinated way.

### Sharding Turns Compilation into Distributed Execution

Introduce named sharding and SPMD lowering only enough to prepare for Chapters 8–10. The reader should leave understanding that collectives can be compiler-inserted consequences of partitioned tensors.

### When the Compiler Is Not Enough: Pallas

Explain Pallas as a lower-level kernel interface for explicit memory hierarchy control. Use `grid`, `BlockSpec`, VMEM refs, pipelining, and output aliasing as concrete mechanisms.

### Splash Attention as a Boundary Case

Brief sidebar: Splash Attention shows why modern kernels combine algorithmic sparsity, tiling, metadata, and compiler/runtime support. Keep the deep attention algorithm in Chapter 6 and future serving chapters.

## Technical Checks

- Formula correctness: Autodiff examples should use a small scalar or matrix expression with verified gradients.
- Complexity / memory accounting: Any statement about materializing attention or avoiding HBM writes must carry shape/backend conditions.
- Hardware assumptions: TPU Ironwood/MXU/VLIW details need official documentation or source-code cross-check before final draft.
- Benchmark conditions: Do not include Pallas tile-size speedups or bandwidth utilization unless shape, precision, tile size, TPU generation, and measurement method are stated.
- Terminology consistency: Use `computation graph`, `JAXpr`, `StableHLO`, `HLO`, `lowering`, `compilation`, `backend`, `SPMD`, `collective`, `HBM`, `VMEM`, `BlockSpec`, and `pallas_call` consistently.
- Chapter boundary: Do not re-teach FlashAttention; use Splash/Pallas only to explain compiler/kernel abstraction boundaries.

## Sidebar Decisions

- PyTorch 2 / TorchDynamo: possible sidebar later, but needs fresh official/source evidence.
- TensorFlow paper: useful historical reference if the chapter needs production graph-execution context; not required for first draft.
- StableHLO official docs: initial card added; use exact official compatibility wording if included.
- TPU official docs: should be added before draft if the text includes hardware generation specifics.
- Splash Attention: sidebar only unless the chapter is expanded into custom kernel design.

## Open Questions

- Are the four initial official-doc cards sufficient, or should the draft add narrower docs for `jax.jit`, JAXpr internals, Pallas `BlockSpec`, and TPU backend behavior?
- Should Chapter 7 include a short PyTorch 2 compiler-stack comparison, or stay focused on JAX/XLA/TPU as the clean case study?
- Should TPU hardware specifics remain qualitative unless official documentation is added?
- How much of sharding belongs here versus Chapter 8 distributed data parallelism and Chapter 9 model parallelism?

## Handoff

Owner: Book Architect  
Purpose: Chapter 7 brief from framework, JAX/XLA/TPU, and Pallas/Splash source extraction  
Evidence grade: A for course-framed claims from lecture PDFs and initial official-doc claims from JAX/OpenXLA docs; source code or narrower docs still needed for exact API behavior and TPU hardware specifics  
Assumptions: Chapter 7 uses JAX/XLA/TPU as a case study for framework/compiler/runtime design, not as a recommendation that all LLM systems should use JAX  
Open questions: Add narrower official docs/source cards before draft if making precise claims about JAXpr internals, Pallas `BlockSpec`, TPU backend behavior, or compatibility guarantees  
Handoff: Systems Explainer for Chapter 7 draft
