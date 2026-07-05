# Source Note: llmsys-05 Deep Learning Framework and Auto Differentiation

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-05-dlframework-fa0770d636572de3f7b48ccae0ba8848.pdf`

## Scope

This source covers why deep learning frameworks exist as systems: tensor operators, computation graphs, automatic differentiation, graph execution, framework differences, and the TensorFlow paper as a large-scale ML system reference.

## Key Claims

- Training a neural network requires computing gradients of a loss with respect to parameters; frameworks make that a graph/program transformation problem rather than a hand-derived calculus exercise.
- Automatic differentiation works by constructing a computation graph from primitive operators and attaching or deriving backward operations for those primitives.
- Correctness of gradients can be checked with finite differences, but finite differences are a validation tool, not a scalable training method.
- TensorFlow-style graph execution separates graph definition from optimized execution on available devices.
- Frameworks differ in programming model: PyTorch emphasizes dynamic computation; TensorFlow historically emphasized static computation graphs; JAX emphasizes functional transformations.

## Chapter 7 Use

- Open the chapter with the concrete problem: the model author writes Python, but the accelerator needs a scheduled tensor program with gradients and memory decisions.
- Explain autodiff as the first compiler-like layer: forward computation becomes a backward computation.
- Use static-versus-dynamic graph framing to motivate why JAX/XLA is not just “another NumPy,” but a staged programming system.
- Bridge into XLA by showing why graphs expose optimization opportunities unavailable to opaque Python execution.

## Do Not Use As

- A primary source for current framework market share or production adoption.
- A performance benchmark source.
- A complete specification of PyTorch, TensorFlow, or JAX internals.

## Candidate Source Cards

- `llmsys-05-computation-graph`
- `llmsys-05-automatic-differentiation`
- `llmsys-05-gradient-checking`
- `llmsys-05-framework-programming-models`
- `llmsys-05-tensorflow-graph-execution`

Owner: Technical Researcher  
Purpose: Chapter 7 source extraction  
Evidence grade: A for course framing; original framework papers or official docs needed for publication-level implementation details  
Assumptions: Chapter 7 treats frameworks as compiler/runtime interfaces, not as a product comparison  
Open questions: Whether Chapter 7 should include a short PyTorch 2 / TorchDynamo sidebar later  
Handoff: Book Architect for Chapter 7 brief
