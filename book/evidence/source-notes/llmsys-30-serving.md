# Source Note: llmsys-30 Serving Frameworks

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-30-serving-c4a70ab21cde01fb60068a256c6e163a.pdf`

## Scope

This source covers the LLM application stack, prompt execution/inference, serving frameworks, batching in Triton plus inference engines, Text Generation Inference, OpenLLM, MLC LLM, and LightLLM token attention / efficient router concepts.

## Key Claims

- LLM applications include data preprocessing/embedding, prompt construction/retrieval, and prompt execution/inference.
- Prompt execution can happen through hosted APIs, hosted models, serving frameworks, and application infrastructure.
- Serving systems can batch multiple client requests and rely on optimized inference engines for model execution.
- NVIDIA Triton can group client requests into batches, while engines such as TensorRT-LLM conduct batched inference.
- Text Generation Inference includes optimized transformer inference support.
- LightLLM performs tokenization, inference, and detokenization asynchronously, and includes token-wise KV cache memory management plus routing.

## Chapter 12 Use

- Use only for high-level serving architecture and framework boundary.
- Avoid turning Chapter 12 into a product survey.
- Use framework examples to show separation between API/server layer, scheduler, model worker, tokenizer/detokenizer, and memory manager.

## Do Not Use As

- A current product recommendation source.
- A source for current feature support without official docs verification.
- A benchmark source.

## Candidate Source Cards

- `llmsys-30-llm-app-stack`
- `llmsys-30-serving-framework-boundary`
- `llmsys-30-triton-batching-engine`
- `llmsys-30-lightllm-async-token-attention`

Owner: Technical Researcher  
Purpose: Chapter 12 serving framework source extraction  
Evidence grade: A for course framing; official docs needed for current product claims  
Assumptions: Chapter 12 uses framework examples only to explain serving architecture  
Open questions: Whether to include named framework examples in main prose or sidebars  
Handoff: Book Architect for Chapter 12 brief
