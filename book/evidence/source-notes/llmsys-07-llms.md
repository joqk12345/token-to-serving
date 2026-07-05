# Source Notes — llmsys-07-llms

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-07-llms-acf5db9438a8d9a86f86d29d9c563c00.pdf`

## High-Level Role

This lecture supports Chapter 2's transition from the original Transformer to pretrained LLM families: T5, LLaMA, and GPT-3.

## Core Claims

| Claim | Evidence grade | Source card |
|---|---|---|
| T5 uses a standard encoder-decoder Transformer and a unified text-to-text training format. | A | `llmsys-07-t5-text-to-text` |
| LLaMA is based on decoder-only Transformer architecture with pre-normalization, SwiGLU, and RoPE. | A | `llmsys-07-llama-architecture` |
| GPT-3 is based on Transformer architecture with modified initialization, pre-normalization, reversible tokenization, and alternating dense/local sparse attention patterns. | A | `llmsys-07-gpt3-architecture` |
| Modern pretrained LLM ideas include span corruption / next-token prediction, instruction-style finetuning, RoPE, SwiGLU, and sparse attention. | A | `llmsys-07-modern-llm-ideas` |

## Chapter Use

Use as a taxonomy bridge. The book should avoid becoming a model catalog; T5, LLaMA, and GPT-3 should illustrate architectural choices that affect systems.

## Risks / Checks

- Add original paper source cards before making detailed parameter-count claims.
- Keep T5 span-corruption distinct from decoder-only next-token training.
- Introduce RoPE and SwiGLU only as mechanisms that matter for modern LLM architecture; detailed derivations can stay short.

Owner: Technical Researcher  
Purpose: Source extraction for Chapter 2  
Evidence grade: A for course-framing claims  
Assumptions: Chapter 2 should use model families as examples, not as a survey  
Open questions: Whether to create separate appendix notes for T5/LLaMA/GPT-3  
Handoff: Book Architect for Chapter 2 brief

