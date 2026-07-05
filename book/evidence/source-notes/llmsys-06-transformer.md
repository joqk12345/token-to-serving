# Source Notes — llmsys-06-transformer

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-06-transformer-14bd7575a2f6c8bac60522354c11d691.pdf`

Course: 11868/11968 Large Language Model Systems  
Instructor: Lei Li

## High-Level Role

This lecture supports Chapter 2. It connects sequence-to-sequence modeling, Transformer architecture, attention, embeddings, FFN, residual connections, layer normalization, and training techniques.

## Core Claims

| Claim | Evidence grade | Source card |
|---|---|---|
| Encoder-decoder conditional generation factors target probability autoregressively conditioned on source and prior target tokens. | A | `llmsys-06-encoder-decoder-factorization` |
| Transformer removes recurrence to enable concurrent encoding and attention over full context. | A | `llmsys-06-transformer-motivation` |
| A Transformer block combines multi-head attention, feed-forward networks, residual connections, and layer normalization. | A | `llmsys-06-transformer-components` |
| Decoder self-attention masks future positions before softmax. | A | `llmsys-06-masked-self-attention` |
| Transformer training uses cross-entropy with teacher forcing for conditional generation. | A | `llmsys-06-teacher-forcing` |

## Chapter Use

Chapter 2 should not teach every Transformer variant. It should use this lecture to show how next-token or conditional-token probability becomes a tensor program with repeated blocks.

## Figure Candidates

| Figure ID | Purpose | Form |
|---|---|---|
| `fig-02-transformer-block` | Show MHA, FFN, residual, layer norm. | Block diagram |
| `fig-02-masked-attention` | Show causal mask hiding future tokens. | Attention matrix |
| `fig-02-qkv-flow` | Show Q, K, V projections and attention output. | Dataflow diagram |

## Risks / Checks

- Verify positional encoding formula before using it in final prose; the extracted text shows `1000`, while the original Transformer paper uses `10000`.
- Be explicit when switching from encoder-decoder Transformer to decoder-only LLMs.
- Do not overstate attention as the only expensive operation; FFN and memory movement matter.

Owner: Technical Researcher  
Purpose: Source extraction for Chapter 2  
Evidence grade: A for course-framing claims  
Assumptions: Chapter 2 will bridge math objective to Tensor/Transformer computation  
Open questions: Whether to include original Vaswani paper cards now or later  
Handoff: Book Architect for Chapter 2 brief

