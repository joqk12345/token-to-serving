# Source Note: llmsys-21 FlashAttention

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-21-FlashAttention_tridao2026.4-50476379a6127697ae7fbf974ad28348.pdf`

## Scope

This source explains attention optimization for modern hardware, starting from the memory bottleneck of standard attention and moving through FlashAttention, FlashAttention-3, FlashAttention-4, and decoding-oriented optimizations.

## Key Claims

- Standard attention materializes large `N x N` score and probability matrices.
- The main cost of attention can be moving data between HBM and on-chip SRAM, not only the number of FLOPs.
- FlashAttention uses tiling plus online softmax rescaling to compute exact attention by blocks.
- FlashAttention avoids writing the full attention matrix to HBM and uses recomputation in backward.
- Newer hardware changes the optimization problem through asynchronous instructions, low precision, persistent kernels, and load balancing.
- Decoding has a different shape: short query length and long KV context.

## Chapter 6 Use

- Explain why attention is the right case study after Chapters 4-5.
- Present tiling and online softmax as an algorithmic change motivated by memory hierarchy.
- Present recomputation as a memory-traffic tradeoff, not just gradient checkpointing.
- Use FA3/FA4 material as a short "modern hardware keeps moving" note.

## Do Not Use As

- The primary publication source for FlashAttention v1. Use `2205.14135.pdf` for that.
- A source for final FA3/FA4 benchmark claims without primary papers.
- A full tutorial on Hopper/Blackwell kernel programming.

## Candidate Source Cards

- `llmsys-21-attention-memory-bottleneck`
- `llmsys-21-online-softmax-rescaling`
- `llmsys-21-recomputation-backward`
- `llmsys-21-modern-hardware-attention`
- `llmsys-21-decoding-attention-shape`

Owner: Technical Researcher  
Purpose: Chapter 6 source extraction  
Evidence grade: A for course framing; use primary papers for publication-level claims  
Open questions: Whether FA3/FA4 should be sidebar-only in Chapter 6  
Handoff: Book Architect for Chapter 6 brief
