# Source Note: llmsys-13 Pallas Kernels and Splash Attention

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-13-pallas_splash_attention_srinath_mandalapu-b0bc7990950b84561ff9aa8e1791f727.pdf`

## Scope

This source covers Pallas as a JAX kernel-writing layer for explicit control over TPU memory movement, VMEM/HBM transfers, BlockSpec tiling, software pipelining, output aliasing, matmul tiling, tile-size tuning, Pallas compilation, Flash/Splash Attention, sparse attention masks, and profiling with Pacchetto.

## Key Claims

- Generic JIT fusion can hit an abstraction ceiling when the programmer needs explicit control over memory hierarchy and loop structure.
- Pallas exposes a Python kernel abstraction while still lowering through the compiler stack; it lets kernels manage HBM-to-VMEM movement, tiling, pipelining, and scratch buffers.
- `BlockSpec` maps global HBM tensors into VMEM blocks for each grid coordinate; the grid defines the iteration space over those blocks.
- Tiling is necessary because loading entire LLM-sized tensors into VMEM can exceed VMEM capacity.
- Pipelining overlaps HBM transfers with computation; output aliasing can reduce extra allocation and HBM bandwidth.
- Tile size and grid shape are performance-sensitive; changing tile size changes invocation count and instruction overhead.
- Splash Attention combines FlashAttention-style tiling/fusion with sparse block masks so the kernel can skip irrelevant attention blocks.

## Chapter 7 Use

- Use Pallas as the “escape hatch” section: after automatic compiler lowering, kernel authors sometimes need explicit memory/schedule control.
- Explain `pallas_call`, `grid`, `BlockSpec`, and kernel refs as a controlled way to cross the boundary between array programming and accelerator-specific kernels.
- Use VMEM/HBM examples to connect Chapter 7 back to Chapters 4–6 without duplicating GPU CUDA material.
- Treat Splash Attention as a preview: compiler/runtime abstractions matter because attention kernels are now part algorithm, part memory schedule, part sparse execution map.

## Do Not Use As

- A standalone proof that Pallas kernels outperform all XLA-generated kernels.
- A benchmark source unless shape, tile size, precision, device, and measurement method are preserved.
- A complete public API reference for Pallas or Tokamax.

## Candidate Source Cards

- `llmsys-13-pallas-memory-hierarchy`
- `llmsys-13-pallas-blockspec`
- `llmsys-13-pallas-pipelining`
- `llmsys-13-pallas-output-aliasing`
- `llmsys-13-pallas-vmem-constraint`
- `llmsys-13-pallas-tile-size-tuning`
- `llmsys-13-splash-attention-sparse-flash`
- `llmsys-13-splash-attention-mask-metadata`

Owner: Technical Researcher  
Purpose: Chapter 7 source extraction  
Evidence grade: A for course framing; official JAX/Pallas docs and source code needed for API-level claims  
Assumptions: Pallas belongs in Chapter 7 as the lower-level compiler/runtime interface, while Chapter 6 remains the main FlashAttention algorithm chapter  
Open questions: Whether Splash Attention should be a Chapter 7 sidebar or deferred to serving/sparse-attention discussion  
Handoff: Book Architect for Chapter 7 brief
