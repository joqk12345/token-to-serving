# Chapter 4 Figure Artwork Review

Date: 2026-07-06

Scope: `book/figures/artwork/ch04/*.svg`

## Verdict

The four Chapter 4 diagrams are cleared as first-pass book artwork:

- `fig-04-host-device-lifecycle`
- `fig-04-grid-block-thread`
- `fig-04-sm-warp-scheduling`
- `fig-04-memory-paths`

The assets continue the established 1200 by 760 canvas, type hierarchy, line weights, and semantic palette. Each SVG includes an accessible title and description.

## Checks

| Check | Result | Notes |
| --- | --- | --- |
| XML validity | pass | All four files pass `xmllint --noout`. |
| Technical overclaim scan | pass | No unsupported bandwidth, capacity, occupancy, or speedup number was introduced. |
| Terminology | pass | Host, device, grid, block, thread, SM, warp, registers, shared memory, L2, and GPU memory match Chapter 4. |
| Formula | pass | The selected thread example satisfies `i = blockDim.x * blockIdx.x + threadIdx.x = 8 * 1 + 3 = 11`. |
| Layout | pass | Text and shapes remain within the SVG view box; rendered thumbnails confirm hierarchy and dataflow. |
| Accessibility | pass | Each file has `role="img"` plus linked `title` and `desc` elements. |

## Review Notes

- The lifecycle figure keeps CUDA API names visible without expanding into an API reference.
- The launch hierarchy uses one-dimensional indexing and retains the bounds-check caveat.
- The scheduling figure explicitly labels its 32-thread warp semantics as NVIDIA CUDA terminology.
- The memory-path figure avoids exact capacity and bandwidth values and states that they depend on the specific GPU and server.
- The diagrams make resource scope visible: registers per thread, shared memory per block, and global state in GPU memory.

Owner: Diagram Producer  
Purpose: First-pass Chapter 4 artwork and rendered-artwork review  
Evidence grade: A for source-aligned conceptual diagrams; no benchmark claims  
Assumptions: SVG is the canonical editable format and final typography may be adjusted during publication layout  
Open questions: Whether the publication pipeline requires PDF, EPS, or outlined-font exports  
Handoff: Chapter 5 diagram production using the same visual system
