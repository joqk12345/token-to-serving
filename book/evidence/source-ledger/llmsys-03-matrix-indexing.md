# Source Card: llmsys-03-matrix-indexing

Source ID: llmsys-03-matrix-indexing  
Title: GPU Programming 2  
Author/issuer: CMU 11868/11968 LLM Systems course staff  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-03-gpu-programming2-b82b6ffdf554494747d00ce7ac606c3b.pdf`  
Pages/slides: matrix addition and multidimensional indexing examples  
Claim supported: Two-dimensional CUDA launch shapes can map thread indices onto matrix rows and columns.  
Exact quote: "blockIdx.x"; "threadIdx.y"; "C[i * N + j]"  
Paraphrase: The source shows how a 2D block/grid shape maps to matrix elements through index arithmetic.  
Evidence grade: A  
Technical risk: Low for indexing concept; omit unchecked code details in final prose if not tested.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Use as a bridge from vector addition to matrix/tensor computation.
