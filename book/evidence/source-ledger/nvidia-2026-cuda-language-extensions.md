# Source Card: nvidia-2026-cuda-language-extensions

Source ID: nvidia-2026-cuda-language-extensions  
Title: CUDA Programming Guide, C/C++ Language Extensions  
Author/issuer: NVIDIA  
Date: 2026  
Source type: documentation  
File path: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html  
Pages/slides: CUDA Programming Guide v13.3, Section 5.4  
Claim supported: CUDA C++ execution-space specifiers such as `__host__`, `__device__`, and `__global__` describe where functions execute and from where they may be called; `__shared__` declares shared-memory variables.  
Exact quote: "Execution Space Specifiers"; "`__shared__` memory variables"  
Paraphrase: NVIDIA's language-extension documentation anchors Chapter 4's CUDA qualifier and shared-memory syntax notes.  
Evidence grade: A  
Technical risk: Low; exact syntax should still be checked if runnable code is published.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Use for CUDA API-level review before adding runnable examples.
