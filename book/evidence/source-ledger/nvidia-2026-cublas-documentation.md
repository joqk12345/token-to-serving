# Source Card: nvidia-2026-cublas-documentation

Source ID: nvidia-2026-cublas-documentation  
Title: cuBLAS 13.3 Documentation  
Author/issuer: NVIDIA  
Date: 2026  
Source type: documentation  
File path: https://docs.nvidia.com/cuda/cublas/index.html  
Pages/slides: cuBLAS 13.3 documentation, Sections 2.4, 2.5, 2.6, 2.7  
Claim supported: cuBLAS provides BLAS routines including Level-1 vector operations, Level-2 matrix-vector operations, Level-3 matrix-matrix operations, and handle lifecycle functions such as `cublasCreate` and `cublasDestroy`.  
Exact quote: "cublasCreate"; "cublasDestroy"; "cublas<t>gemm()"  
Paraphrase: NVIDIA's cuBLAS documentation anchors Chapter 5's claim that optimized dense linear algebra should generally use mature BLAS libraries rather than custom GEMM kernels.  
Evidence grade: A  
Technical risk: Low for library role; exact API signatures and behavior should be checked when publishing runnable code.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Official source for cuBLAS claims in Chapter 5.
