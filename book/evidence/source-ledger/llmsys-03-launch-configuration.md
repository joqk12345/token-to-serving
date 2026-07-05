# Source Card: llmsys-03-launch-configuration

Source ID: llmsys-03-launch-configuration  
Title: GPU Programming 2  
Author/issuer: CMU 11868/11968 LLM Systems course staff  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-03-gpu-programming2-b82b6ffdf554494747d00ce7ac606c3b.pdf`  
Pages/slides: calling kernel at runtime section  
Claim supported: Kernel launches specify grid and block dimensions, and the product of block dimensions gives threads per block.  
Exact quote: "kernelFuncName<<<Dg, Db>>>(args)"; "threads per block, <=1024"  
Paraphrase: The source shows runtime launch configuration as an explicit choice made by the host program.  
Evidence grade: A  
Technical risk: Low for introductory CUDA; exact limits vary by device and should be qualified.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Use to introduce grid/block shape decisions.
