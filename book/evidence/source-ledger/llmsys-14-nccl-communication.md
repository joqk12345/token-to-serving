# Source Card: llmsys-14-nccl-communication

Source ID: llmsys-14-nccl-communication  
Title: Distributed Training  
Author/issuer: CMU 11868/11968 LLM Systems course staff; Lei Li  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-14-distributed-training-b27c3d1dc185e680c6f5cc924e9ec9d7.pdf`  
Pages/slides: Multi-GPU Communication section  
Claim supported: NCCL provides inter-GPU communication APIs, including collective and point-to-point primitives, across interconnect technologies such as PCIe, NVLink, InfiniBand, and IP sockets.  
Exact quote: "provides inter-GPU communication APIs"  
Paraphrase: The lecture introduces NCCL as the communication substrate for multi-GPU collectives and send/receive operations.  
Evidence grade: A  
Technical risk: Low for concept; official NCCL docs needed for exact API behavior.  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Vocabulary card for Chapter 8 communication substrate.
