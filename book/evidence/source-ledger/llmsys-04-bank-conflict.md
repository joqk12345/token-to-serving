# Source Card: llmsys-04-bank-conflict

Source ID: llmsys-04-bank-conflict  
Title: GPU Acceleration  
Author/issuer: CMU 11868/11968 LLM Systems course staff  
Date: Spring 2026  
Source type: lecture  
File path: `downloads/llmsystem2026spring/source_pdfs/llmsys-04-gpu-acceleration-48ffa5768ba62c54138f0a71ca2b68b8.pdf`  
Pages/slides: shared memory bank conflict section  
Claim supported: Shared memory bank conflicts can serialize accesses, and padding a shared-memory tile can avoid some conflict patterns.  
Exact quote: "Shared memory has 32 banks"; "access to same bank's different element will be sequential"  
Paraphrase: The source shows a matrix transpose variant that adds one column to a shared-memory tile to avoid bank conflicts.  
Evidence grade: A  
Technical risk: Medium; bank organization and behavior are architecture-specific.  
Checked by: Codex  
Checked date: 2026-07-04  
Notes: Treat as advanced detail or sidebar.
