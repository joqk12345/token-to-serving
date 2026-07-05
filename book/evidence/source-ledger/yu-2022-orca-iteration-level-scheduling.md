# Source Card: yu-2022-orca-iteration-level-scheduling

Source ID: yu-2022-orca-iteration-level-scheduling  
Title: Orca: A Distributed Serving System for Transformer-Based Generative Models  
Author/issuer: Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun  
Date: 2022  
Source type: paper  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/osdi22-yu.pdf`  
Pages / slides / sections: Abstract, Introduction  
Claim supported: Iteration-level scheduling invokes the execution engine for a single model iteration and lets the scheduler update request membership between iterations.  
Exact quote: "schedules execution at the granularity of iteration"  
Paraphrase: Iteration-level scheduling is the paper basis for continuous batching in autoregressive serving.  
Evidence grade: A  
Technical sensitivity: scheduling  
Conditions:
  model: Transformer-based generative serving
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Mechanism only; no performance numbers.
