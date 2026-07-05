# Source Card: yu-2022-orca-autoregressive-iterations

Source ID: yu-2022-orca-autoregressive-iterations  
Title: Orca: A Distributed Serving System for Transformer-Based Generative Models  
Author/issuer: Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun  
Date: 2022  
Source type: paper  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/osdi22-yu.pdf`  
Pages / slides / sections: Abstract, Introduction, Section 2  
Claim supported: Autoregressive Transformer generation processes a request over multiple model iterations, where each iteration generates one output token.  
Exact quote: "each iteration of the model generates a single output token"  
Paraphrase: LLM serving differs from one-shot inference because each request may require many sequential decode iterations.  
Evidence grade: A  
Technical sensitivity: serving workload  
Conditions:
  model: Transformer-based generative model
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Avoid performance claims without ORCA experiment setup.
