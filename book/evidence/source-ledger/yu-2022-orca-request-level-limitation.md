# Source Card: yu-2022-orca-request-level-limitation

Source ID: yu-2022-orca-request-level-limitation  
Title: Orca: A Distributed Serving System for Transformer-Based Generative Models  
Author/issuer: Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun  
Date: 2022  
Source type: paper  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/osdi22-yu.pdf`  
Pages / slides / sections: Abstract, Introduction  
Claim supported: Request-level scheduling is inefficient for variable-length generative workloads because finished requests wait for the batch and newly arrived requests wait for the current batch to complete.  
Exact quote: "requests that have finished earlier than other requests in a batch cannot return to the client"  
Paraphrase: Static request batches create latency and queueing problems for variable-length autoregressive requests.  
Evidence grade: A  
Technical sensitivity: scheduling | latency  
Conditions:
  model: Transformer-based generative serving
  hardware:
  batch size:
  sequence length: variable output lengths
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Avoid benchmark comparison numbers.
