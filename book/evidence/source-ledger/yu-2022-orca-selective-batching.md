# Source Card: yu-2022-orca-selective-batching

Source ID: yu-2022-orca-selective-batching  
Title: Orca: A Distributed Serving System for Transformer-Based Generative Models  
Author/issuer: Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun  
Date: 2022  
Source type: paper  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/osdi22-yu.pdf`  
Pages / slides / sections: Abstract, Introduction  
Claim supported: Selective batching applies batching only to selected operations, handling attention separately when sequence lengths differ.  
Exact quote: "apply batching only to a selected set of operations"  
Paraphrase: Selective batching keeps batching benefits for compatible operations while addressing variable-shape attention.  
Evidence grade: A  
Technical sensitivity: batching | attention  
Conditions:
  model: Transformer generative serving
  hardware:
  batch size:
  sequence length: variable processed-token lengths
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: This is a mechanism card, not a benchmark card.
