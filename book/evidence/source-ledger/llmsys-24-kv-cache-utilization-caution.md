# Source Card: llmsys-24-kv-cache-utilization-caution

Source ID: llmsys-24-kv-cache-utilization-caution  
Title: Paged Attention & vLLM for Efficient LLM Inference Engine  
Author/issuer: Woosuk Kwon / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-24-vLLM_woosuk_kwon-b6a0750bb310949461ba5a635a1126eb.pdf`  
Pages / slides / sections: Slide 16  
Claim supported: The lecture reports that earlier systems could use only a minority of KV-cache space for actual token states, citing ORCA.  
Exact quote: "Only 20–40% of KV Cache space"  
Paraphrase: The slide uses ORCA-era serving systems as motivation for treating KV-cache allocation waste as a first-order serving problem.  
Evidence grade: A  
Technical sensitivity: benchmark | memory utilization  
Conditions:
  model: not specified on slide
  hardware: not specified on slide
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: needs original experiment context before direct use  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Do not state this number as a general fact in prose unless the ORCA/source experiment conditions are recovered.
