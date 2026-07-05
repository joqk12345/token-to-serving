# Source Card: llmsys-22-cache-aware-scheduling

Source ID: llmsys-22-cache-aware-scheduling  
Title: Design of Efficient LLM Inference Server  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-22-llm-serving-scheduler-radixattention-dfa87a4515092525676277a85bc4425d.pdf`  
Pages / slides / sections: Slides 34-38  
Claim supported: Cache-aware scheduling can sort requests by matched prefix length and route requests to workers with higher predicted prefix KV cache hit rates.  
Exact quote: "sort the requests according to matched prefix length"  
Paraphrase: Serving schedulers can account for KV prefix reuse when ordering and routing requests.  
Evidence grade: A  
Technical sensitivity: scheduling | cache  
Conditions:
  model: LLM serving with prefix KV cache
  hardware: multiple worker nodes in load-balancing slides
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Do not quote throughput/cache-hit table without setup.
