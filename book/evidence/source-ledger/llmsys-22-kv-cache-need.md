# Source Card: llmsys-22-kv-cache-need

Source ID: llmsys-22-kv-cache-need  
Title: Design of Efficient LLM Inference Server  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-22-llm-serving-scheduler-radixattention-dfa87a4515092525676277a85bc4425d.pdf`  
Pages / slides / sections: Slides 18-20  
Claim supported: Transformer generation stores keys and values for previous tokens in GPU memory as KV cache, which can become a major memory cost.  
Exact quote: "Store all keys, values for each layer in GPU memory"  
Paraphrase: Autoregressive decoding avoids recomputing previous attention keys/values by keeping a KV cache, turning sequence length into serving memory pressure.  
Evidence grade: A  
Technical sensitivity: memory | inference  
Conditions:
  model: Transformer autoregressive generation
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Avoid slide's LLaMA KV numeric example unless precision/model/context assumptions are carried.
