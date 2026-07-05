# Source Card: kwon-2023-pagedattention

Source ID: kwon-2023-pagedattention  
Title: Efficient Memory Management for Large Language Model Serving with PagedAttention  
Author/issuer: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica  
Date: 2023  
Source type: paper  
File path / URL: `https://arxiv.org/abs/2309.06180`  
Pages / slides / sections: Abstract and system framing  
Claim supported: PagedAttention and vLLM address KV-cache memory waste in LLM serving by managing cache memory in blocks and enabling flexible sharing.  
Exact quote: "key-value cache (KV cache)"  
Paraphrase: The paper identifies dynamic KV-cache growth and fragmentation as a serving bottleneck, proposes PagedAttention, and builds vLLM as a serving system on top of that memory-management design.  
Evidence grade: A  
Technical sensitivity: memory management | serving system | benchmark  
Conditions:
  model: LLM serving workloads evaluated in the paper
  hardware:
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Do not use the paper's throughput or memory-saving numbers without creating experiment-specific cards that include model, GPU, workload trace, decoding method, and baselines.
