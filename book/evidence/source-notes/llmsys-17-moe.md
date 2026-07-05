# Source Note: llmsys-17 MoE

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-17-MoE-3aa3125f9ccdd4bb7109ef077fbe9260.pdf`

## Scope

This source covers Transformer Mixture-of-Experts models, Switch-style routing, shared and routed experts, expert parallelism, all-to-all communication, load balancing, MoE inference bottlenecks, DeepSpeed-MoE system optimizations, and DeepSeek-style fine-grained/shared expert designs.

## Key Claims

- Transformer MoE replaces a dense Transformer FFN with multiple expert FFNs and a gating/router network that selects one or more experts for each token.
- Switch-style MoE can route one token through one selected FFN.
- Shared-routed expert designs combine an always-used shared expert with routed experts for token-specific computation.
- Expert parallelism keeps experts on different worker devices while replicating non-expert network components.
- Expert parallelism requires fast all-to-all communication for token dispatch and expert outputs.
- MoE training needs load balancing because routing can collapse toward a subset of experts.
- MoE inference performance depends on model size, number of activated experts, and memory bandwidth.
- Optimizing MoE inference involves grouping tokens by critical data path, coordinating communication with parallelism strategy, and optimizing MoE-specific kernels.
- DeepSeek-style MoE uses fine-grained experts, shared experts, routed experts, and load-balancing mechanisms.

## Chapter 10 Use

- Use MoE as the sparse-activation side of the chapter: the system stores many parameters but activates only a subset per token.
- Explain router/gating and expert FFNs before discussing distributed expert placement.
- Pair expert parallelism with all-to-all communication as the core systems bottleneck.
- Treat load balancing as correctness-for-throughput: the router is part of both the model and the scheduler.
- Avoid throughput or latency numbers from the slides unless original paper setup is carried.

## Do Not Use As

- A benchmark source for DeepSpeed-MoE kernel latency or throughput claims without original paper conditions.
- A complete current DeepSeek-V3 architecture source.
- A source for unconditional claims that MoE is faster or cheaper than dense models across workloads.

## Candidate Source Cards

- `llmsys-17-moe-ffn-router`
- `llmsys-17-switch-top1-routing`
- `llmsys-17-shared-routed-experts`
- `llmsys-17-expert-parallelism`
- `llmsys-17-moe-load-balancing`
- `llmsys-17-moe-inference-bottlenecks`
- `llmsys-17-moe-alltoall-optimization`
- `llmsys-17-deepseek-moe-design`

Owner: Technical Researcher  
Purpose: Chapter 10 MoE source extraction  
Evidence grade: A for course framing; original papers needed for publication-level named-system claims  
Assumptions: Chapter 10 treats MoE as training-memory/sparse-activation systems material, not as a full model-family survey  
Open questions: How much DeepSeek-specific architecture detail belongs in the chapter versus a sidebar  
Handoff: Book Architect for Chapter 10 brief
