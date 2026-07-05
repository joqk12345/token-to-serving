# Source Note: llmsys-26 Dynamo and Inference at Scale

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-26-dynamo-vikram_mailthody-0ddbeb69382d5c168b6d4636a82185d0.pdf`

## Scope

This source covers inference-at-scale challenges, Dynamo as a modular distributed inference stack, disaggregated serving, KV-aware routing, online/offline performance planning, multi-tier memory for KV offload, and NIXL-style cross-node/cross-memory transfer.

## Key Claims

- LLM inference differs from training because it serves many smaller online jobs with rapid scale-up/scale-down requirements.
- At scale, serving must handle variable input/output lengths, multi-turn and agentic workloads, heterogeneous hardware, fault tolerance, multiple models, TCO, goodput, power, and KV-cache hit rate.
- Dynamo is presented as a systematic, modular approach to distributed inference with scheduling, disaggregated serving, memory management, data transfer, and production-serving components.
- Disaggregated serving scales prefill and decode independently on separate GPUs and can apply different parallelism strategies to the two phases.
- KV-aware routing is presented as a scheduling technique that accounts for KV hit rate and load, but benchmark numbers need separate experiment-specific cards before use.
- Memory management can leverage HBM, host memory, local SSD, and network storage for KV offload.
- NIXL is presented as a cross-node/cross-memory buffer-list transfer layer with northbound API and southbound backend API.

## Chapter 14 Use

- Use Dynamo as a systems example of routing, scheduling, disaggregation, memory tiers, and transfer substrate.
- Avoid vendor-performance numbers unless full model, hardware, workload, SLA, and deployment details are carried.
- Use "AI factory" framing sparingly; the chapter should remain mechanism-first.

## Candidate Source Cards

- `llmsys-26-inference-vs-training`
- `llmsys-26-scale-serving-challenges`
- `llmsys-26-dynamo-modular-stack`
- `llmsys-26-dynamo-disaggregated-serving`
- `llmsys-26-kv-aware-routing`
- `llmsys-26-memory-tiers-kv-offload`
- `llmsys-26-nixl-transfer-layer`

Owner: Technical Researcher
Purpose: Chapter 14 Dynamo source extraction
Evidence grade: A for course lecture framing; benchmark claims require experiment-specific cards
Assumptions: Chapter 14 will use Dynamo as one example of distributed serving architecture, not as a product recommendation
Open questions: Whether to add official Dynamo docs/source-code cards for implementation details
Handoff: Book Architect for Chapter 14 brief
