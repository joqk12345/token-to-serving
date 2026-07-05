# Source Note: llmsys-28 Mooncake and kTransformers

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-28-mooncake-kTransformer-1243bfbecbb0c3610bf95eac030acb2a.pdf`

## Scope

This source covers Mooncake as a KV-cache-centric disaggregated architecture, prefill/decode disaggregation, distributed KV-cache pools, multi-layer KV-cache storage, Mooncake Store integration, and kTransformers offloading for heterogeneous hardware.

## Key Claims

- Mooncake is presented as a KV-cache-centric disaggregated architecture with conductor, cache-aware prefill scheduler, KV-cache pool, KV-cache balance scheduler, and decoding scheduler.
- P/D disaggregation is used to avoid interference between prefill and decode and to decouple resources and parallelism.
- KV-cache caching introduces storage challenges because cache size and transfer bandwidth matter for GPU stalls.
- Mooncake Store is presented as a distributed multi-layer KV-cache cache using pooled memory and high-performance connections such as RDMA/storage.
- Mooncake Store exposes integration points such as zero-copy object put/get and batch-transfer APIs.
- kTransformers is presented later as an offloading strategy for heterogeneous hardware; this should be secondary to Chapter 14 unless the chapter expands into heterogeneous serving.

## Chapter 14 Use

- Use Mooncake as an example of KV-cache-centric distributed serving architecture.
- Avoid the evaluation numbers unless adding experiment-specific cards.
- Keep kTransformers as a boundary/sidebar; the main Chapter 14 topic is disaggregated serving and distributed KV-cache management.

## Candidate Source Cards

- `llmsys-28-mooncake-kvcache-centric`
- `llmsys-28-pd-disaggregation-interference`
- `llmsys-28-kvcache-storage-challenges`
- `llmsys-28-distributed-kv-pool`
- `llmsys-28-mooncake-store-integration`

Owner: Technical Researcher
Purpose: Chapter 14 Mooncake/kTransformers source extraction
Evidence grade: A for course lecture framing; evaluation claims require conditions
Assumptions: Chapter 14 focuses on Mooncake's KV-cache/disaggregation concepts, not detailed kTransformers implementation
Open questions: Whether to add Mooncake technical report cards for publication-level support
Handoff: Book Architect for Chapter 14 brief
