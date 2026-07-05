# Source Note: llmsys-29 Disaggregating Prefill and Decode

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-29-disaggregating_prefill_decode_hao_zhang-c0e55139d20512a2348783423397cc7f.pdf`

## Scope

This source covers goodput-optimized LLM serving, TTFT/TPOT SLOs, continuous batching, prefill/decode interference, prefill/decode disaggregation, DistServe placement and communication problems, KV-cache transfer, and the relationship between continuous batching and disaggregation.

## Key Claims

- TTFT and TPOT represent different user-visible latency constraints.
- High throughput does not imply high goodput; goodput counts requests completed within SLO criteria.
- Prefill is compute-bound, while decode is memory-bound and needs many batched requests to saturate compute.
- Colocating prefill and decode can cause interference and coupled parallelism strategy.
- Disaggregating prefill and decode eliminates direct prefill/decode interference and lets prefill optimize TTFT while decode optimizes TPOT.
- Disaggregation introduces KV-cache communication overhead and a harder per-GPU goodput optimization problem.
- DistServe placement chooses parallelism strategy, number of instances, and physical cluster placement for prefill/decode instances.
- Continuous batching and disaggregation address different goals: continuous batching improves utilization/throughput, while disaggregation targets goodput under SLOs.

## Chapter 14 Use

- Use this as the spine for Chapter 14's main argument: scheduling at serving scale means optimizing goodput, not raw throughput.
- Use TTFT/TPOT terminology here with source support.
- Avoid DistServe evaluation numbers unless full setup is carried.
- Use placement and KV-transfer challenges qualitatively.

## Candidate Source Cards

- `llmsys-29-ttft-tpot-slos`
- `llmsys-29-goodput-definition`
- `llmsys-29-prefill-decode-characteristics`
- `llmsys-29-colocation-interference`
- `llmsys-29-disaggregation-opportunity`
- `llmsys-29-disaggregation-challenges`
- `llmsys-29-distserve-placement`
- `llmsys-29-cb-vs-disaggregation`

Owner: Technical Researcher
Purpose: Chapter 14 prefill/decode disaggregation source extraction
Evidence grade: A for course lecture framing; DistServe evaluation numbers require experiment-specific cards
Assumptions: Chapter 14 will use this source as the main conceptual bridge from Chapter 12 continuous batching to distributed disaggregation
Open questions: Whether to add the DistServe paper as a separate paper card
Handoff: Book Architect for Chapter 14 brief
