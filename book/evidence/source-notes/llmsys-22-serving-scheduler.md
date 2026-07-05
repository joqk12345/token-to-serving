# Source Note: llmsys-22 Serving Scheduler and RadixAttention

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-22-llm-serving-scheduler-radixattention-dfa87a4515092525676277a85bc4425d.pdf`

## Scope

This source covers efficient LLM inference server design, serving patterns, request scheduler responsibilities, continuous batching, selective batching, prefill/decode behavior, KV cache management, RadixAttention, prefix-cache-aware scheduling, cache-aware load balancing, and scheduler/worker overlap.

## Key Claims

- LLM servers must handle multi-user sessions, variable generation lengths, common prompt prefixes, latency, throughput, and heterogeneous CPU/GPU resources.
- A request scheduler receives requests, streams model outputs, checks stop conditions, prepares batches, and allocates memory for new and running batches.
- Naive batching can wait for the longest generation in a request batch.
- Continuous batching schedules at iteration/token-generation granularity, admitting new requests when others finish.
- Selective batching batches non-attention operations while attention may need special handling through an attention engine.
- Transformer inference stores keys and values for previous tokens in GPU memory as KV cache.
- KV cache can be large and is a first-class serving memory object.
- RadixAttention stores KV memory pointers in a radix tree keyed by prompt prefixes to reuse shared prefixes.
- Cache-aware scheduling can sort queued requests by matched prefix length to increase KV cache hits.
- Scheduler/worker overlap reduces CPU scheduler overhead on the GPU critical path.

## Chapter 12 Use

- Use this source to introduce the serving cost model: request length, prefill, decode, batching, scheduler, and KV memory.
- Keep RadixAttention details light in Chapter 12; Chapter 13 will cover KV cache and vLLM-style memory management in depth.
- Use continuous batching as the key response to variable output lengths.
- Avoid throughput table numbers unless full setup is carried.

## Do Not Use As

- A benchmark source without model, hardware, request distribution, and implementation conditions.
- A complete source for vLLM or PagedAttention.
- A reason to move all KV-cache internals into Chapter 12.

## Candidate Source Cards

- `llmsys-22-serving-goals`
- `llmsys-22-request-scheduler`
- `llmsys-22-naive-batch-longest`
- `llmsys-22-continuous-batching`
- `llmsys-22-selective-batching`
- `llmsys-22-kv-cache-need`
- `llmsys-22-radixattention-prefix-cache`
- `llmsys-22-cache-aware-scheduling`
- `llmsys-22-scheduler-worker-overlap`

Owner: Technical Researcher  
Purpose: Chapter 12 serving scheduler source extraction  
Evidence grade: A for course framing; ORCA paper needed for publication-level batching claims  
Assumptions: Chapter 12 focuses on inference cost model; detailed KV-cache systems move to Chapter 13  
Open questions: How much RadixAttention detail to keep before overlapping Chapter 13  
Handoff: Book Architect for Chapter 12 brief
