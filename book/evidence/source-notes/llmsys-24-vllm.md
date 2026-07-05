# Source Note: llmsys-24 vLLM and PagedAttention

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-24-vLLM_woosuk_kwon-b6a0750bb310949461ba5a635a1126eb.pdf`

## Scope

This source covers PagedAttention, KV-cache memory pressure, block-based KV-cache allocation, logical-to-physical block tables, block sharing, request preemption and recovery, and the broader vLLM inference engine architecture.

## Key Claims

- LLM inference becomes expensive because model size, context length, and generated-token count increase the amount of GPU work and memory state.
- Attention KV cache is persistent serving state: generated tokens add key/value states that later decode steps need.
- Efficient KV-cache management is central to high-throughput serving because parameters are not the only large GPU-memory consumer.
- Earlier contiguous preallocation strategies reserve space up to a request maximum length and can create internal fragmentation from unknown output length plus external fragmentation from non-uniform request sizes.
- PagedAttention applies paging and virtualization ideas at the application level for attention KV cache.
- A KV block is a fixed-size unit of KV-cache memory; a request has logical KV blocks that map to physical KV blocks through a block table.
- PagedAttention fetches non-contiguous KV blocks through the block table and applies attention without materializing gathered keys and values.
- PagedAttention's internal fragmentation is limited to the last block of each sequence, and it avoids external fragmentation under the block allocator model.
- Paging enables sharing KV blocks across related sequences, such as parallel samples with the same prompt prefix.
- When physical KV blocks run out, the system can preempt requests by swapping KV cache to CPU memory or by deleting/recomputing it later.
- The lecture draws an explicit analogy to operating-system virtual memory: OS pages correspond to KV blocks, shared pages correspond to shared KV blocks, and preemption/recovery operates at request level.
- vLLM exposes both an offline Python `LLM` API and an OpenAI-compatible online server interface.
- vLLM optimization areas include reducing CPU overhead, efficient GPU kernels, model parallelism, and memory management/caching.
- vLLM overlaps scheduling/input preparation with model execution through asynchronous scheduling.
- vLLM's parallelism discussion distinguishes data parallel, tensor parallel, expert parallel, context parallel, pipeline parallel, and prefill/decode disaggregation tradeoffs.

## Chapter 13 Use

- Use this source to make KV cache the chapter's central memory-management problem.
- Explain PagedAttention as a systems design: fixed-size blocks, logical-to-physical mapping, custom attention kernel, sharing, and preemption.
- Use operating-system virtual memory as an analogy, while calling out differences: single-level block tables, request-level preemption, and recomputation-based recovery.
- Keep Chapter 13 focused on per-engine KV-cache management and vLLM. Leave distributed serving, disaggregated prefill/decode, LMCache, Dynamo, and Mooncake to Chapter 14.
- Avoid vLLM benchmark speedup numbers unless the text carries the model, GPU, workload trace, decoding mode, batch/request conditions, and baseline.

## Do Not Use As

- A source for a complete product survey of serving engines.
- A source for current vLLM feature support without checking official documentation or source code.
- A source for unconditional throughput claims.
- A reason to move disaggregated-serving architecture into Chapter 13.

## Candidate Source Cards

- `llmsys-24-inference-scaling-pressure`
- `llmsys-24-kv-cache-serving-state`
- `llmsys-24-kv-cache-memory-management`
- `llmsys-24-contiguous-preallocation-fragmentation`
- `llmsys-24-kv-cache-utilization-caution`
- `llmsys-24-pagedattention-definition`
- `llmsys-24-kv-block-definition`
- `llmsys-24-block-table-virtualization`
- `llmsys-24-pagedattention-kernel`
- `llmsys-24-pagedattention-fragmentation`
- `llmsys-24-kv-block-sharing`
- `llmsys-24-preemption-recovery`
- `llmsys-24-os-virtual-memory-analogy`
- `llmsys-24-vllm-api-surface`
- `llmsys-24-vllm-optimization-areas`
- `llmsys-24-vllm-parallelism-options`

Owner: Technical Researcher
Purpose: Chapter 13 vLLM/PagedAttention source extraction
Evidence grade: A for course lecture framing; original PagedAttention paper card added for publication-level support
Assumptions: Chapter 13 focuses on KV-cache memory management inside an inference engine, not full distributed serving
Open questions: Whether to include benchmark numbers after adding experiment-specific cards from the original paper
Handoff: Book Architect for Chapter 13 brief
