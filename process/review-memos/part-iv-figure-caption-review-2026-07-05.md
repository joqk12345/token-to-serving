# Part IV Figure Caption Review

Date: 2026-07-05

Scope: `book/figures/part-iv-figure-specs.md`

## Verdict

Part IV figure specs are ready for first-pass diagram production. The captions below are source-aligned and avoid unsupported claims about latency, throughput, cache hit rate, memory savings, or serving efficiency.

Do not add numeric TTFT, TPOT, goodput, memory-capacity, bandwidth, or speedup values to captions unless the figure also names the workload, model, hardware, precision, serving policy, and source context.

## Chapter 12 Captions

### `fig-12-prefill-decode`

Caption: Inference begins with prompt prefill, which processes the input context, then moves into decode steps that append one generated token at a time while updating the KV cache.

Source anchors: `yu-2022-orca-autoregressive-iterations`

Risk: Keep this as a work-shape diagram. Do not attach latency values without setup.

### `fig-12-request-level-batching-problem`

Caption: Request-level batching can leave shorter generations waiting on longer ones, tying batch progress to the slowest unfinished request.

Source anchors: `llmsys-22-naive-batch-longest`

Risk: Avoid exact utilization or waste claims unless the workload distribution is specified.

### `fig-12-continuous-batching`

Caption: Continuous batching changes active batch membership between decode iterations, adding new requests and removing completed ones as the scheduler advances.

Source anchors: `llmsys-22-continuous-batching`, `yu-2022-orca-iteration-level-scheduling`

Risk: Do not imply every serving engine uses the same admission or eviction policy.

### `fig-12-scheduler-loop`

Caption: An inference scheduler repeatedly admits requests, forms model-step batches, returns streamed tokens, updates memory state, and frees completed requests.

Source anchors: `llmsys-22-request-scheduler`

Risk: Keep implementation-specific class names out of the caption unless the figure is tied to one engine.

### `fig-12-selective-batching`

Caption: Selective batching separates operations that batch cleanly across requests from attention and cache operations that depend on request-specific state.

Source anchors: `llmsys-22-selective-batching`

Risk: Name ORCA or the source system if the artwork uses system-specific terminology.

### `fig-12-kv-cache-cost`

Caption: During autoregressive decoding, each generated token adds per-layer key and value state that future decode steps reuse.

Source anchors: `llmsys-22-kv-cache-need`

Risk: Do not render byte formulas unless dimensions, precision, layer count, and sequence length assumptions are shown.

### `fig-12-prefix-cache-routing`

Caption: Prefix caching can reuse work for requests with shared prompt prefixes when the scheduler routes them to workers that already hold matching cached state.

Source anchors: `llmsys-22-radixattention-prefix-cache`, `llmsys-22-cache-aware-scheduling`

Risk: Do not imply every shared prefix results in a cache hit; routing and eviction policy matter.

## Chapter 13 Captions

### `fig-13-kv-cache-growth`

Caption: KV cache is serving state that grows with active sequence context across model layers as new tokens are generated.

Source anchors: `llmsys-24-kv-cache-serving-state`

Risk: Avoid memory-size claims without model and precision conditions.

### `fig-13-contiguous-fragmentation`

Caption: Contiguous max-length reservations can leave unused tails inside allocations and free holes between allocations.

Source anchors: `llmsys-24-contiguous-preallocation-fragmentation`

Risk: Do not include waste percentages without a workload and allocation policy.

### `fig-13-kv-blocks`

Caption: PagedAttention divides a sequence's KV cache into fixed-size logical blocks, with the final block possibly only partly filled.

Source anchors: `llmsys-24-kv-block-definition`

Risk: Use symbolic block size unless a source-specific value is named.

### `fig-13-block-table`

Caption: A block table maps logical sequence blocks to physical KV blocks, allowing the sequence to be stored non-contiguously.

Source anchors: `llmsys-24-block-table-virtualization`

Risk: The virtual-memory analogy is useful, but do not imply OS page tables and KV block tables are identical.

### `fig-13-pagedattention-kernel`

Caption: The PagedAttention kernel follows block-table indirection to read a request's KV blocks from non-contiguous physical memory during attention.

Source anchors: `llmsys-24-pagedattention-kernel`

Risk: Keep kernel scheduling details out of the caption unless they are explicitly sourced.

### `fig-13-fragmentation-boundary`

Caption: Under fixed-size KV block allocation, unused space is concentrated in the final partially filled block rather than in a max-length contiguous reservation.

Source anchors: `llmsys-24-pagedattention-fragmentation`

Risk: Keep the fixed-block assumption in the caption or nearby prose.

### `fig-13-prefix-sharing`

Caption: Multiple continuations can share prompt-prefix KV blocks and then branch into separate continuation blocks after their outputs diverge.

Source anchors: `llmsys-24-kv-block-sharing`

Risk: Do not imply sharing applies after sequences diverge unless copy-on-write or reference behavior is shown.

### `fig-13-preemption-recovery`

Caption: When memory pressure preempts a request, the serving engine can recover by restoring swapped KV state or by recomputing needed prefix state before resuming.

Source anchors: `llmsys-24-preemption-recovery`

Risk: Do not rank swap and recompute without bandwidth, compute, and request-state conditions.

### `fig-13-vllm-engine-boundary`

Caption: PagedAttention sits inside a larger vLLM serving engine that includes API handling, scheduling, KV management, workers, and model kernels.

Source anchors: `llmsys-24-vllm-optimization-areas`

Risk: Keep engine boundaries aligned with the cited source and chapter text.

## Chapter 14 Captions

### `fig-14-throughput-vs-goodput`

Caption: Throughput counts completed work, while goodput counts work that completes within the serving objective being evaluated.

Source anchors: `llmsys-29-goodput-definition`

Risk: Define the serving objective or SLO in surrounding prose; do not use raw counts as universal quality measures.

### `fig-14-ttft-tpot`

Caption: Time to first token measures the delay before streaming begins, while time per output token measures the pace of later generated tokens.

Source anchors: `llmsys-29-ttft-tpot-slos`

Risk: Avoid numeric latency labels unless source and workload conditions are shown.

### `fig-14-prefill-decode-interference`

Caption: Colocating prefill and decode on the same resources can create interference because the two phases place different pressure on compute, memory, and scheduling.

Source anchors: `llmsys-29-colocation-interference`

Risk: Do not say colocating prefill and decode is categorically poor; workload and policy matter.

### `fig-14-pd-disaggregation`

Caption: Prefill-decode disaggregation routes prompt processing to prefill workers, transfers KV state, and streams generation from decode workers.

Source anchors: `llmsys-29-disaggregation-opportunity`, `llmsys-26-dynamo-disaggregated-serving`

Risk: Make the KV transfer path visible and avoid implying it is cost-free.

### `fig-14-placement-problem`

Caption: Serving placement chooses how model instances, parallelism groups, and prefill/decode roles map onto physical nodes and devices.

Source anchors: `llmsys-29-distserve-placement`

Risk: Do not encode a specific cluster topology unless it is labeled illustrative or sourced.

### `fig-14-kv-aware-routing`

Caption: KV-aware routing considers cache locality as well as load when assigning incoming requests to workers.

Source anchors: `llmsys-26-kv-aware-routing`

Risk: Do not imply cache locality should dominate all other routing signals.

### `fig-14-kv-memory-hierarchy`

Caption: KV cache management can span GPU memory, CPU DRAM, SSD, and remote storage tiers when active state exceeds the fastest local memory.

Source anchors: `llmsys-26-memory-tiers-kv-offload`, `llmsys-28-distributed-kv-pool`

Risk: Do not add tier latency or bandwidth numbers without hardware and system conditions.

### `fig-14-external-kv-manager`

Caption: An external KV manager can let inference engines look up, share, offload, or reuse KV state outside a single engine process.

Source anchors: `llmsys-27-lmcache-separated-service`, `llmsys-27-zero-copy-cpu-sharing`

Risk: Keep LMCache or other system names source-specific; external KV designs vary.

### `fig-14-transfer-substrate`

Caption: Disaggregated serving relies on a transfer substrate to move KV blocks between GPU memory, host memory, remote nodes, and storage services.

Source anchors: `llmsys-26-nixl-transfer-layer`, `llmsys-28-mooncake-store-integration`

Risk: Avoid protocol details unless they are part of the cited system description.

## Chapter 15 Captions

### `fig-15-codesign-loop`

Caption: LLM system design links model architecture, algorithms, software/runtime choices, and hardware capabilities in a feedback loop.

Source anchors: `llmsys-01-codesign`

Risk: Keep this as a synthesis figure. Do not imply one layer is permanently primary.

### `fig-15-stack-recap`

Caption: The book stack connects next-token objectives, tokenization, Transformer computation, kernels, compilers, distributed training, compression, inference scheduling, KV cache, and serving infrastructure.

Source anchors: `llmsys-01-system-challenges`

Risk: Keep labels aligned with chapter terminology.

### `fig-15-bottleneck-routing`

Caption: Systems diagnosis starts by identifying whether the active bottleneck is compute, memory, communication, scheduling, reliability, or abstraction, then choosing an intervention that can affect that bottleneck.

Source anchors: `llmsys-01-system-challenges`

Risk: Present the flow as a diagnostic aid, not a deterministic recipe.

### `fig-15-optimization-moves-bottleneck`

Caption: An optimization can relieve one bottleneck and expose another, shifting pressure from compute to memory, communication, scheduling, or another system resource.

Source anchors: Synthesis from Chapters 7-14

Risk: Avoid speedup language; this figure is a co-design pattern.

## Remaining Review Notes

- Captions are cleared for draft diagrams.
- Any rendered TTFT/TPOT timeline must define the exact interval boundaries used in the chapter.
- Any rendered KV-cache memory formula must include model dimensions, precision, layer count, and sequence length assumptions.
- Serving architecture figures should label cache ownership and transfer direction explicitly.
- Chapter 15 synthesis figures should reuse earlier visual motifs rather than inventing new notation.

Owner: Technical Reviewer  
Purpose: Part IV figure-caption review  
Evidence grade: A for review process; captions inherit their source anchors; Chapter 15 synthesis captions inherit evidence from Chapters 7-14  
Assumptions: Part IV figures are mechanism and synthesis diagrams, not benchmark artifacts  
Open questions: Whether Chapter 15 should include a separate chapter-to-bottleneck table during final layout  
Handoff: Illustrator / diagram producer
