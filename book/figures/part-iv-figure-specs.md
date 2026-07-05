# Part IV Figure Specs

Scope: Chapters 12-15

These specs cover inference execution, scheduler behavior, KV-cache memory management, PagedAttention, serving disaggregation, external KV systems, and the final co-design synthesis. The goal is to make request lifecycle, memory ownership, and bottleneck routing concrete without embedding unsourced benchmark numbers.

## Chapter 12

### `fig-12-prefill-decode`

- Chapter: 12
- Purpose: Show prompt prefill followed by token-by-token decode.
- Core message: Inference has two different work shapes: prefill processes the prompt in bulk, while decode extends the sequence one token at a time.
- Visual form: Timeline/dataflow.
- Layout: prompt tokens enter prefill block; generated tokens appear in repeated decode steps; KV cache grows alongside the timeline.
- Labels: prompt, prefill, decode step, generated token, KV cache.
- Source anchors: `yu-2022-orca-autoregressive-iterations`
- Production note: Keep the distinction qualitative; do not attach latency values without setup.

### `fig-12-request-level-batching-problem`

- Chapter: 12
- Purpose: Show short requests waiting for the longest generation.
- Core message: Request-level batching wastes work or time when sequences finish at different lengths.
- Visual form: Timeline.
- Layout: several requests in one batch, with shorter requests ending early while batch slots remain tied to the longest request.
- Labels: request, batch, finished request, longest sequence, idle slot.
- Source anchors: `llmsys-22-naive-batch-longest`
- Production note: Avoid exact utilization claims; the point is scheduling shape.

### `fig-12-continuous-batching`

- Chapter: 12
- Purpose: Show batch membership changing between decode iterations.
- Core message: Continuous batching treats decode as an iteration-level scheduling problem, admitting and removing requests between steps.
- Visual form: Timeline.
- Layout: decode iterations as columns; active batch rows change as requests finish and new requests enter.
- Labels: iteration, active request, admitted request, completed request, scheduler.
- Source anchors: `llmsys-22-continuous-batching`, `yu-2022-orca-iteration-level-scheduling`
- Production note: Make membership changes visible; avoid suggesting every system uses the same policy.

### `fig-12-scheduler-loop`

- Chapter: 12
- Purpose: Show scheduler receive/run/result/free loop.
- Core message: Serving is a control loop that admits requests, forms batches, runs model steps, returns tokens, and frees memory.
- Visual form: Loop diagram.
- Layout: receive queue -> scheduling decision -> model execution -> token output -> memory update/free -> back to scheduling.
- Labels: request queue, scheduler, model step, token stream, memory manager, free.
- Source anchors: `llmsys-22-request-scheduler`
- Production note: Keep this architecture-level; do not overfit to one serving engine's class names.

### `fig-12-selective-batching`

- Chapter: 12
- Purpose: Show non-attention batching with special attention path.
- Core message: Some operations batch naturally across requests, while attention and cache access may require request-specific handling.
- Visual form: Block diagram.
- Layout: request batch enters shared non-attention operators, diverges into attention/KV-cache handling, then rejoins.
- Labels: batched operator, attention, KV cache, request-specific state, output.
- Source anchors: `llmsys-22-selective-batching`
- Production note: Keep the mechanism conceptual unless using ORCA-specific terminology in the caption.

### `fig-12-kv-cache-cost`

- Chapter: 12
- Purpose: Show per-layer KV cache accumulating with tokens.
- Core message: Autoregressive decoding stores key/value tensors for prior tokens, so memory grows with generated context across layers.
- Visual form: Memory growth diagram.
- Layout: layers as stacked rows, token positions as columns, K/V cells added at each decode step.
- Labels: layer, token position, key, value, cache growth, decode step.
- Source anchors: `llmsys-22-kv-cache-need`
- Production note: Do not include a byte formula in the artwork unless model dimensions and precision are stated.

### `fig-12-prefix-cache-routing`

- Chapter: 12
- Purpose: Show prompt prefix reuse and cache-aware scheduling.
- Core message: Shared prefixes can reuse cached computation if routing and scheduling place requests where the relevant prefix state is available.
- Visual form: Prefix tree plus queue.
- Layout: request prompts share prefix nodes; scheduler routes requests to workers with matching prefix-cache state.
- Labels: prefix, cache hit, cache-aware scheduler, worker, reuse.
- Source anchors: `llmsys-22-radixattention-prefix-cache`, `llmsys-22-cache-aware-scheduling`
- Production note: Caption should say prefix cache or radix-style cache only if the chapter names the specific system.

## Chapter 13

### `fig-13-kv-cache-growth`

- Chapter: 13
- Purpose: Show KV state accumulating per generated token across layers.
- Core message: KV cache is a serving-state object whose size follows active sequence length, layer count, head structure, and precision.
- Visual form: Memory growth diagram.
- Layout: one request's sequence grows token by token; each token owns per-layer K/V entries.
- Labels: request, generated token, layer, key, value, cache block.
- Source anchors: `llmsys-24-kv-cache-serving-state`
- Production note: Keep formulas in prose unless conditions are present.

### `fig-13-contiguous-fragmentation`

- Chapter: 13
- Purpose: Show internal and external fragmentation from max-length contiguous reservations.
- Core message: Reserving contiguous memory for possible maximum sequence length can leave unused holes and stranded capacity.
- Visual form: Memory strip.
- Layout: memory bars with allocated regions, unused reserved tails, and gaps between requests.
- Labels: reserved region, used tokens, internal fragmentation, external fragmentation, free hole.
- Source anchors: `llmsys-24-contiguous-preallocation-fragmentation`
- Production note: Avoid exact waste percentages unless sourced with workload assumptions.

### `fig-13-kv-blocks`

- Chapter: 13
- Purpose: Show a sequence split into fixed-size logical KV blocks.
- Core message: PagedAttention virtualizes KV storage by dividing each sequence into logical blocks.
- Visual form: Block diagram.
- Layout: token sequence divided into equal logical blocks, with the last block partially filled.
- Labels: logical block, token, partially filled block, block size.
- Source anchors: `llmsys-24-kv-block-definition`
- Production note: If showing block size, use symbolic `B` unless the chapter states a system-specific value.

### `fig-13-block-table`

- Chapter: 13
- Purpose: Show logical blocks mapped to non-contiguous physical blocks.
- Core message: A block table decouples sequence order from physical KV memory placement.
- Visual form: Address-translation diagram.
- Layout: logical block table on left points to physical memory blocks scattered in a memory pool on right.
- Labels: logical block, block table, physical block, memory pool, indirection.
- Source anchors: `llmsys-24-block-table-virtualization`
- Production note: Make the analogy to virtual memory visual but do not imply OS page-table mechanics are identical.

### `fig-13-pagedattention-kernel`

- Chapter: 13
- Purpose: Show attention reading non-contiguous blocks through block-table indirection.
- Core message: The attention kernel follows block-table mappings to gather the sequence's KV blocks even when they are not physically contiguous.
- Visual form: Dataflow.
- Layout: query vector enters attention kernel; kernel reads block table; arrows fetch K/V blocks from non-contiguous memory; output is produced.
- Labels: query, block table lookup, KV block, physical memory, attention kernel.
- Source anchors: `llmsys-24-pagedattention-kernel`
- Production note: Keep implementation detail bounded; this figure explains access pattern, not low-level CUDA scheduling.

### `fig-13-fragmentation-boundary`

- Chapter: 13
- Purpose: Show waste limited to the final partially filled block.
- Core message: Block allocation changes fragmentation from max-length reservation waste to at most unused space in active boundary blocks under the stated block model.
- Visual form: Before/after memory diagram.
- Layout: contiguous reservation with large unused tail beside block allocation with only the last block partially empty.
- Labels: contiguous reservation, block allocation, final block, unused slots, fragmentation boundary.
- Source anchors: `llmsys-24-pagedattention-fragmentation`
- Production note: The phrase "at most" should travel with the fixed-block allocation assumption.

### `fig-13-prefix-sharing`

- Chapter: 13
- Purpose: Show multiple samples sharing prompt-prefix KV blocks before divergence.
- Core message: Reference-counted KV blocks let related sequences share prompt state until their generated continuations diverge.
- Visual form: Tree/block reference diagram.
- Layout: shared prefix blocks branch into separate continuation blocks; block references point to shared physical blocks.
- Labels: shared prefix, reference count, branch, continuation, copy-on-write.
- Source anchors: `llmsys-24-kv-block-sharing`
- Production note: Keep the branch example small; this is a mechanism figure, not a decoding-quality claim.

### `fig-13-preemption-recovery`

- Chapter: 13
- Purpose: Compare swap-to-CPU and recompute recovery paths.
- Core message: When memory pressure interrupts a request, the system can preserve KV state externally or discard and recompute it, each with different resource costs.
- Visual form: Two-lane flow.
- Layout: lane one swaps KV blocks to CPU and restores; lane two evicts and recomputes prefix state before resuming.
- Labels: preempt, swap, restore, recompute, resume, memory pressure.
- Source anchors: `llmsys-24-preemption-recovery`
- Production note: Avoid saying one path is always better; costs depend on bandwidth, compute, and request state.

### `fig-13-vllm-engine-boundary`

- Chapter: 13
- Purpose: Place PagedAttention inside vLLM's API, scheduler, KV manager, worker, and kernels.
- Core message: PagedAttention is one mechanism inside a larger serving engine boundary.
- Visual form: Architecture block diagram.
- Layout: API/input queue -> scheduler -> KV manager/block allocator -> worker/model executor -> PagedAttention kernel.
- Labels: API, scheduler, KV manager, block table, worker, kernel, token output.
- Source anchors: `llmsys-24-vllm-optimization-areas`
- Production note: Keep product/version-specific names aligned with source cards and chapter text.

## Chapter 14

### `fig-14-throughput-vs-goodput`

- Chapter: 14
- Purpose: Show raw throughput versus requests completed within SLO.
- Core message: Serving quality is not only tokens per second; work that misses latency objectives may not count as useful goodput.
- Visual form: Timeline/counting diagram.
- Layout: request completions across a deadline boundary; count all completions versus SLO-satisfying completions.
- Labels: throughput, goodput, SLO, completed request, missed deadline.
- Source anchors: `llmsys-29-goodput-definition`
- Production note: Use symbolic request counts; no performance numbers.

### `fig-14-ttft-tpot`

- Chapter: 14
- Purpose: Show first-token latency and inter-token latency on a streamed response.
- Core message: Streaming LLM latency has at least two user-visible components: time to first token and time per output token.
- Visual form: Token timeline.
- Layout: request arrival, prefill interval, first token, repeated decode intervals, later tokens.
- Labels: request arrival, TTFT, TPOT, prefill, decode, streamed token.
- Source anchors: `llmsys-29-ttft-tpot-slos`
- Production note: Define acronyms in caption or surrounding prose.

### `fig-14-prefill-decode-interference`

- Chapter: 14
- Purpose: Show colocated prefill/decode creating wasted time.
- Core message: Prefill and decode can compete for the same resources even though their compute and latency profiles differ.
- Visual form: Two-lane timeline.
- Layout: prefill jobs and decode jobs sharing a worker; show decode waiting behind long prefill and prefill affected by decode scheduling.
- Labels: prefill, decode, queueing, interference, wasted time.
- Source anchors: `llmsys-29-colocation-interference`
- Production note: Keep "wasted time" qualitative unless a source-specific workload is named.

### `fig-14-pd-disaggregation`

- Chapter: 14
- Purpose: Show prefill pool, KV transfer, and decode pool.
- Core message: Disaggregated serving separates prefill and decode workers and moves KV state between them.
- Visual form: Architecture diagram.
- Layout: request enters prefill pool, KV state transfers over a data path, decode pool streams output tokens.
- Labels: prefill worker, KV transfer, decode worker, router, token stream.
- Source anchors: `llmsys-29-disaggregation-opportunity`, `llmsys-26-dynamo-disaggregated-serving`
- Production note: Show the transfer path explicitly; otherwise disaggregation looks like ordinary load balancing.

### `fig-14-placement-problem`

- Chapter: 14
- Purpose: Show choices for instance count, parallelism, and physical placement.
- Core message: Serving performance depends on how model instances, parallelism groups, and network locality are placed across hardware.
- Visual form: Cluster diagram.
- Layout: racks or nodes containing GPUs; annotate possible prefill/decode replicas and parallelism groups.
- Labels: instance, GPU, node, rack, parallelism group, placement.
- Source anchors: `llmsys-29-distserve-placement`
- Production note: Avoid a specific cluster topology unless sourced.

### `fig-14-kv-aware-routing`

- Chapter: 14
- Purpose: Show router choosing workers by KV hit and load.
- Core message: A serving router can consider both load and cache locality because routing affects future KV reuse.
- Visual form: Routing diagram.
- Layout: incoming requests with prefixes, router, workers with different cached prefixes and queue depths.
- Labels: router, KV hit, queue load, prefix, worker, routing decision.
- Source anchors: `llmsys-26-kv-aware-routing`
- Production note: Do not imply cache locality dominates load in all policies.

### `fig-14-kv-memory-hierarchy`

- Chapter: 14
- Purpose: Show HBM, CPU DRAM, SSD, and network storage tiers.
- Core message: KV cache management becomes a memory hierarchy problem when hot state cannot all remain in GPU memory.
- Visual form: Memory hierarchy.
- Layout: tiers from HBM to CPU DRAM to SSD to remote/network store, with latency/capacity tradeoff arrows but no numbers.
- Labels: HBM, CPU DRAM, SSD, remote store, hot KV, cold KV, offload.
- Source anchors: `llmsys-26-memory-tiers-kv-offload`, `llmsys-28-distributed-kv-pool`
- Production note: Do not include latency or bandwidth numbers without hardware and system conditions.

### `fig-14-external-kv-manager`

- Chapter: 14
- Purpose: Show inference engines connected to LMCache or a remote KV pool.
- Core message: KV state can be managed outside a single engine process, enabling sharing, offload, or reuse across workers.
- Visual form: Block diagram.
- Layout: multiple inference engines connect to an external KV manager/service and memory backends.
- Labels: inference engine, KV manager, remote KV pool, CPU memory, zero-copy sharing, cache lookup.
- Source anchors: `llmsys-27-lmcache-separated-service`, `llmsys-27-zero-copy-cpu-sharing`
- Production note: Keep system names source-specific; avoid suggesting one external KV architecture is universal.

### `fig-14-transfer-substrate`

- Chapter: 14
- Purpose: Show transfer layer between memory domains and nodes.
- Core message: Disaggregated serving needs an explicit data-movement substrate to move KV blocks across devices, host memory, and remote storage.
- Visual form: Data-movement diagram.
- Layout: GPU memory, host memory, remote node/store connected by a transfer layer with arrows for KV movement.
- Labels: transfer layer, GPU memory, host memory, remote store, KV block, network.
- Source anchors: `llmsys-26-nixl-transfer-layer`, `llmsys-28-mooncake-store-integration`
- Production note: Avoid protocol-level detail unless the chapter names the specific substrate.

## Chapter 15

### `fig-15-codesign-loop`

- Chapter: 15
- Purpose: Show model architecture, algorithm, software/runtime, and hardware acceleration as a loop.
- Core message: LLM-system improvements often move through feedback among model design, algorithmic change, runtime software, and hardware capability.
- Visual form: Four-node loop.
- Layout: four labeled nodes in a cycle with cross-links for memory, communication, scheduling, and precision pressure.
- Labels: model architecture, algorithm, software/runtime, hardware, feedback, bottleneck.
- Source anchors: `llmsys-01-codesign`
- Production note: This should echo `fig-01-codesign-loop` while showing the reader's expanded vocabulary.

### `fig-15-stack-recap`

- Chapter: 15
- Purpose: Recap the book stack from objective to serving infrastructure.
- Core message: The book's chapters form one system stack rather than independent topics.
- Visual form: Layered stack.
- Layout: probability objective and tokens at top, Transformer computation, kernels/compilers, training systems, compression, inference/serving, hardware and network substrate.
- Labels: objective, tokens, Transformer, kernels, compiler, distributed training, compression, inference scheduler, KV cache, serving infrastructure.
- Source anchors: `llmsys-01-system-challenges`
- Production note: Keep layer names aligned with chapter titles.

### `fig-15-bottleneck-routing`

- Chapter: 15
- Purpose: Show how a bottleneck routes to compute, memory, communication, scheduling, or abstraction fixes.
- Core message: Systems reasoning starts by identifying the active bottleneck, then choosing the class of intervention that can actually affect it.
- Visual form: Decision flow.
- Layout: start with observed symptom, branch to compute-bound, memory-bound, communication-bound, scheduling-bound, or abstraction-bound; each branch lists representative interventions.
- Labels: symptom, compute, memory, communication, scheduling, abstraction, intervention.
- Source anchors: `llmsys-01-system-challenges`
- Production note: Do not make the flowchart look deterministic; real diagnosis can require measurement and iteration.

### `fig-15-optimization-moves-bottleneck`

- Chapter: 15
- Purpose: Show one optimization shifting pressure to another resource.
- Core message: A successful optimization often exposes the next bottleneck instead of ending the systems problem.
- Visual form: Cause/effect diagram.
- Layout: before state with one dominant bottleneck, optimization arrow, after state with a different resource becoming limiting.
- Labels: before, optimization, after, memory pressure, communication pressure, scheduling pressure, new bottleneck.
- Source anchors: Synthesis from Chapters 7-14
- Production note: Use a generic example such as reducing compute time exposing communication or KV-transfer pressure; avoid unqualified speedup claims.

## Cross-Chapter Notes

- Use consistent colors for prefill, decode, KV cache, scheduler, and external memory tiers across Chapters 12-14.
- Show KV cache as state with ownership and lifetime, not as a passive tensor blob.
- Serving figures should distinguish throughput, goodput, TTFT, and TPOT visually and terminologically.
- Any numeric latency, throughput, memory, or bandwidth labels require source-card-backed conditions.
- Chapter 15 figures should reuse visual motifs from earlier chapters to create closure rather than introduce new notation.

Owner: Book Architect  
Purpose: Part IV figure planning  
Evidence grade: A for structural decisions derived from chapter briefs and source-ledger anchors; synthesis figure inherits evidence from Chapters 7-14  
Assumptions: Figures are technical book diagrams and should avoid unsourced performance numbers  
Open questions: Whether Chapter 15 should include a chapter-to-bottleneck table in addition to the four planned figures  
Handoff: Technical reviewer for caption review, then illustrator for first-pass diagrams
