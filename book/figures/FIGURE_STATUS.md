# Figure Production Status

Date: 2026-07-06

Scope: required figures declared in `book/chapters/*-brief.md`, plus existing figure specification files under `book/figures/`.

## Summary

- Brief-declared required figures: 88.
- Existing figure spec files: `part-i-figure-specs.md`, `part-ii-figure-specs.md`, `part-iii-figure-specs.md`, `part-iv-figure-specs.md`.
- Caption review completed: all parts, in `process/review-memos/part-i-figure-caption-review-2026-07-05.md`, `process/review-memos/part-ii-figure-caption-review-2026-07-04.md`, `process/review-memos/part-iii-figure-caption-review-2026-07-05.md`, and `process/review-memos/part-iv-figure-caption-review-2026-07-05.md`.
- Current production state: Chapters 1-5 artwork complete; twenty-two editable SVG assets are present under `book/figures/artwork/`.

## Reconciliation Items

| Item | Status | Action |
| --- | --- | --- |
| `fig-01-abstraction-levels` appeared in `part-i-figure-specs.md` but not Chapter 1 brief. | resolved | Added to Chapter 1 brief because the Part I spec already treats it as a core abstraction-ladder figure. |
| `fig-03-eagle-feature-loop` appeared in `part-i-figure-specs.md` but not Chapter 3 brief. | resolved | Added to Chapter 3 brief as an advanced-sidebar figure, matching the Chapter 3 draft treatment of EAGLE. |
| Chapter 5 brief named `fig-05-roofline-intuition`; Part II specs and caption review named `fig-05-memory-bound-intuition`. | resolved | Renamed the Chapter 5 brief row to `fig-05-memory-bound-intuition`, preserving the qualitative memory-bound/computation-bound intent. |

## Production Readiness by Part

| Part | Chapters | Brief inventory | Specs | Caption review | Artwork state | Next action |
| --- | --- | --- | --- | --- | --- | --- |
| I | 1-3 | complete | drafted | passed | Chapters 1-3 reviewed | Complete. |
| II | 4-6 | complete | drafted | passed | Chapters 4-5 reviewed | Produce Chapter 6 diagrams. |
| III | 7-11 | complete | drafted | passed | not started | Begin diagram production. |
| IV | 12-15 | complete | drafted | passed | not started | Begin diagram production. |

## Canonical Brief Inventory

### Chapter 1 — `01-why-llm-systems`

| Figure ID | Purpose | Form | Source | State |
| --- | --- | --- | --- | --- |
| `fig-01-visible-vs-hidden-stack` | Contrast LLM user capabilities with system layers. | Stack diagram | `llmsys-01-system-challenges` | artwork-reviewed |
| `fig-01-token-probability-chain` | Show sequence probability as chained next-token predictions. | Token chain | `llmsys-01-next-token-probability` | artwork-reviewed |
| `fig-01-abstraction-levels` | Show the abstraction ladder from model behavior to kernels and hardware. | Layered architecture diagram | `llmsys-01-system-challenges` | artwork-reviewed |
| `fig-01-codesign-loop` | Show model, algorithm, software, hardware feedback loop. | Loop diagram | `llmsys-01-codesign` | artwork-reviewed |

### Chapter 2 — `02-tokens-probability-transformers`

| Figure ID | Purpose | Form | Source | State |
| --- | --- | --- | --- | --- |
| `fig-02-token-to-logits` | Show token -> embedding -> blocks -> logits. | Dataflow | `llmsys-06-transformer-components` | artwork-reviewed |
| `fig-02-qkv-attention` | Explain Q/K/V projections and weighted value mixing. | Dataflow + tiny matrix | `llmsys-06-transformer-components` | artwork-reviewed |
| `fig-02-causal-mask` | Show future-token masking before softmax. | Attention matrix | `llmsys-06-masked-self-attention` | artwork-reviewed |
| `fig-02-model-family-map` | Compare encoder-only, encoder-decoder, decoder-only. | Table | `llmsys-01-decoder-only`; `llmsys-07-t5-text-to-text`; `llmsys-07-llama-architecture` | artwork-reviewed |

### Chapter 3 — `03-tokenization-context-decoding`

| Figure ID | Purpose | Form | Source | State |
| --- | --- | --- | --- | --- |
| `fig-03-tokenizer-comparison` | Compare word/char/subword tokenization on one sentence. | Table | `llmsys-08-tokenization-tradeoffs` | artwork-reviewed |
| `fig-03-bpe-loop` | Show BPE merge loop. | Step diagram | `llmsys-08-bpe-algorithm` | artwork-reviewed |
| `fig-03-decode-methods` | Compare greedy, sampling, beam. | Search diagram | `llmsys-09-beam-search` | artwork-reviewed |
| `fig-03-speculative-pipeline` | Show draft model and target model validation. | Pipeline | `llmsys-09-speculative-decoding` | artwork-reviewed |
| `fig-03-eagle-feature-loop` | Show EAGLE as feature-level speculative decoding in an advanced sidebar. | Feature prediction loop | `li-2024-eagle`, `llmsys-09-speculative-decoding` | artwork-reviewed |

### Chapter 4 — `04-gpu-programming-model`

| Figure ID | Purpose | Form | Source | State |
| --- | --- | --- | --- | --- |
| `fig-04-host-device-lifecycle` | Show CPU allocation/copy/launch/copy/free flow. | Sequence diagram | `llmsys-03-cuda-memory-lifecycle` | artwork-reviewed |
| `fig-04-grid-block-thread` | Show grid, blocks, threads, and one thread's global index. | Hierarchy diagram | `llmsys-02-warp-execution`, `llmsys-03-thread-indexing` | artwork-reviewed |
| `fig-04-sm-warp-scheduling` | Show blocks assigned to SMs and warps executing inside an SM. | Execution schematic | `llmsys-02-gpu-architecture`, `llmsys-02-warp-execution` | artwork-reviewed |
| `fig-04-memory-paths` | Show system memory, PCIe/NVLink, GPU memory, L2, shared memory, registers. | Memory hierarchy diagram | `llmsys-02-cpu-gpu-data-movement`, `llmsys-02-gpu-server-components` | artwork-reviewed |

### Chapter 5 — `05-kernels-memory-transformer-blocks`

| Figure ID | Purpose | Form | Source | State |
| --- | --- | --- | --- | --- |
| `fig-05-memory-bound-intuition` | Show memory-bound vs compute-bound without full roofline formalism. | Two-region sketch | `llmsys-04-memory-access-efficiency` | artwork-reviewed |
| `fig-05-naive-vs-tiled-matmul` | Compare global-memory reloads with shared-memory tile reuse. | Side-by-side dataflow | `llmsys-04-tiling-shared-memory` | artwork-reviewed |
| `fig-05-coalescing-transpose` | Show coalesced and uncoalesced memory access in matrix transpose. | Warp access diagram | `llmsys-04-coalesced-access` | artwork-reviewed |
| `fig-05-transformer-operator-map` | Map Transformer block operations to GEMM, elementwise, reduction, and memory reuse. | Annotated block diagram | `llmsys-10-transformer-operator-stack` | artwork-reviewed |
| `fig-05-kernel-fusion` | Show two elementwise kernels versus one fused kernel. | Before/after execution trace | `llmsys-10-kernel-fusion` | artwork-reviewed |

### Chapter 6 — `06-flashattention-transformer-acceleration`

| Figure ID | Purpose | Form | Source | State |
| --- | --- | --- | --- | --- |
| `fig-06-standard-attention-hbm` | Show standard attention writing `S` and `P` to HBM. | Dataflow diagram | `dao-2022-standard-attention-materialization` | caption-reviewed |
| `fig-06-flashattention-tiling` | Show Q/K/V tiles moving between HBM and SRAM. | Blocked matrix diagram | `dao-2022-flashattention-tiling` | caption-reviewed |
| `fig-06-online-softmax` | Show blockwise max/sum update and output rescaling. | Step diagram | `dao-2022-online-softmax` | caption-reviewed |
| `fig-06-forward-backward-memory` | Compare storing attention matrix versus storing normalization stats and recomputing. | Before/after memory diagram | `dao-2022-backward-recomputation` | caption-reviewed |
| `fig-06-modern-hardware-sidebar` | Summarize FA3/FA4 as hardware-aware evolution. | Sidebar timeline | `llmsys-21-modern-hardware-attention` | caption-reviewed |

### Chapter 7 — `07-dl-frameworks-and-compilers`

| Figure ID | Purpose | Form | Source | State |
| --- | --- | --- | --- | --- |
| `fig-07-python-to-device-program` | Show the full path from Python function to compiled accelerator executable. | Pipeline diagram | `llmsys-12-xla-compilation-pipeline` | specified |
| `fig-07-forward-to-backward-graph` | Explain autodiff as a program transformation. | Small computation graph with backward edges | `llmsys-05-automatic-differentiation` | specified |
| `fig-07-ir-stack` | Locate JAXpr, StableHLO, HLO, backend IR, and executable. | Layered stack diagram | `llmsys-12-jax-tracing-jaxpr`, `llmsys-12-stablehlo-portability` | specified |
| `fig-07-xla-memory-planning` | Show why shape/layout/fusion choices affect buffers and HBM traffic. | Dataflow and memory-layout diagram | `llmsys-12-xla-optimization-passes` | specified |
| `fig-07-tpu-systolic-compile-target` | Connect HLO tiling/layout to TPU matrix execution. | Systolic-array sketch | `llmsys-12-tpu-systolic-mxu` | specified |
| `fig-07-pallas-blockspec-grid` | Explain grid coordinates, BlockSpec slices, VMEM refs, and HBM tensors. | Kernel tiling diagram | `llmsys-13-pallas-blockspec` | specified |
| `fig-07-pallas-pipeline` | Show overlapped HBM-to-VMEM transfer and compute. | Timeline diagram | `llmsys-13-pallas-pipelining` | specified |

### Chapter 8 — `08-distributed-training-ddp`

| Figure ID | Purpose | Form | Source | State |
| --- | --- | --- | --- | --- |
| `fig-08-data-parallel-step` | Show replicas, data shards, local gradients, all-reduce, local optimizer update. | Step diagram | `llmsys-14-data-parallel-allreduce` | specified |
| `fig-08-collective-semantics` | Compare broadcast, reduce, all-reduce, reduce-scatter, all-gather. | Collective table/diagram | `llmsys-14-collective-primitives`, `llmsys-14-allreduce-semantics` | specified |
| `fig-08-ring-allreduce` | Show scatter-reduce and all-gather phases over chunks. | Ring diagram | `llmsys-14-ring-allreduce-phases` | specified |
| `fig-08-naive-vs-overlap-ddp` | Compare post-backward all-reduce with bucketed overlap. | Timeline | `llmsys-15-ddp-naive-allreduce`, `llmsys-15-ddp-overlap` | specified |
| `fig-08-ddp-reducer-hooks` | Show autograd hooks, bucket pending counts, and async all-reduce trigger. | Control-flow diagram | `llmsys-15-ddp-autograd-hooks` | specified |

### Chapter 9 — `09-model-parallelism`

| Figure ID | Purpose | Form | Source | State |
| --- | --- | --- | --- | --- |
| `fig-09-data-vs-model-parallel` | Contrast data-parallel replication with model partitioning. | Side-by-side architecture diagram | `llmsys-16-model-parallel-motivation` | specified |
| `fig-09-naive-pipeline-bubble` | Show idle devices in naive layer-wise pipeline. | Timeline | `llmsys-16-naive-pipeline-idle` | specified |
| `fig-09-gpipe-microbatches` | Show micro-batches filling pipeline stages. | Timeline | `llmsys-16-gpipe-microbatching` | specified |
| `fig-09-1f1b-vs-gpipe` | Compare all-forward-then-backward with 1F1B. | Timeline | `llmsys-16-one-f-one-b` | specified |
| `fig-09-tensor-parallel-ffn` | Show column/row split of FFN projections. | Matrix partition diagram | `llmsys-16-tensor-parallel-ffn` | specified |
| `fig-09-tensor-parallel-attention-heads` | Show attention head partitioning. | Block diagram | `llmsys-16-tensor-parallel-attention` | specified |
| `fig-09-3d-parallelism` | Show tensor × pipeline × data parallel axes. | 3D grid or nested groups | `llmsys-16-parallelism-composition` | specified |

### Chapter 10 — `10-zero-moe-and-memory`

| Figure ID | Purpose | Form | Source | State |
| --- | --- | --- | --- | --- |
| `fig-10-ddp-memory-ledger` | Show what a DDP worker stores. | Stacked memory ledger | `llmsys-18-ddp-memory-accounting` | specified |
| `fig-10-zero-stages` | Compare ZeRO-1/2/3 partitioned state. | Three-row state partition diagram | `llmsys-18-zero-key-idea` | specified |
| `fig-10-zero3-parameter-gather` | Show parameter shard availability during forward/backward. | Timeline/dataflow | `llmsys-18-zero-stage3-parameters` | specified |
| `fig-10-moe-router-experts` | Show token routing from FFN input to selected experts. | Block diagram | `llmsys-17-moe-ffn-router` | specified |
| `fig-10-expert-parallel-alltoall` | Show token dispatch and return across expert devices. | All-to-all diagram | `llmsys-17-expert-parallelism` | specified |
| `fig-10-load-balance-skew` | Show router collapse versus balanced expert use. | Histogram or token flow | `llmsys-17-moe-load-balancing` | specified |

### Chapter 11 — `11-quantization-and-peft`

| Figure ID | Purpose | Form | Source | State |
| --- | --- | --- | --- | --- |
| `fig-11-quantization-map` | Show float values mapped to low-bit buckets with scale and optional zero point. | Number line | `llmsys-19-absmax-zeropoint` | specified |
| `fig-11-quantization-error` | Distinguish rounding error, clipping, and range mismatch. | Three small plots | `llmsys-19-direct-quantization-errors` | specified |
| `fig-11-llmint8-outlier-path` | Show normal 8-bit path plus higher-precision outlier path. | Dataflow | `llmsys-19-llmint8-outliers` | specified |
| `fig-11-gptq-compensation` | Show quantize column, compute error, update remaining columns. | Matrix block diagram | `llmsys-20-gptq-blockwise-compensation` | specified |
| `fig-11-full-ft-vs-lora-state` | Compare full fine-tuning state with LoRA trainable state. | Memory ledger | `llmsys-23-full-finetuning-cost`, `llmsys-23-lora-training-state` | specified |
| `fig-11-lora-update` | Show `W' = W0 + A B` with low-rank matrices. | Matrix factor diagram | `llmsys-23-lora-lowrank-update` | specified |
| `fig-11-qlora-stack` | Show quantized frozen base plus trainable adapters. | Layer stack | `llmsys-23-qlora-quantized-lora` | specified |

### Chapter 12 — `12-inference-cost-model`

| Figure ID | Purpose | Form | Source | State |
| --- | --- | --- | --- | --- |
| `fig-12-prefill-decode` | Show prompt prefill followed by token-by-token decode. | Timeline/dataflow | `yu-2022-orca-autoregressive-iterations` | specified |
| `fig-12-request-level-batching-problem` | Show short requests waiting for longest generation. | Timeline | `llmsys-22-naive-batch-longest` | specified |
| `fig-12-continuous-batching` | Show batch membership changing between decode iterations. | Timeline | `llmsys-22-continuous-batching`, `yu-2022-orca-iteration-level-scheduling` | specified |
| `fig-12-scheduler-loop` | Show scheduler receive/run/result/free loop. | Loop diagram | `llmsys-22-request-scheduler` | specified |
| `fig-12-selective-batching` | Show non-attention batching with special attention path. | Block diagram | `llmsys-22-selective-batching` | specified |
| `fig-12-kv-cache-cost` | Show per-layer KV cache accumulating with tokens. | Memory growth diagram | `llmsys-22-kv-cache-need` | specified |
| `fig-12-prefix-cache-routing` | Show prompt prefix reuse and cache-aware scheduling. | Prefix tree + queue | `llmsys-22-radixattention-prefix-cache`, `llmsys-22-cache-aware-scheduling` | specified |

### Chapter 13 — `13-kv-cache-vllm-pagedattention`

| Figure ID | Purpose | Form | Source | State |
| --- | --- | --- | --- | --- |
| `fig-13-kv-cache-growth` | Show KV state accumulating per generated token across layers. | Memory growth diagram | `llmsys-24-kv-cache-serving-state` | specified |
| `fig-13-contiguous-fragmentation` | Show internal and external fragmentation from max-length contiguous reservations. | Memory strip | `llmsys-24-contiguous-preallocation-fragmentation` | specified |
| `fig-13-kv-blocks` | Show a sequence split into fixed-size logical KV blocks. | Block diagram | `llmsys-24-kv-block-definition` | specified |
| `fig-13-block-table` | Show logical blocks mapped to non-contiguous physical blocks. | Address-translation diagram | `llmsys-24-block-table-virtualization` | specified |
| `fig-13-pagedattention-kernel` | Show attention reading non-contiguous blocks through block-table indirection. | Dataflow | `llmsys-24-pagedattention-kernel` | specified |
| `fig-13-fragmentation-boundary` | Show waste limited to the final partially filled block. | Before/after memory diagram | `llmsys-24-pagedattention-fragmentation` | specified |
| `fig-13-prefix-sharing` | Show multiple samples sharing prompt-prefix KV blocks before divergence. | Tree/block reference diagram | `llmsys-24-kv-block-sharing` | specified |
| `fig-13-preemption-recovery` | Compare swap-to-CPU and recompute recovery paths. | Two-lane flow | `llmsys-24-preemption-recovery` | specified |
| `fig-13-vllm-engine-boundary` | Place PagedAttention inside vLLM's API, scheduler, KV manager, worker, and kernels. | Architecture block diagram | `llmsys-24-vllm-optimization-areas` | specified |

### Chapter 14 — `14-serving-scheduling-and-disaggregation`

| Figure ID | Purpose | Form | Source | State |
| --- | --- | --- | --- | --- |
| `fig-14-throughput-vs-goodput` | Show raw throughput versus requests completed within SLO. | Timeline/counting diagram | `llmsys-29-goodput-definition` | specified |
| `fig-14-ttft-tpot` | Show first-token latency and inter-token latency on a streamed response. | Token timeline | `llmsys-29-ttft-tpot-slos` | specified |
| `fig-14-prefill-decode-interference` | Show colocated prefill/decode creating wasted time. | Two-lane timeline | `llmsys-29-colocation-interference` | specified |
| `fig-14-pd-disaggregation` | Show prefill pool, KV transfer, and decode pool. | Architecture diagram | `llmsys-29-disaggregation-opportunity`, `llmsys-26-dynamo-disaggregated-serving` | specified |
| `fig-14-placement-problem` | Show choices for instance count, parallelism, and physical placement. | Cluster diagram | `llmsys-29-distserve-placement` | specified |
| `fig-14-kv-aware-routing` | Show router choosing workers by KV hit and load. | Routing diagram | `llmsys-26-kv-aware-routing` | specified |
| `fig-14-kv-memory-hierarchy` | Show HBM, CPU DRAM, SSD, network storage tiers. | Memory hierarchy | `llmsys-26-memory-tiers-kv-offload`, `llmsys-28-distributed-kv-pool` | specified |
| `fig-14-external-kv-manager` | Show inference engines connected to LMCache/remote KV pool. | Block diagram | `llmsys-27-lmcache-separated-service`, `llmsys-27-zero-copy-cpu-sharing` | specified |
| `fig-14-transfer-substrate` | Show transfer layer between memory domains and nodes. | Data-movement diagram | `llmsys-26-nixl-transfer-layer`, `llmsys-28-mooncake-store-integration` | specified |

### Chapter 15 — `15-llm-system-codesign`

| Figure ID | Purpose | Form | Source | State |
| --- | --- | --- | --- | --- |
| `fig-15-codesign-loop` | Show model architecture, algorithm, software/runtime, and hardware acceleration as a loop. | Four-node loop | `llmsys-01-codesign` | specified |
| `fig-15-stack-recap` | Recap the book stack from objective to serving infrastructure. | Layered stack | `llmsys-01-system-challenges` | specified |
| `fig-15-bottleneck-routing` | Show how a bottleneck routes to compute/memory/communication/scheduling/abstraction fixes. | Decision flow | `llmsys-01-system-challenges` | specified |
| `fig-15-optimization-moves-bottleneck` | Show one optimization shifting pressure to another resource. | Cause/effect diagram | Synthesis from Chapters 7-14 | specified |

## State Definitions

- `specified`: figure appears in the chapter brief and has purpose/form/source metadata.
- `caption-reviewed`: caption text has been reviewed for source alignment and overclaim risk.
- `needs ID reconciliation`: figure intent is present, but its identifier differs across planning artifacts.
- `artwork-draft`: rendered diagram or editable source exists.
- `artwork-reviewed`: rendered output has passed terminology, citation, and overclaim review.

Owner: Book Architect  
Purpose: Figure production inventory and resumability ledger  
Evidence grade: A for inventory extracted from chapter briefs and existing figure specs; figure content still inherits source-card evidence from each row  
Assumptions: Brief tables are canonical after the 2026-07-05 reconciliation pass  
Open questions: Whether final publication layout requires a second export format in addition to editable SVG
Handoff: Next owner should produce Chapter 2 diagrams and reuse the Chapter 1 type, color, line-weight, and accessibility conventions
