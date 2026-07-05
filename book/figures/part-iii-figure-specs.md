# Part III Figure Specs

Scope: Chapters 7-11

These specs cover framework/compiler execution, distributed training, model parallelism, optimizer-state partitioning, MoE routing, quantization, and parameter-efficient fine-tuning. The goal is to make hidden program transformations, communication, memory ownership, and compression error visible without turning the figures into benchmark claims.

## Chapter 7

### `fig-07-python-to-device-program`

- Chapter: 7
- Purpose: Show the path from a Python model function to a compiled accelerator executable.
- Core message: A framework does not send Python directly to the accelerator; it traces or captures a computation, lowers it through IRs, optimizes it, and dispatches executable device work.
- Visual form: Pipeline diagram.
- Layout: Python function -> traced graph/JAXpr -> StableHLO/HLO -> optimized HLO -> backend code/executable -> device execution.
- Labels: Python, tracing, graph/IR, lowering, optimization, backend, executable, accelerator.
- Source anchors: `llmsys-12-xla-compilation-pipeline`
- Production note: Keep this framework-neutral in the title, but use JAX/XLA labels where the chapter does.

### `fig-07-forward-to-backward-graph`

- Chapter: 7
- Purpose: Explain autodiff as a program transformation.
- Core message: The backward pass is generated from the forward computation graph and reuses saved values where gradients require them.
- Visual form: Small computation graph with backward edges.
- Layout: forward nodes on top, loss at right, reverse-mode gradient arrows flowing back through the graph.
- Labels: forward value, loss, gradient, saved activation, backward rule.
- Source anchors: `llmsys-05-automatic-differentiation`
- Production note: Use a tiny graph such as multiply/add or matmul/add; avoid full Transformer complexity.

### `fig-07-ir-stack`

- Chapter: 7
- Purpose: Locate JAXpr, StableHLO, HLO, backend IR, and executable in the compilation stack.
- Core message: Intermediate representations separate user intent, portable operator semantics, compiler optimization, and hardware-specific code generation.
- Visual form: Layered stack diagram.
- Layout: user Python at top, JAXpr/traced graph below, StableHLO/HLO middle, backend IR and executable near bottom, device hardware at bottom.
- Labels: JAXpr, StableHLO, HLO, backend IR, executable, portability boundary, hardware-specific boundary.
- Source anchors: `llmsys-12-jax-tracing-jaxpr`, `llmsys-12-stablehlo-portability`
- Production note: Do not imply every framework uses this exact stack; caption should name JAX/XLA where applicable.

### `fig-07-xla-memory-planning`

- Chapter: 7
- Purpose: Show why shape, layout, fusion, and buffer assignment affect HBM traffic.
- Core message: Compiler choices can remove intermediate buffers or change when tensors occupy memory.
- Visual form: Dataflow and memory-layout diagram.
- Layout: unfused graph with two intermediate buffers beside a fused/optimized graph with fewer live buffers; include a small memory timeline.
- Labels: buffer, layout, fusion, live range, HBM read, HBM write.
- Source anchors: `llmsys-12-xla-optimization-passes`
- Production note: Keep the example qualitative; avoid claiming a universal memory reduction.

### `fig-07-tpu-systolic-compile-target`

- Chapter: 7
- Purpose: Connect HLO tiling/layout to TPU matrix execution.
- Core message: A compiler target matters because matrix work must be shaped for the accelerator's execution structure.
- Visual form: Systolic-array sketch.
- Layout: matrix tile entering a grid of multiply-accumulate cells, with compiler tiling/layout annotations upstream.
- Labels: matrix tile, systolic array, MXU, layout, tile, accumulation.
- Source anchors: `llmsys-12-tpu-systolic-mxu`
- Production note: Say TPU/MXU explicitly; do not generalize the systolic-array picture to all accelerators.

### `fig-07-pallas-blockspec-grid`

- Chapter: 7
- Purpose: Explain grid coordinates, BlockSpec slices, VMEM refs, and HBM tensors.
- Core message: A kernel DSL exposes block-level tiling while still connecting the block program to global tensors.
- Visual form: Kernel tiling diagram.
- Layout: HBM tensor grid, highlighted block selected by program IDs, BlockSpec slice arrow into VMEM refs used by the block program.
- Labels: grid, program ID, BlockSpec, HBM tensor, VMEM ref, block program.
- Source anchors: `llmsys-13-pallas-blockspec`
- Production note: Use a single two-dimensional tensor example; avoid API syntax unless needed for orientation.

### `fig-07-pallas-pipeline`

- Chapter: 7
- Purpose: Show overlapped HBM-to-VMEM transfer and compute.
- Core message: Software pipelining can stage data movement and computation so the next tile is prepared while the current tile computes.
- Visual form: Timeline diagram.
- Layout: lanes for load, compute, store across consecutive tiles; overlapping colored bars.
- Labels: HBM load, VMEM, compute, store, tile `t`, tile `t+1`.
- Source anchors: `llmsys-13-pallas-pipelining`
- Production note: Do not attach speedup values unless a separate benchmark card is added.

## Chapter 8

### `fig-08-data-parallel-step`

- Chapter: 8
- Purpose: Show replicas, data shards, local gradients, all-reduce, and local optimizer update.
- Core message: Data parallelism keeps model replicas synchronized by aggregating gradients across workers.
- Visual form: Step diagram.
- Layout: multiple workers with identical model replicas, different mini-batch shards, local backward pass, all-reduce, local optimizer update.
- Labels: replica, data shard, local gradient, all-reduce, averaged gradient, optimizer step.
- Source anchors: `llmsys-14-data-parallel-allreduce`
- Production note: Make replication explicit; this is not model sharding.

### `fig-08-collective-semantics`

- Chapter: 8
- Purpose: Compare broadcast, reduce, all-reduce, reduce-scatter, and all-gather.
- Core message: Distributed training uses a small vocabulary of collective operations with different input/output ownership patterns.
- Visual form: Collective table/diagram.
- Layout: rows for collectives; columns for before/after tensor ownership across workers.
- Labels: broadcast, reduce, all-reduce, reduce-scatter, all-gather, shard, full tensor.
- Source anchors: `llmsys-14-collective-primitives`, `llmsys-14-allreduce-semantics`
- Production note: Keep tensor sizes symbolic; communication-volume formulas belong in prose only if conditions are stated.

### `fig-08-ring-allreduce`

- Chapter: 8
- Purpose: Show scatter-reduce and all-gather phases over chunks.
- Core message: Ring all-reduce decomposes global aggregation into chunk movement around a logical ring.
- Visual form: Ring diagram.
- Layout: workers arranged in a circle; phase one reduces chunks as they circulate, phase two distributes completed chunks.
- Labels: worker, chunk, scatter-reduce, all-gather, logical ring.
- Source anchors: `llmsys-14-ring-allreduce-phases`
- Production note: Avoid latency or bandwidth numbers; topology and implementation details affect performance.

### `fig-08-naive-vs-overlap-ddp`

- Chapter: 8
- Purpose: Compare post-backward all-reduce with bucketed overlap.
- Core message: DDP overlaps communication with backward computation by reducing gradient buckets as they become ready.
- Visual form: Timeline.
- Layout: two rows: naive backward then all-reduce, and overlapped backward with bucket all-reduces starting earlier.
- Labels: backward compute, gradient bucket, all-reduce, overlap, idle gap.
- Source anchors: `llmsys-15-ddp-naive-allreduce`, `llmsys-15-ddp-overlap`
- Production note: Do not claim overlap always hides all communication; bucket order and network conditions matter.

### `fig-08-ddp-reducer-hooks`

- Chapter: 8
- Purpose: Show autograd hooks, bucket pending counts, and async all-reduce trigger.
- Core message: DDP is coordinated by control flow tied to gradient readiness, not by a single monolithic communication call at the end.
- Visual form: Control-flow diagram.
- Layout: autograd engine emits gradient-ready events; reducer tracks bucket counts; full bucket triggers asynchronous all-reduce; completion writes reduced gradients.
- Labels: autograd hook, reducer, bucket, pending count, async all-reduce, reduced gradient.
- Source anchors: `llmsys-15-ddp-autograd-hooks`
- Production note: Keep PyTorch-specific terms visible if the surrounding section discusses PyTorch DDP.

## Chapter 9

### `fig-09-data-vs-model-parallel`

- Chapter: 9
- Purpose: Contrast data-parallel replication with model partitioning.
- Core message: Data parallelism copies the model across workers, while model parallelism splits model computation or state across workers.
- Visual form: Side-by-side architecture diagram.
- Layout: left side shows identical model replicas and different data shards; right side shows one model split across devices.
- Labels: data parallel, model parallel, replica, shard, activation transfer.
- Source anchors: `llmsys-16-model-parallel-motivation`
- Production note: Use this as the chapter's orientation figure.

### `fig-09-naive-pipeline-bubble`

- Chapter: 9
- Purpose: Show idle devices in naive layer-wise pipeline execution.
- Core message: Splitting layers across devices creates pipeline bubbles when stages wait for inputs or gradients.
- Visual form: Timeline.
- Layout: stages as rows, time as columns, one micro-batch moving through forward then backward with visible idle gaps.
- Labels: stage, forward, backward, idle, pipeline bubble.
- Source anchors: `llmsys-16-naive-pipeline-idle`
- Production note: Keep the timeline small; do not introduce formula notation in the figure.

### `fig-09-gpipe-microbatches`

- Chapter: 9
- Purpose: Show micro-batches filling pipeline stages.
- Core message: Micro-batching improves stage utilization by letting different micro-batches occupy different pipeline stages.
- Visual form: Timeline.
- Layout: several micro-batches flowing through stages; idle gaps shrink after pipeline warmup.
- Labels: micro-batch, stage, warmup, steady state, drain.
- Source anchors: `llmsys-16-gpipe-microbatching`
- Production note: Do not imply micro-batching removes all bubbles or memory costs.

### `fig-09-1f1b-vs-gpipe`

- Chapter: 9
- Purpose: Compare all-forward-then-backward scheduling with one-forward-one-backward scheduling.
- Core message: Pipeline schedule changes activation lifetime and utilization patterns.
- Visual form: Timeline comparison.
- Layout: top row for GPipe-style all-forward-then-backward; bottom row for 1F1B alternating forward/backward after warmup.
- Labels: forward, backward, activation lifetime, warmup, 1F1B.
- Source anchors: `llmsys-16-one-f-one-b`
- Production note: Keep memory implication qualitative unless the chapter gives explicit assumptions.

### `fig-09-tensor-parallel-ffn`

- Chapter: 9
- Purpose: Show column and row split of FFN projections.
- Core message: Tensor parallelism partitions matrix operations and uses collectives to assemble the same logical result.
- Visual form: Matrix partition diagram.
- Layout: input activation feeding split first projection, local nonlinearities, split second projection, collective communication for final result.
- Labels: column split, row split, partial output, all-reduce/all-gather, FFN.
- Source anchors: `llmsys-16-tensor-parallel-ffn`
- Production note: Use symbolic matrices; avoid implementation-specific tensor shapes unless sourced in prose.

### `fig-09-tensor-parallel-attention-heads`

- Chapter: 9
- Purpose: Show attention head partitioning.
- Core message: Multi-head attention creates a natural partition axis, but communication is still needed around shared projections or outputs.
- Visual form: Block diagram.
- Layout: attention heads distributed across devices, local head computation, output projection or gather boundary.
- Labels: head shard, QKV projection, local attention, output projection, communication boundary.
- Source anchors: `llmsys-16-tensor-parallel-attention`
- Production note: Do not imply head partitioning is the only tensor-parallel strategy.

### `fig-09-3d-parallelism`

- Chapter: 9
- Purpose: Show tensor, pipeline, and data parallel axes together.
- Core message: Large training jobs compose parallelism strategies, and each axis creates a different communication or scheduling concern.
- Visual form: 3D grid or nested groups.
- Layout: cube or nested boxes with axes labeled data, pipeline, tensor; show one worker's group memberships.
- Labels: data-parallel group, pipeline stage, tensor-parallel group, replica group.
- Source anchors: `llmsys-16-parallelism-composition`
- Production note: Keep this conceptual; do not encode a specific cluster size unless the text provides conditions.

## Chapter 10

### `fig-10-ddp-memory-ledger`

- Chapter: 10
- Purpose: Show what a DDP worker stores.
- Core message: Training memory includes parameters, gradients, optimizer state, and activations, so replication can be expensive even before communication is considered.
- Visual form: Stacked memory ledger.
- Layout: one worker box containing stacked bars for parameters, gradients, optimizer state, activations, and temporary buffers.
- Labels: parameters, gradients, optimizer state, activations, temporary buffers, replica.
- Source anchors: `llmsys-18-ddp-memory-accounting`
- Production note: Avoid byte counts unless the chapter states model size, precision, optimizer, and activation assumptions.

### `fig-10-zero-stages`

- Chapter: 10
- Purpose: Compare ZeRO-1, ZeRO-2, and ZeRO-3 partitioned state.
- Core message: ZeRO progressively partitions optimizer state, gradients, and parameters across workers.
- Visual form: Three-row state partition diagram.
- Layout: rows for ZeRO stages; columns for parameters, gradients, optimizer state; use repeated vs sharded patterns.
- Labels: replicated, partitioned, optimizer state, gradients, parameters, data-parallel worker.
- Source anchors: `llmsys-18-zero-key-idea`
- Production note: Treat stage numbers as named algorithm stages, not performance guarantees.

### `fig-10-zero3-parameter-gather`

- Chapter: 10
- Purpose: Show parameter shard availability during forward and backward.
- Core message: ZeRO-3 saves memory by keeping parameters sharded, but computation requires gathering needed parameters around layer execution.
- Visual form: Timeline/dataflow.
- Layout: workers hold parameter shards; before a layer executes, shards gather; after use, full parameters are released or repartitioned.
- Labels: parameter shard, all-gather, layer compute, release/repartition, backward.
- Source anchors: `llmsys-18-zero-stage3-parameters`
- Production note: Keep lifetime arrows clear; this figure is about availability over time.

### `fig-10-moe-router-experts`

- Chapter: 10
- Purpose: Show token routing from FFN input to selected experts.
- Core message: MoE replaces a dense FFN path with routing that sends tokens to a subset of expert FFNs.
- Visual form: Block diagram.
- Layout: token representations enter router; router scores choose experts; experts process assigned tokens; outputs combine back into sequence order.
- Labels: token, router, expert, selected expert, combine, FFN.
- Source anchors: `llmsys-17-moe-ffn-router`
- Production note: If using top-k wording, avoid hard capacity or quality claims unless sourced.

### `fig-10-expert-parallel-alltoall`

- Chapter: 10
- Purpose: Show token dispatch and return across expert devices.
- Core message: Expert parallelism turns routing decisions into all-to-all communication before and after expert computation.
- Visual form: All-to-all diagram.
- Layout: devices with local tokens on left, cross-device dispatch to expert owners, expert compute, return path.
- Labels: dispatch, all-to-all, expert device, token permutation, combine.
- Source anchors: `llmsys-17-expert-parallelism`
- Production note: Make communication visible; otherwise MoE looks like a purely local routing trick.

### `fig-10-load-balance-skew`

- Chapter: 10
- Purpose: Show router collapse versus balanced expert use.
- Core message: MoE efficiency depends on keeping expert load reasonably balanced while preserving useful routing.
- Visual form: Histogram or token-flow comparison.
- Layout: left panel shows most tokens routed to one expert; right panel shows tokens spread across experts.
- Labels: expert load, skew, balance loss, dropped/delayed tokens, capacity.
- Source anchors: `llmsys-17-moe-load-balancing`
- Production note: Avoid implying perfectly uniform routing is always ideal; the point is avoiding pathological imbalance.

## Chapter 11

### `fig-11-quantization-map`

- Chapter: 11
- Purpose: Show float values mapped to low-bit buckets with scale and optional zero point.
- Core message: Quantization represents a continuous or high-precision range with discrete integer levels plus metadata.
- Visual form: Number line.
- Layout: float values above a number line, bucket boundaries and integer codes below, with scale and zero point callouts.
- Labels: scale, zero point, integer code, bucket, dequantize.
- Source anchors: `llmsys-19-absmax-zeropoint`
- Production note: Use symbolic values; do not imply one quantization scheme covers all deployments.

### `fig-11-quantization-error`

- Chapter: 11
- Purpose: Distinguish rounding error, clipping, and range mismatch.
- Core message: Quantization error can come from finite bucket resolution, values outside the representable range, or a range estimate that does not fit the data.
- Visual form: Three small plots.
- Layout: three panels labeled rounding, clipping, range mismatch; show input distribution and quantized representation.
- Labels: rounding, clipping, saturation, range, outlier.
- Source anchors: `llmsys-19-direct-quantization-errors`
- Production note: Keep this conceptual and avoid metric values unless sourced.

### `fig-11-llmint8-outlier-path`

- Chapter: 11
- Purpose: Show normal 8-bit path plus higher-precision outlier path.
- Core message: LLM.int8-style execution separates common values from outliers so the main path can use low precision while preserving sensitive components.
- Visual form: Dataflow.
- Layout: activation/weight input splits into normal low-precision matmul path and outlier higher-precision path, then combines outputs.
- Labels: normal path, outlier path, int8, higher precision, combine.
- Source anchors: `llmsys-19-llmint8-outliers`
- Production note: Tie the caption to the specific method; do not generalize outlier handling to all int8 systems.

### `fig-11-gptq-compensation`

- Chapter: 11
- Purpose: Show quantize column, compute error, and update remaining columns.
- Core message: GPTQ-style post-training quantization compensates future weights for quantization error instead of quantizing each element independently.
- Visual form: Matrix block diagram.
- Layout: weight matrix columns; one column marked quantized; error arrow updates remaining block using Hessian-aware information.
- Labels: weight column, quantized column, error, compensation, remaining columns.
- Source anchors: `llmsys-20-gptq-blockwise-compensation`
- Production note: Keep Hessian detail in prose unless the chapter includes the exact formula.

### `fig-11-full-ft-vs-lora-state`

- Chapter: 11
- Purpose: Compare full fine-tuning state with LoRA trainable state.
- Core message: Parameter-efficient fine-tuning reduces trainable state by freezing the base model and updating smaller adapter parameters.
- Visual form: Memory ledger.
- Layout: two columns: full fine-tuning with base weights, gradients, optimizer states; LoRA with frozen base plus trainable adapters and smaller optimizer state.
- Labels: frozen base, trainable parameter, gradient, optimizer state, adapter.
- Source anchors: `llmsys-23-full-finetuning-cost`, `llmsys-23-lora-training-state`
- Production note: Avoid exact memory savings without model, rank, precision, and optimizer assumptions.

### `fig-11-lora-update`

- Chapter: 11
- Purpose: Show `W' = W0 + A B` with low-rank matrices.
- Core message: LoRA adds a learned low-rank update to a frozen base weight matrix.
- Visual form: Matrix factor diagram.
- Layout: frozen matrix `W0` plus product of skinny matrices `A` and `B` equals adapted weight `W'`.
- Labels: frozen base, low-rank adapter, rank, trainable, adapted weight.
- Source anchors: `llmsys-23-lora-lowrank-update`
- Production note: Keep matrix orientation consistent with Chapter 11 notation.

### `fig-11-qlora-stack`

- Chapter: 11
- Purpose: Show quantized frozen base plus trainable adapters.
- Core message: QLoRA combines a quantized base model with trainable low-rank adapters so fine-tuning changes a smaller high-precision parameter set.
- Visual form: Layer stack.
- Layout: quantized frozen base layer, dequantization/compute boundary, LoRA adapter path, combined output.
- Labels: quantized base, frozen, dequantize, adapter, trainable, combined output.
- Source anchors: `llmsys-23-qlora-quantized-lora`
- Production note: Do not imply every QLoRA implementation has identical precision choices; preserve source-specific terminology in caption review.

## Cross-Chapter Notes

- Use one visual language for ownership: replicated state, sharded state, and temporary gathered state should look different.
- Communication figures should distinguish data movement from computation with separate colors or line styles.
- Memory ledgers should avoid absolute byte counts unless the caption includes model, precision, optimizer, batch/sequence, and activation assumptions.
- Compiler figures should make the abstraction boundary visible without suggesting the stack is universal across all frameworks.
- Quantization and PEFT figures must distinguish algorithm mechanism from benchmark outcome.

Owner: Book Architect  
Purpose: Part III figure planning  
Evidence grade: A for structural decisions derived from chapter briefs and source-ledger anchors; figure content must be checked against source cards before drawing  
Assumptions: Figures are technical book diagrams, and benchmark or memory-number claims will stay out of captions unless separately sourced with conditions  
Open questions: Whether Chapter 7 should keep Pallas figures as main-line diagrams or move one into an advanced sidebar  
Handoff: Technical reviewer for caption review, then illustrator for first-pass diagrams
