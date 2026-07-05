# Part III Figure Caption Review

Date: 2026-07-05

Scope: `book/figures/part-iii-figure-specs.md`

## Verdict

Part III figure specs are ready for first-pass diagram production. The captions below are mechanism-focused and avoid unsourced claims about speedup, utilization, memory savings, or scaling efficiency.

Do not add numeric memory, communication, throughput, or speedup values to captions unless model, hardware, precision, topology, batch/sequence, and source context are attached.

## Chapter 7 Captions

### `fig-07-python-to-device-program`

Caption: A framework turns Python model code into device work by tracing or capturing computation, lowering it through intermediate representations, optimizing the program, and dispatching an executable to the accelerator.

Source anchors: `llmsys-12-xla-compilation-pipeline`

Risk: Name JAX/XLA if using JAX-specific IR labels; do not imply every framework uses the same compilation path.

### `fig-07-forward-to-backward-graph`

Caption: Reverse-mode autodiff builds a backward computation from the forward graph, using saved forward values where gradient rules require them.

Source anchors: `llmsys-05-automatic-differentiation`

Risk: Keep the example small. Do not make Transformer-scale activation memory claims in the caption.

### `fig-07-ir-stack`

Caption: Intermediate representations such as JAXpr, StableHLO, HLO, backend IR, and executable code separate user-level computation from hardware-specific execution.

Source anchors: `llmsys-12-jax-tracing-jaxpr`, `llmsys-12-stablehlo-portability`

Risk: Mark the stack as JAX/XLA-oriented; other compiler stacks may use different IR boundaries.

### `fig-07-xla-memory-planning`

Caption: Compiler optimization can change tensor layout, fusion, and buffer lifetimes, which changes where intermediate values are stored and when memory traffic occurs.

Source anchors: `llmsys-12-xla-optimization-passes`

Risk: Avoid saying fusion or memory planning always reduces memory traffic; shape and backend choices matter.

### `fig-07-tpu-systolic-compile-target`

Caption: For TPU execution, compiler tiling and layout choices shape matrix work for a systolic matrix unit rather than for arbitrary scalar execution.

Source anchors: `llmsys-12-tpu-systolic-mxu`

Risk: Keep TPU/MXU terminology explicit. Do not generalize systolic details to all accelerators.

### `fig-07-pallas-blockspec-grid`

Caption: Pallas-style block programs map grid coordinates to slices of HBM tensors, exposing block-level work through BlockSpec and VMEM references.

Source anchors: `llmsys-13-pallas-blockspec`

Risk: Avoid overloading the figure with API syntax; show the data mapping first.

### `fig-07-pallas-pipeline`

Caption: A tiled kernel can overlap movement of later tiles with computation on current tiles by staging data through VMEM.

Source anchors: `llmsys-13-pallas-pipelining`

Risk: Do not include speedup or latency claims without a benchmark setup.

## Chapter 8 Captions

### `fig-08-data-parallel-step`

Caption: Data parallel training runs model replicas on different data shards, computes local gradients, aggregates them with all-reduce, and applies the synchronized update on each worker.

Source anchors: `llmsys-14-data-parallel-allreduce`

Risk: Do not imply parameters are sharded; this figure is about replicated data parallelism.

### `fig-08-collective-semantics`

Caption: Broadcast, reduce, all-reduce, reduce-scatter, and all-gather differ in which workers own full tensors or shards before and after communication.

Source anchors: `llmsys-14-collective-primitives`, `llmsys-14-allreduce-semantics`

Risk: Keep communication volume symbolic unless assumptions are stated.

### `fig-08-ring-allreduce`

Caption: Ring all-reduce circulates tensor chunks through scatter-reduce and all-gather phases so each worker receives the reduced result.

Source anchors: `llmsys-14-ring-allreduce-phases`

Risk: Do not imply the logical ring matches physical network topology.

### `fig-08-naive-vs-overlap-ddp`

Caption: Naive DDP waits until backward computation finishes before reducing gradients, while bucketed DDP can start asynchronous all-reduce as buckets become ready.

Source anchors: `llmsys-15-ddp-naive-allreduce`, `llmsys-15-ddp-overlap`

Risk: Avoid saying overlap hides all communication; bucket order, tensor sizes, and network conditions affect the result.

### `fig-08-ddp-reducer-hooks`

Caption: PyTorch DDP uses autograd hooks and reducer buckets to trigger asynchronous all-reduce when a bucket's gradients are ready.

Source anchors: `llmsys-15-ddp-autograd-hooks`

Risk: Keep PyTorch-specific wording explicit; do not generalize hook implementation details to all DDP systems.

## Chapter 9 Captions

### `fig-09-data-vs-model-parallel`

Caption: Data parallelism replicates the model across workers, while model parallelism partitions model computation or state across workers.

Source anchors: `llmsys-16-model-parallel-motivation`

Risk: Use this as a conceptual contrast; real systems may combine both.

### `fig-09-naive-pipeline-bubble`

Caption: A layer-wise pipeline can leave devices idle while stages wait for inputs or gradients, creating pipeline bubbles.

Source anchors: `llmsys-16-naive-pipeline-idle`

Risk: Do not add bubble-size formulas unless stage count and micro-batch assumptions are stated.

### `fig-09-gpipe-microbatches`

Caption: GPipe-style micro-batching lets different micro-batches occupy different pipeline stages, reducing idle gaps after warmup.

Source anchors: `llmsys-16-gpipe-microbatching`

Risk: Do not imply micro-batching removes all idle time or activation-memory cost.

### `fig-09-1f1b-vs-gpipe`

Caption: One-forward-one-backward scheduling alternates forward and backward work after warmup, changing utilization and activation lifetime relative to all-forward-then-backward scheduling.

Source anchors: `llmsys-16-one-f-one-b`

Risk: Keep memory implications qualitative unless assumptions are included.

### `fig-09-tensor-parallel-ffn`

Caption: Tensor parallelism splits FFN matrix operations across devices and uses collective communication to assemble the logical layer result.

Source anchors: `llmsys-16-tensor-parallel-ffn`

Risk: Keep matrix shapes symbolic; exact splits depend on implementation.

### `fig-09-tensor-parallel-attention-heads`

Caption: Attention heads can be partitioned across devices so each worker computes a subset of heads before outputs are combined.

Source anchors: `llmsys-16-tensor-parallel-attention`

Risk: Do not imply head partitioning is sufficient for every attention implementation.

### `fig-09-3d-parallelism`

Caption: Large training jobs can compose data, pipeline, and tensor parallel groups, with each axis introducing different communication or scheduling constraints.

Source anchors: `llmsys-16-parallelism-composition`

Risk: Avoid showing a concrete cluster size unless the figure states it as illustrative.

## Chapter 10 Captions

### `fig-10-ddp-memory-ledger`

Caption: A DDP worker stores model parameters, gradients, optimizer state, activations, and temporary buffers, so replicated training memory includes more than weights.

Source anchors: `llmsys-18-ddp-memory-accounting`

Risk: Do not include byte totals without model, optimizer, precision, batch, and activation assumptions.

### `fig-10-zero-stages`

Caption: ZeRO partitions optimizer state, gradients, and parameters in progressively deeper stages instead of replicating all training state on every data-parallel worker.

Source anchors: `llmsys-18-zero-key-idea`

Risk: Treat stage labels as algorithm structure, not guaranteed performance ranking.

### `fig-10-zero3-parameter-gather`

Caption: ZeRO-3 keeps parameters sharded between uses and gathers the needed shards around layer computation during forward and backward execution.

Source anchors: `llmsys-18-zero-stage3-parameters`

Risk: Do not imply gathering is free; communication and scheduling costs remain.

### `fig-10-moe-router-experts`

Caption: A mixture-of-experts layer routes token representations from a shared input stream to selected expert FFNs and combines the expert outputs back into sequence order.

Source anchors: `llmsys-17-moe-ffn-router`

Risk: Avoid quality or sparsity-efficiency claims; this is the routing mechanism.

### `fig-10-expert-parallel-alltoall`

Caption: Expert parallelism dispatches tokens to devices that own selected experts and returns processed token outputs through all-to-all communication.

Source anchors: `llmsys-17-expert-parallelism`

Risk: Keep both dispatch and return paths visible; communication is part of the mechanism.

### `fig-10-load-balance-skew`

Caption: Router imbalance can overload some experts while leaving others underused, so MoE systems track and regularize expert load.

Source anchors: `llmsys-17-moe-load-balancing`

Risk: Do not imply perfectly uniform routing is always the right target.

## Chapter 11 Captions

### `fig-11-quantization-map`

Caption: Quantization maps high-precision values into discrete integer levels using scale metadata and, for some schemes, a zero point.

Source anchors: `llmsys-19-absmax-zeropoint`

Risk: Do not imply all quantization schemes use both scale and zero point.

### `fig-11-quantization-error`

Caption: Quantization error can arise from rounding within buckets, clipping outside the chosen range, or a range estimate that mismatches the data distribution.

Source anchors: `llmsys-19-direct-quantization-errors`

Risk: Keep error plots qualitative unless metrics and setup are supplied.

### `fig-11-llmint8-outlier-path`

Caption: LLM.int8-style execution routes typical values through an int8 path while handling outlier components through a higher-precision path before combining results.

Source anchors: `llmsys-19-llmint8-outliers`

Risk: Tie the caption to LLM.int8-style execution; outlier handling is not universal across all int8 methods.

### `fig-11-gptq-compensation`

Caption: GPTQ-style quantization accounts for quantization error by updating remaining weights after a column or block is quantized.

Source anchors: `llmsys-20-gptq-blockwise-compensation`

Risk: Keep Hessian details out of the caption unless the exact formula is rendered and reviewed.

### `fig-11-full-ft-vs-lora-state`

Caption: Full fine-tuning trains the base model state, while LoRA freezes the base weights and trains smaller adapter matrices with their own gradient and optimizer state.

Source anchors: `llmsys-23-full-finetuning-cost`, `llmsys-23-lora-training-state`

Risk: Avoid exact memory-savings claims without rank, precision, optimizer, and target-module assumptions.

### `fig-11-lora-update`

Caption: LoRA represents an adapted weight as a frozen base matrix plus a trainable low-rank update.

Source anchors: `llmsys-23-lora-lowrank-update`

Risk: Keep matrix orientation consistent with Chapter 11 notation.

### `fig-11-qlora-stack`

Caption: QLoRA combines a quantized frozen base model with trainable low-rank adapters, separating base-model storage from adapter training.

Source anchors: `llmsys-23-qlora-quantized-lora`

Risk: Do not imply every QLoRA implementation uses the same precision choices or memory manager.

## Remaining Review Notes

- Captions are cleared for draft diagrams.
- Communication captions should not include volume formulas until assumptions are attached.
- Memory-ledger captions should not include totals until model, precision, optimizer, and activation assumptions are attached.
- If Chapter 7 rendered figures include API syntax, verify syntax against the source cards and current chapter text.
- If Chapter 11 rendered figures include formulas, rerun notation review on the artwork text.

Owner: Technical Reviewer  
Purpose: Part III figure-caption review  
Evidence grade: A for review process; captions inherit their source anchors  
Assumptions: Part III figures are mechanism diagrams, not benchmark artifacts  
Open questions: Whether one Chapter 7 Pallas figure should be marked as an advanced sidebar during layout  
Handoff: Illustrator / diagram producer
