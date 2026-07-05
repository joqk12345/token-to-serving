# Source Note: llmsys-16 Model Parallel Training

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-16-model-parallel-83b41612547620ee0e172caa1ee448ed.pdf`

## Scope

This source covers model parallel training motivation, pipeline parallelism, GPipe-style micro-batching, activation memory and gradient checkpointing, 1F1B scheduling, interleaved/chunked pipeline stages, PyTorch pipeline APIs, tensor parallelism for matrix multiplication, FFN, self-attention, embeddings, and combinations of tensor, pipeline, and data parallelism.

## Key Claims

- Model parallelism is motivated by model sizes that no longer fit in a single GPU's memory.
- Model-parallel training partitions forward, backward, and update computation across multiple workers.
- Pipeline parallelism splits the model by layers; naive layer-wise partitioning leaves most devices idle at any given time.
- GPipe improves pipeline utilization by dividing a mini-batch into micro-batches and pipelining them through model partitions.
- Pipeline parallelism has bubble overhead, communication at partition boundaries, and activation-memory pressure.
- Gradient checkpointing/rematerialization trades extra computation for lower activation memory.
- 1F1B schedules start backward as soon as possible to reduce the number of in-flight micro-batches and activation storage.
- Tensor parallelism splits large matrix computations across GPUs.
- Transformer FFN tensor parallelism can split the first projection over columns and the second projection over rows so the intermediate does not require all-reduce but the final result is a sum across partitions.
- Self-attention tensor parallelism can split weights over columns/heads without requiring all-reduce for the head-local computation.
- Embedding tensor parallelism has different communication behavior for input and output embeddings.
- Tensor parallelism is often used within a node, while pipeline parallelism can extend across servers; data parallelism can then scale training to more replicas.

## Chapter 9 Use

- Open from Chapter 8's boundary: data parallelism splits data but still replicates the model; model parallelism starts when model/activations/optimizer state do not fit.
- Explain pipeline parallelism first as a layer-wise split, then show why the naive schedule wastes devices.
- Use micro-batching and 1F1B as scheduling answers to pipeline bubbles and activation memory.
- Explain tensor parallelism as splitting individual matrix operations, not layers.
- Use FFN and attention as concrete Transformer examples.
- Close by positioning tensor, pipeline, and data parallelism as composable axes.

## Do Not Use As

- A benchmark source for GPipe, PipeDream, or Megatron speedups.
- A complete current PyTorch pipeline API reference.
- A definitive rule that tensor parallelism must always stay within a node.

## Candidate Source Cards

- `llmsys-16-model-parallel-motivation`
- `llmsys-16-layerwise-pipeline`
- `llmsys-16-naive-pipeline-idle`
- `llmsys-16-gpipe-microbatching`
- `llmsys-16-pipeline-costs`
- `llmsys-16-gradient-checkpointing`
- `llmsys-16-one-f-one-b`
- `llmsys-16-pipeline-chunking`
- `llmsys-16-pytorch-pipelining`
- `llmsys-16-tensor-parallel-matmul`
- `llmsys-16-tensor-parallel-ffn`
- `llmsys-16-tensor-parallel-attention`
- `llmsys-16-tensor-parallel-embeddings`
- `llmsys-16-parallelism-composition`

Owner: Technical Researcher  
Purpose: Chapter 9 source extraction  
Evidence grade: A for course framing; GPipe/PipeDream/Megatron/PyTorch official or paper sources needed for publication-level algorithm/API details  
Assumptions: Chapter 9 focuses on pipeline and tensor parallelism; ZeRO/MoE memory partitioning belongs to Chapter 10  
Open questions: Add GPipe, PipeDream, Megatron-LM, and PyTorch pipeline docs cards before draft if named systems become central  
Handoff: Book Architect for Chapter 9 brief
