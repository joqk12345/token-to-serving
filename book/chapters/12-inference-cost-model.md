---
status: ready
chapter: 12
slug: 12-inference-cost-model
title: The Cost Model of LLM Inference
primary_sources:
  - llmsys-22-llm-serving-scheduler-radixattention-dfa87a4515092525676277a85bc4425d.pdf
  - llmsys-30-serving-c4a70ab21cde01fb60068a256c6e163a.pdf
papers:
  - downloads/llmsystem2026spring/source_pdfs/osdi22-yu.pdf
secondary_sources: []
reader_level: engineer or graduate student who has read Chapters 1-11
technical_depth: intermediate-to-advanced
---

# The Cost Model of LLM Inference

Training systems work on known batches. Serving systems do not.

An LLM server receives requests over time. Some prompts are short. Some are long. Some users ask for one sentence. Others ask for a page. Some requests share a prefix. Some arrive while the GPU is already decoding tokens for other users.

The serving problem is therefore not just:

```text
run model.forward()
```

It is:

```text
admit requests
batch compatible work
stream outputs
manage KV memory
route prefix reuse
keep GPU workers busy
control user-facing latency
```

The serving lecture states that efficient LLM servers must handle multi-user sessions, many simultaneous requests, varying generation lengths, common prompt prefixes, throughput, latency, and heterogeneous CPU/GPU devices. [CITE: llmsys-22-serving-goals]

This chapter builds the cost model that later chapters will refine. Chapter 13 will focus on KV cache, PagedAttention, and vLLM. Chapter 14 will focus on scheduling, caching, and disaggregation. Here the goal is to understand the terms that make inference expensive.

## Serving Is Online, Not a Training Step

The LLM application stack contains more than model execution. The serving framework lecture separates data preprocessing/embedding, prompt construction/retrieval, and prompt execution/inference. [CITE: llmsys-30-llm-app-stack]

This chapter focuses on prompt execution, but the boundary matters. A user request may arrive after retrieval, prompt construction, safety checks, or application orchestration. The inference server receives a prompt and has to produce output tokens under latency and throughput constraints.

The serving layer is often split into:

```text
API / request layer
tokenizer
scheduler
model worker
memory manager
detokenizer / streaming output
```

The course lecture uses serving frameworks to show this boundary: a serving layer may group client requests, while an inference engine performs batched model execution. [CITE: llmsys-30-serving-framework-boundary] Triton plus an inference engine is one example of this separation. [CITE: llmsys-30-triton-batching-engine]

The point is architectural, not a product recommendation:

```text
serving = request system + model execution system + memory system
```

The cost model has to include all three.

## Autoregressive Inference Has Two Phases

Autoregressive generation produces text one token at a time. ORCA describes Transformer-based generative inference as multiple model iterations, where each iteration generates a single output token. [CITE: yu-2022-orca-autoregressive-iterations]

![Inference begins with prompt prefill, which processes the input context, then moves into decode steps that append one generated token at a time while updating the KV cache.](../figures/artwork/ch12/fig-12-prefill-decode.svg)

In serving systems, it is useful to split a request into two phases:

```text
prefill:
  process the input prompt
  build the initial attention state
  produce the first output-token distribution

decode:
  repeatedly generate one new token
  reuse previously stored keys and values
  stop when the request reaches its stop condition
```

Prefill cost grows with prompt length. Decode cost grows with generated length, but each decode step depends on the previous step's output. That dependency makes decode sequential at the request level:

```text
token t must be generated before token t+1 can be generated
```

Across requests, however, decode iterations can be batched. The serving scheduler decides which requests participate in each iteration.

## Latency Is Not One Number

Users experience several different delays:

```text
queueing delay:
  time before the request starts useful model work

time to first token:
  time until the first generated token can be streamed

per-token decode time:
  time between streamed output tokens

total response time:
  time until the request finishes
```

Throughput is also multi-sided. A server may report requests per second, tokens per second, or useful output tokens per second. These can move in different directions depending on prompt length, output length, batching policy, and cache reuse.

This chapter avoids formal metric abbreviations unless a later revision adds dedicated metric source cards. The practical point is enough:

```text
prefill dominates first-token latency for long prompts;
decode scheduling dominates streaming cadence for long outputs;
queueing depends on admission and batching;
KV memory limits how many requests can stay active
```

## Why Request-Level Batching Fails

Batching is natural for GPU utilization. Training uses batches constantly. But serving batches are different because requests have different output lengths.

![Request-level batching can leave shorter generations waiting on longer ones, tying batch progress to the slowest unfinished request.](../figures/artwork/ch12/fig-12-request-level-batching-problem.svg)

The serving lecture describes the challenge: a request batch may lead to different generation lengths, and a naive implementation waits for the longest sequence. [CITE: llmsys-22-naive-batch-longest]

ORCA makes the same point at the paper level. Under request-level scheduling, finished requests can wait for the whole batch, and newly arrived requests can wait for the current batch to finish. [CITE: yu-2022-orca-request-level-limitation]

Imagine a batch with three requests:

```text
request A: generate 2 tokens
request B: generate 20 tokens
request C: generate 5 tokens
```

If the engine treats the batch as one unit, A and C may be stuck behind B. The GPU may also miss opportunities to admit new requests after A and C finish.

The bottleneck is not only compute. It is batch membership:

```text
which requests are active in this iteration?
when can finished requests leave?
when can new requests enter?
```

## Continuous Batching

Continuous batching changes the scheduling granularity. Instead of choosing a batch and waiting for every request in it to finish, the scheduler updates the active set between generation iterations.

![Continuous batching changes active batch membership between decode iterations, adding new requests and removing completed ones as the scheduler advances.](../figures/artwork/ch12/fig-12-continuous-batching.svg)

The serving lecture describes continuous batching as iteration-level scheduling: at each token-generation step, schedule a new request whenever one request finishes in a batch. [CITE: llmsys-22-continuous-batching]

ORCA calls this iteration-level scheduling: the serving system invokes the execution engine to run one model iteration on a batch, then can update request membership. [CITE: yu-2022-orca-iteration-level-scheduling]

The loop is:

```text
choose active requests
run one generation iteration
stream produced tokens
remove finished requests
admit waiting requests if memory and policy allow
repeat
```

Continuous batching improves the system's ability to keep the GPU occupied under variable output lengths. It does not remove every tradeoff. Larger batches may improve hardware utilization but increase contention for KV memory. Admitting prefill work can affect ongoing decode latency. Scheduler policy still determines fairness, queueing, and user-visible delay.

## Selective Batching

A Transformer block contains operations with different batching behavior. Linear layers, layer normalization, and elementwise operations can often benefit from batching across requests. Attention is harder when requests have different processed lengths and KV cache states.

![Selective batching separates operations that batch cleanly across requests from attention and cache operations that depend on request-specific state.](../figures/artwork/ch12/fig-12-selective-batching.svg)

The serving lecture describes selective batching as batching non-attention operations while attention is executed by an attention engine. [CITE: llmsys-22-selective-batching]

ORCA describes selective batching as applying batching only to selected operations. [CITE: yu-2022-orca-selective-batching]

The reason is shape and reuse:

```text
non-attention operations:
  often share compatible dense computation across requests

attention:
  depends on each request's current context length and KV state
```

This is the operation-level version of the serving problem. Not all work in a batch is equally batchable.

## The Scheduler Is Part of the Critical Path

The scheduler does more than maintain a queue. The serving lecture lists scheduler responsibilities: receive input requests, stream outputs, check stop conditions, reorder requests, prepare batches, and allocate memory for next and running batches. [CITE: llmsys-22-request-scheduler]

![An inference scheduler repeatedly admits requests, forms model-step batches, returns streamed tokens, updates memory state, and frees completed requests.](../figures/artwork/ch12/fig-12-scheduler-loop.svg)

A simplified loop:

```text
while server is running:
  receive new requests
  process input requests
  choose next batch
  run model worker
  process generated tokens
  check stop conditions
  update memory ownership
```

This work can affect GPU utilization. If the GPU finishes an iteration and waits for the CPU scheduler, CPU-side overhead is now in the critical path.

The serving lecture therefore discusses overlapping CPU scheduler work with GPU model-worker execution. [CITE: llmsys-22-scheduler-worker-overlap]

The lesson is:

```text
serving overhead is not outside the model;
it can become part of the model's effective latency
```

## KV Cache Is Serving Memory

During decode, the model needs attention keys and values from previous tokens. Recomputing them every step would be wasteful. Transformer serving systems therefore store keys and values for previous tokens in GPU memory as a KV cache. [CITE: llmsys-22-kv-cache-need]

![During autoregressive decoding, each generated token adds per-layer key and value state that future decode steps reuse.](../figures/artwork/ch12/fig-12-kv-cache-cost.svg)

This changes the memory model:

```text
training memory:
  parameters + gradients + optimizer state + activations

serving memory:
  parameters + KV cache + temporary buffers + scheduler/runtime state
```

KV cache grows with:

```text
number of active requests
prompt length
generated length so far
number of layers
attention head dimensions
precision
```

This chapter intentionally avoids KV cache byte formulas. Chapter 13 will study KV cache and PagedAttention in detail. For the inference cost model, the key point is:

```text
active tokens consume persistent serving memory
```

The scheduler cannot admit requests based only on compute availability. It also needs enough KV memory for active and newly admitted requests.

## Prefix Reuse and Cache-Aware Scheduling

Many serving workloads contain repeated prefixes. A chat system may reuse prior conversation state. An in-context learning application may send the same examples with many queries. A system prompt may appear across many requests.

![Prefix caching can reuse work for requests with shared prompt prefixes when the scheduler routes them to workers that already hold matching cached state.](../figures/artwork/ch12/fig-12-prefix-cache-routing.svg)

RadixAttention stores KV memory pointers in a radix tree keyed by prompt prefixes. [CITE: llmsys-22-radixattention-prefix-cache]

The mechanism is:

```text
shared prompt prefix
  -> shared path in prefix tree
  -> pointers to existing KV memory
  -> less repeated prefill work if cache is hit
```

Cache-aware scheduling can use matched prefix length or predicted prefix KV cache hit rates when ordering or routing requests. [CITE: llmsys-22-cache-aware-scheduling]

This introduces another cost-model term:

```text
effective prefill cost depends on prefix reuse
```

If a request hits a useful prefix cache, some prompt processing work may be reused. If it misses, the server pays the full prefill cost. If cache memory is full, eviction policy matters.

The detailed data structures belong in Chapter 13 and Chapter 14. Here the point is that a scheduler can be cache-aware, not only queue-aware.

## Tokenization, Detokenization, and Routing Are Also Work

Model kernels are the largest visible part of serving, but they are not the only work. The serving framework lecture notes systems such as LightLLM that perform tokenization, model inference, and detokenization asynchronously, and include token-wise KV cache memory management and routing. [CITE: llmsys-30-lightllm-async-token-attention]

This reinforces the boundary:

```text
request arrives as text
text becomes tokens
tokens enter scheduler/model worker
output tokens become streamed text
memory state is updated
```

When model execution is highly optimized, these surrounding steps can become visible. They may also affect batching because tokenization and detokenization happen at request boundaries, while model execution happens at prefill/decode iteration boundaries.

## A Conceptual Cost Model

A useful inference cost model does not start with one latency number. It starts with the pieces:

```text
request cost ≈ queueing
             + prefill(prompt length, prefix reuse)
             + decode(output length, batch policy)
             + KV memory pressure
             + scheduler/runtime overhead
```

This is not a numerical formula. It is a checklist.

For a long prompt and short output, prefill may dominate. For a short prompt and long output, decode scheduling and per-token latency may dominate. For many concurrent long-context requests, KV memory may dominate admission. For repeated system prompts or shared examples, prefix reuse can change effective prefill cost. For an overloaded server, queueing delay may dominate user experience.

The scheduler's job is to trade these costs against each other:

```text
admit more requests -> better occupancy, more KV memory pressure
larger batch -> better throughput, possible latency tradeoff
cache reuse -> less prefill work, more cache-management policy
overlap scheduler -> less CPU overhead on GPU critical path
```

## What to Remember

LLM inference is not a single forward pass. It is online, stateful, and iterative.

The core serving loop is:

```text
prefill prompt
store KV cache
decode one token
update request state
rebatch
repeat
```

The misconception to discard is:

```text
serving is just batching requests through the model
```

Batching is necessary, but insufficient. A serving system also manages variable output lengths, queueing, streaming, stop conditions, KV memory, cache reuse, CPU/GPU coordination, and user-facing latency. The model computes tokens; the server decides which tokens can be computed now.

Owner: Principal Author
Purpose: Chapter 12 ready draft after source extraction, brief, technical review, and red-team review
Evidence grade: A for course lecture claims and ORCA paper; no benchmark numbers used
Assumptions: Chapter 12 is a conceptual inference cost-model chapter; detailed PagedAttention/vLLM and disaggregated serving mechanisms remain Chapters 13-14
Open questions: Whether to add formal TTFT/TPOT terminology with dedicated source cards in a later revision
Handoff: Production can move to Chapter 13 source extraction
