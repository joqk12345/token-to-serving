---
status: ready
chapter: 11
slug: 11-quantization-and-peft
title: Quantization and Parameter-Efficient Adaptation
primary_sources:
  - llmsys-19-quantization-da7a2abad092c802b03672ce1cc7bee9.pdf
  - llmsys-20-quantization2-ba573d7e5d82e68027bbd3a92c3cd819.pdf
  - llmsys-23-peft-791e97c5520e2ea7491153b45a923ac7.pdf
papers:
  - https://arxiv.org/abs/2208.07339
  - https://arxiv.org/abs/2210.17323
  - https://arxiv.org/abs/2106.09685
  - https://arxiv.org/abs/2305.14314
secondary_sources: []
reader_level: engineer or graduate student who has read Chapters 1-10
technical_depth: intermediate
---

# Quantization and Parameter-Efficient Adaptation

Chapter 10 treated memory as a training-systems ledger. ZeRO changed where optimizer states, gradients, and parameters live. MoE changed which expert parameters each token activates.

This chapter keeps the same accounting habit, but changes the problem:

```text
The base model already exists.
How do we store it, run it, and adapt it without carrying unnecessary cost?
```

Two families of techniques matter here.

```text
quantization:
  change how tensors are represented

parameter-efficient fine-tuning:
  change which parameters are trainable
```

Quantization reduces precision. PEFT reduces trainable state. Both can reduce memory pressure. Neither is free. Quantization can introduce numerical error and require special kernels. PEFT can constrain adaptation capacity and add adapter-management complexity.

The systems question is:

```text
which memory category was reduced,
and what new numerical or runtime condition appeared?
```

## Compression and Adaptation Reopen the Memory Ledger

A trained LLM is expensive in several modes.

For inference, the base weights must be loaded and moved through memory. For adaptation, the system may need gradients, optimizer states, activations, and updated weights. Full fine-tuning brings the training ledger back: weights, gradients, optimizer states, activations, and temporary buffers.

The PEFT lecture states that full-parameter fine-tuning updates all model parameters and requires large GPU memory for weights, gradients, optimizer states, and activations. [CITE: llmsys-23-full-finetuning-cost]

Quantization and PEFT reduce different terms:

```text
quantization:
  fewer bits per stored value

PEFT:
  fewer trainable values with gradient and optimizer state
```

That distinction is important. A quantized model can still be hard to fine-tune if the training method dequantizes too much state or maintains full optimizer state. A LoRA-adapted model can still be expensive to serve if the base weights are large and kernels are inefficient.

## Quantization Changes Representation

The quantization lecture defines model quantization as using low-bit precision to store parameters and layer outputs. It can reduce memory and may improve calculation throughput, but can also reduce accuracy. [CITE: llmsys-19-quantization-purpose]

![Quantization maps high-precision values into discrete integer levels using scale metadata and, for some schemes, a zero point.](../figures/artwork/ch11/fig-11-quantization-map.svg)

The basic idea is to map a real-valued tensor into a smaller set of representable values:

```text
float tensor -> integer code tensor + scale metadata
```

At inference or during a kernel, the system may use those integer codes directly, dequantize them, or use a mixed computation path. The details depend on the bitwidth, hardware, kernel implementation, and operation.

The safe claim is:

```text
quantization reduces representation cost;
performance depends on whether the runtime can exploit that representation
```

Lower precision does not automatically mean faster inference. If a kernel has to dequantize inefficiently, if the hardware lacks the right low-bit path, or if memory movement is not the bottleneck, the expected speedup may not appear.

## Scale, Zero Point, and Error

A quantizer maps a real value to an integer code. The simplest version uses a scale:

![Quantization error can arise from rounding within buckets, clipping outside the chosen range, or a range estimate that mismatches the data distribution.](../figures/artwork/ch11/fig-11-quantization-error.svg)

```text
q = round(x / s)
x_hat = s * q
```

where `x` is the original value, `q` is the integer code, `s` is the scale, and `x_hat` is the reconstructed approximation.

Absmax quantization chooses a scale from the largest absolute value in the tensor. The lecture summarizes it as linearly scaling according to the maximum absolute value. [CITE: llmsys-19-absmax-zeropoint]

Zero-point quantization adds an offset:

```text
q = round(x / s + z)
x_hat = s * (q - z)
```

where `z` is the zero point. This is useful when the representable integer range should be aligned with a real-valued range whose midpoint is not zero.

The mapping creates error. The lecture names several failure modes: loss of precision, range mismatch that clips values, and rounding error. [CITE: llmsys-19-direct-quantization-errors]

Those errors have different shapes:

```text
rounding:
  value lands on nearest representable bucket

clipping:
  value lies outside representable range and is cut off

range mismatch:
  scale wastes buckets on rarely used values or underserves important regions
```

Quantization is therefore not only a storage decision. It is an approximation decision.

## Post-Training Quantization and Calibration

Quantization can be built into training, or applied after training. The lecture distinguishes training-time and post-training approaches. [CITE: llmsys-19-quantization-approaches]

Post-training quantization is attractive because it avoids retraining the whole model. But it must preserve behavior using limited calibration data and local adjustments.

For a linear layer:

```text
Y = W X
```

a layer-wise quantization objective tries to choose quantized weights `W_hat` so the output stays close:

```text
minimize || W X - W_hat X ||^2
```

The lecture presents layer-wise quantization as minimizing the linear layer's output difference. [CITE: llmsys-19-layerwise-objective]

This objective shows why calibration data matters. The quantizer is not matching weights in the abstract. It is trying to preserve the layer's behavior on representative inputs `X`.

Layer-wise calibration also has a limitation: local output matching does not automatically remove accumulated error across the whole network. A small mismatch at one layer can shift the input distribution seen by later layers.

## ZeroQuant and LLM.int8 Show Two Scaling Tactics

ZeroQuant and LLM.int8 are useful because they show two different responses to the same problem: naive low-bit conversion is too brittle for large Transformer models.

![LLM.int8-style execution routes typical values through an int8 path while handling outlier components through a higher-precision path before combining results.](../figures/artwork/ch11/fig-11-llmint8-outlier-path.svg)

ZeroQuant uses layer-by-layer knowledge distillation. The lecture describes the full-precision model as the teacher and the quantized model as the student. [CITE: llmsys-19-zeroquant]

That is a calibration strategy:

```text
teacher layer output:
  what the full-precision model would produce

student layer output:
  what the quantized layer produces

training signal:
  reduce the mismatch
```

LLM.int8 addresses a different issue: outlier features. The lecture states that LLM.int8 keeps outliers in higher precision while quantizing the rest to 8-bit. [CITE: llmsys-19-llmint8-outliers]

The LLM.int8 paper frames this as vector-wise quantization plus a mixed-precision decomposition for outlier dimensions. [CITE: dettmers-2022-llmint8]

The systems lesson is:

```text
the rare values may determine the numerical path
```

If most values quantize well but a small set of activation dimensions has large magnitude, a uniform low-bit path can damage accuracy. A mixed path keeps the common case compact while preserving a higher-precision path for outliers.

## GPTQ: Quantize, Measure Error, Compensate

GPTQ is a post-training weight quantization method for generative Transformers. The lecture presents its goal as quantizing very large models while maintaining accuracy through layer-wise weight-matrix quantization. [CITE: llmsys-20-gptq-goal]

![GPTQ-style quantization accounts for quantization error by updating remaining weights after a column or block is quantized.](../figures/artwork/ch11/fig-11-gptq-compensation.svg)

The GPTQ paper describes it as one-shot weight quantization using approximate second-order information. [CITE: frantar-2022-gptq]

The mechanism can be understood without a full derivation:

```text
1. take a layer's weight matrix W
2. use calibration inputs X for that layer
3. quantize part of W
4. measure the induced error
5. update still-unquantized weights to compensate
6. continue until the layer is quantized
```

The lecture emphasizes that GPTQ quantizes one column block at a time and updates not-yet-quantized weights to compensate for the error caused by quantizing earlier weights. [CITE: llmsys-20-gptq-blockwise-compensation]

This is the central idea:

```text
rounding error is not only accepted;
it is pushed into future compensation
```

GPTQ uses information derived from the layer inputs. The lecture describes precomputing information from the inverse Hessian using Cholesky decomposition. [CITE: llmsys-20-gptq-hessian-cholesky]

For this chapter, the important point is not the exact matrix formula. It is why second-order information appears at all. Some weights matter more than others for the layer output on the calibration distribution. Curvature-like information helps estimate how quantization error in one weight can be compensated by adjusting remaining weights.

## GPTQ Is Also a Systems Method

The algorithmic idea is only useful if it can run at model scale. The lecture describes GPTQ lazy updates: rather than applying every compensation update immediately in a way that underuses the GPU, GPTQ batches or delays updates to improve practical efficiency. [CITE: llmsys-20-gptq-lazy-updates]

That makes GPTQ a systems method, not just an optimization formula:

```text
calibration data determines local behavior
block order determines update dependencies
precomputation determines available curvature information
lazy updates determine practical GPU efficiency
custom kernels determine inference benefit
```

This chapter deliberately avoids GPTQ benchmark numbers. The value of a quantized model depends on model family, bitwidth, calibration set, hardware, kernel implementation, batch size, sequence length, and whether inference is memory-bound or compute-bound.

## Full Fine-Tuning Carries Full Training State

Quantization changes representation. PEFT changes what is trained.

![Full fine-tuning trains the base model state, while LoRA freezes the base weights and trains smaller adapter matrices with their own gradient and optimizer state.](../figures/artwork/ch11/fig-11-full-ft-vs-lora-state.svg)

Full fine-tuning updates all model parameters. That means the system needs gradient and optimizer state for all trainable parameters. Chapter 10's ledger returns:

```text
base weights
weight gradients
optimizer states
activations
temporary buffers
```

The PEFT lecture uses full fine-tuning as the motivation for parameter-efficient methods: updating all parameters requires large GPU memory. [CITE: llmsys-23-full-finetuning-cost]

The problem is not that full fine-tuning is conceptually hard. It is that full fine-tuning reintroduces the memory footprint of training for every parameter, even if the adaptation task may only need a much smaller change.

## PEFT Changes What Is Trainable

Parameter-efficient fine-tuning updates a small subset or low-rank set of parameters rather than the entire model. [CITE: llmsys-23-peft-definition]

The lecture groups PEFT methods into selective, reparameterization, and additive methods. [CITE: llmsys-23-peft-categories]

At a systems level, the categories differ in what state they add or expose:

```text
selective:
  train chosen existing parameters

reparameterization:
  train a smaller parameterization of an update

additive:
  add trainable modules or prompts around the frozen model
```

This chapter focuses on LoRA because its system tradeoff is especially direct: freeze a large base matrix and train a low-rank update.

## LoRA: Low-Rank Updates

LoRA freezes pretrained weights and trains a low-rank update. The lecture writes the adapted weight as:

![LoRA represents an adapted weight as a frozen base matrix plus a trainable low-rank update.](../figures/artwork/ch11/fig-11-lora-update.svg)

```text
W' = W0 + A B
```

where `W0` is the frozen pretrained weight and `A B` is a low-rank update with rank `r` much smaller than the full dimension. [CITE: llmsys-23-lora-lowrank-update]

The LoRA paper states the same core mechanism: freeze pretrained model weights and inject trainable rank-decomposition matrices into Transformer layers. [CITE: hu-2021-lora]

The forward path becomes:

```text
y = W0 x + A B x
```

The base matrix still participates in inference and training forward passes. But the trainable parameters are `A` and `B`, not all of `W0`.

That distinction matters for memory:

```text
frozen base weight:
  stored, used in forward/backward computation,
  but no optimizer state for updating it

LoRA matrices:
  stored, trained,
  carry gradients and optimizer state
```

The lecture states that LoRA training stores original parameters, adapter weights, adapter gradients, adapter states, and activations, and does not need original parameter states. [CITE: llmsys-23-lora-training-state]

LoRA does not remove activation memory. It does not make the base model disappear. Its main training-memory effect is reducing trainable parameter state.

## Adapter Placement Is a Design Choice

LoRA is often described by the equation `W' = W0 + A B`, but the system still has to decide where adapters go.

The lecture discusses applying LoRA/CIAT to Transformer weights such as attention projections, and notes that placement can vary. [CITE: llmsys-23-lora-lowrank-update]

Placement affects:

- trainable parameter count;
- memory for adapter gradients and optimizer state;
- compute overhead in forward/backward;
- task quality;
- adapter storage for multi-task deployment;
- whether adapters can be merged into base weights for a given deployment mode.

This is another example of the book's recurring rule:

```text
the mathematical trick becomes a systems interface
```

The low-rank update is the math. Target-module selection, adapter storage, merging, switching, and kernel behavior are the systems interface.

## QLoRA Combines Quantization and Low-Rank Training

QLoRA connects the two halves of this chapter. The lecture describes it as quantization plus low-rank training. [CITE: llmsys-23-qlora-quantized-lora]

![QLoRA combines a quantized frozen base model with trainable low-rank adapters, separating base-model storage from adapter training.](../figures/artwork/ch11/fig-11-qlora-stack.svg)

The QLoRA paper states that gradients are backpropagated through a frozen 4-bit quantized pretrained language model into LoRA adapters, and introduces NF4, double quantization, and paged optimizers. [CITE: dettmers-2023-qlora]

The basic structure is:

```text
base model:
  frozen
  stored in low precision

adapter:
  trainable
  low-rank
  carries gradients and optimizer state
```

QLoRA reduces memory from two directions:

```text
quantization reduces base model representation cost
LoRA reduces trainable-state cost
```

This does not mean QLoRA is automatically the right choice for every task or deployment. Quality, training stability, kernel support, optimizer behavior, and adapter rank still matter. The safe claim is structural: QLoRA combines a quantized frozen base with trainable low-rank adapters.

## Deployment Consequences

Compression and adaptation affect serving even when this chapter is not a serving chapter.

Quantized models need kernels that can exploit the chosen representation. A weight-only quantized model may reduce memory bandwidth pressure, but dequantization and mixed-precision paths still cost work. Low-bit storage is not the same as low-bit end-to-end execution.

LoRA-style adapters create a different deployment question:

```text
one base model
many task adapters
```

This can be useful because adapters are much smaller than full model copies. But a serving system still has to load, select, batch, merge, or switch adapters correctly. Those details interact with latency and memory.

Part V will handle serving architecture. The point here is narrower:

```text
compression and adaptation change the runtime contract
```

The model artifact is no longer just "a set of weights." It may be integer codes plus scales, mixed-precision outlier paths, frozen base weights, and adapter matrices.

## What to Remember

Quantization asks:

```text
How many bits do we need to represent this tensor well enough,
and can the runtime exploit that representation?
```

PEFT asks:

```text
Which parameters actually need gradients and optimizer state
for this adaptation?
```

GPTQ shows that quantization can be more than rounding: it can use calibration data and compensation to reduce output error. LoRA shows that fine-tuning can be more than updating every parameter: it can train a low-rank update while freezing the base model. QLoRA combines those ideas by storing the base model in low precision and training low-rank adapters.

The misconception to discard is:

```text
compression is just making the model smaller
```

Compression and adaptation are systems decisions. They alter memory layout, numerical error, kernel requirements, trainable state, optimizer state, and deployment mechanics. A smaller artifact is useful only when the surrounding runtime can preserve enough quality and convert the representation change into real memory or throughput benefits.

Owner: Principal Author
Purpose: Chapter 11 ready draft after source extraction, brief, technical review, and red-team review
Evidence grade: A for course lecture claims and primary papers; no benchmark numbers used
Assumptions: Chapter 11 focuses on mechanisms and systems tradeoffs; detailed benchmark comparisons, NF4 derivation, and serving architecture are out of scope
Open questions: Whether to add a deeper QLoRA source pass for NF4, double quantization, and paged optimizers
Handoff: Production can move to Chapter 12 source extraction
