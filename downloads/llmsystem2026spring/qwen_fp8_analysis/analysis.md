# Qwen3.5-122B-A10B FP8 Kernel 分层与量化收益说明

## 1. 分析对象与假设

本文档用于解释 `Qwen3.5-122B-A10B` 在推理阶段的 kernel 耗时排序、FP8 量化收益空间，以及为什么某些层适合 FP8，而某些层应保留 FP16/BF16。

当前公开资料未能确认 `Qwen3.5-122B-A10B` 的完整结构参数，例如层数、hidden size、KV head 数、expert 数量和 top-k 路由配置。因此本文采用如下工程假设：

- 模型类型：MoE decoder-only Transformer
- 总参数规模：约 122B
- 每 token 激活参数：约 10B，即 A10B
- Attention 结构：Hybrid Attention，即不同层或不同 token 路径可能混合 full attention、sliding window attention、sparse/global attention 等机制
- 推理硬件目标：H100/Hopper 或同级支持 FP8 Tensor Core 的 GPU
- 主要优化目标：prefill/decode 吞吐、长上下文 decode、显存占用与 KV cache 带宽

## 2. 参考资料

- `llmsys-21-FlashAttention_tridao2026.4-50476379a6127697ae7fbf974ad28348.pdf`
- `/Users/mac/Downloads/auto-round-vllm-v1.html`
- `/Users/mac/Documents/qiaokai-doc-2025-07/shared/documents/FP8量化优化V1.0推理性能分析报告.pdf`
- 可视化文件：`index.html`

## 3. FP8 与 FP16/BF16 分层判定依据

图中的 `FP8` 和 `FP16/BF16` 标注不是按层名随意划分，而是按以下两个核心问题判断：

1. 该层量化后是否数值安全？
2. 该层量化后是否有实际性能收益？

更具体地说，判断依据包括：

- 是否是大规模矩阵乘法
- 是否主要受显存带宽或权重读取限制
- 是否有硬件原生 FP8 kernel 支持
- 是否对数值稳定性高度敏感
- 是否会引入额外 quant/dequant 开销
- 误差是否会跨层累积并影响模型质量

## 4. 适合 FP8 的层

### 4.1 Linear / GEMM 层

适合 FP8 的典型层包括：

- QKV Projection GEMM
- O Projection GEMM
- MoE expert up GEMM
- MoE expert gate GEMM
- MoE expert down GEMM

这些层的主要计算形式是矩阵乘法。对于 H100/Hopper 这类 GPU，FP8 Tensor Core 可以提供更高的矩阵乘吞吐，同时 FP8 权重读取量约为 BF16 的一半。

因此，Linear/GEMM 层通常同时满足两个条件：

- 计算量大，能吃到 Tensor Core 加速
- 权重体积大，能减少 HBM 带宽和显存占用

在 `Qwen3.5-122B-A10B` 这种 MoE 模型中，MoE expert GEMM 是最优先考虑 FP8 的部分，因为每 token 激活约 10B 参数，大量开销集中在 expert 的 grouped GEMM 上。

### 4.2 KV Cache

KV cache 适合 FP8 的原因与权重量化不同。它的核心收益来自 decode 阶段的读带宽和显存容量。

decode 时每生成一个 token，都需要读取历史 K/V。上下文越长，KV cache 读写越重。将 KV cache 从 FP16/BF16 改为 FP8，理论上可以：

- 将 KV cache 容量约减半
- 将历史 K/V 读取带宽约减半
- 支持更长上下文或更大 batch
- 降低长上下文 decode 延迟

FP8 报告中也显示，加入 FP8 KV cache 优化后，端到端收益从 Fused QKV 阶段进一步提升。

### 4.3 Attention 内部 Matmul

Attention 中的 `QK^T` 和 `P·V` 本质上也是矩阵乘法，因此可以进入 FP8 路径。但 attention 不是全流程都适合 FP8。

通常适合 FP8 的是：

- Q/K/V 参与的 matmul
- V 相关的乘法路径
- 部分 FlashAttention kernel 内的低精度计算路径

但 softmax、mask、scale、累加和稳定化计算通常仍需要 FP16/BF16 或更高精度。

FlashAttention 课件的关键结论是：attention 的瓶颈往往不是单纯 FLOPs，而是 HBM 读写。FlashAttention 通过 block 计算减少 HBM IO；FP8 则进一步降低数据搬运压力。

### 4.4 Hybrid Attention 路径

`Qwen3.5-122B-A10B` 的 attention 不是单一 full attention，而是 hybrid attention。图中已将原来的单一路径拆成三类：

- Full Attention  
  读取完整历史 KV，适合全局信息汇聚，但长上下文 decode 的 KV cache 带宽压力最大。

- Sliding Window Attention  
  只读取固定窗口内的局部 KV，能显著降低长上下文下的 attention 读带宽和计算量。

- Sparse / Global Attention  
  只对部分全局 token、特殊 token 或稀疏模式做额外关注，性能取决于具体 mask 和 kernel 实现。

Hybrid Attention 对 FP8 判定的影响是：

- QKV/O projection 仍然适合 FP8，因为它们仍是大 GEMM。
- KV cache 仍然适合 FP8，因为 full 和 sliding window 都要读 K/V，只是读取范围不同。
- Full attention 层的 FP8 KV cache 收益最大。
- Sliding window 层会降低 attention 占比，因此 MoE expert GEMM 在 decode 中可能更靠前。
- Sparse/global 路径需要看 kernel 是否支持稀疏 mask、block mask 或 paged KV；如果没有高效 kernel，控制流和 gather/scatter 可能抵消部分 FP8 收益。

## 5. 不适合直接 FP8 的层

### 5.1 RMSNorm / LayerNorm

RMSNorm 和 LayerNorm 涉及归一化、平方、求和、缩放等操作，对数值精度比较敏感。

如果这类层直接使用 FP8，可能导致：

- 归一化统计不稳定
- 激活尺度漂移
- 后续层误差放大
- 多层累积后模型质量下降

这些层计算量相对 GEMM 较小，FP8 的性能收益有限，因此通常保留 FP16/BF16。

### 5.2 RoPE / LogN Scaling

RoPE 和 LogN Scaling 属于位置编码与缩放路径。它们不是大矩阵乘法，主要开销不在 Tensor Core。

保留 FP16/BF16 的原因是：

- 计算量较小，FP8 收益不明显
- 位置编码影响 attention score
- 数值误差可能影响长上下文稳定性
- FP8 quant/dequant 可能比计算本身更贵

### 5.3 Softmax

Softmax 对数值稳定性非常敏感，尤其 attention score 中包含 scale、mask、exp 和归一化。

直接 FP8 softmax 容易带来：

- 溢出或下溢风险
- attention 分布失真
- 长上下文中误差放大
- 模型输出质量下降

因此，即使 attention matmul 可以使用 FP8，softmax 本身通常仍保留 FP16/BF16 或使用更高精度累加。

### 5.4 Residual Add

Residual add 是 Transformer 的主干路径。它把当前 block 的输出与输入残差相加，误差会沿着层数不断传播。

保留 FP16/BF16 的原因是：

- 误差会跨层累积
- residual 是主信息通道
- 单个加法计算量小，FP8 加速收益低
- 额外转换可能抵消收益

### 5.5 Router / Dispatch / Combine

MoE router、expert dispatch、token permutation、combine 和 scheduler 这类操作不是大矩阵乘法，而是控制流、索引、排序、重排和通信。

它们不适合直接 FP8 的原因是：

- 不使用 Tensor Core 主路径
- 性能瓶颈常在内存访问、索引和通信
- 量化不会显著减少控制流开销
- 可能引入额外数据格式转换

这类路径的优化重点不是 FP8，而是：

- 更好的 expert 分桶
- grouped GEMM 调度
- 减少 token permutation 开销
- 减少跨卡通信
- batch 和路由负载均衡

## 6. 简化判定表

| 判定问题 | 如果答案是“是” | 结论 |
|---|---|---|
| 是否是大 GEMM / Linear？ | 是 | 优先考虑 FP8 |
| 是否大量读取权重？ | 是 | FP8 有显存和带宽收益 |
| 是否大量读取 KV cache？ | 是 | FP8 KV cache 有明显收益 |
| 是否有 Hopper FP8 kernel 支持？ | 是 | FP8 更可能带来实际加速 |
| 是否数值稳定性敏感？ | 是 | 保留 FP16/BF16 |
| 是否只是小算子或控制流？ | 是 | FP8 收益通常很低 |
| 是否需要额外 quant/dequant？ | 是 | 必须评估转换开销 |
| 误差是否会跨层累积？ | 是 | 谨慎量化，通常保留高精度 |

## 7. Kernel 耗时开销排名

### 7.0 已有 BF16 vs FP8 profile 映射

你提供的现有耗时占比分析如下，这组数据应作为当前模型结构下的优先级依据：

| Rank | Kernel 类别 | BF16 占比 | FP8 占比 | 变化 |
|---:|---|---:|---:|---:|
| 1 | Fused MoE | 28.9% | 23.8% | -5.1pp |
| 2 | all_reduce / NCCL | 18.1% | 19.9% | +1.8pp |
| 3 | dense GEMM / DG + nvjet | 7.9% | 8.7% | +0.8pp |
| 4 | FlashAttention | 7.9% | 8.2% | +0.3pp |
| 5 | RMSNorm reduce + Triton | 9.0% | 10.1% | +1.1pp |
| 6 | GDN 线性注意力 | 9.4% | 9.5% | +0.1pp |
| 7 | Quant / FP8 独有开销 | 0.0% | 5.3% | +5.3pp |
| 8 | Other / elementwise / SiLU / routing | 14.1% | 14.5% | +0.4pp |

这组数据给出几个关键结论：

1. Fused MoE 仍然是第一大头，但 FP8 后占比从 28.9% 降到 23.8%，说明 FP8 对 MoE grouped GEMM 已经产生收益。
2. all_reduce / NCCL 占比从 18.1% 升到 19.9%，说明计算被压缩后，通信占比被动上升。后续端到端收益会越来越受通信限制。
3. dense GEMM 和 FlashAttention 在 FP8 下占比略升，不一定代表绝对耗时变慢，也可能是 MoE 变快后其他模块占比被动上升。
4. RMSNorm 从 9.0% 升到 10.1%，说明 FP8 后小算子和 reduce 类 kernel 更突出，可能需要 fusion 或 persistent path 优化。
5. Quant 是 FP8 独有 5.3% 开销，这是 FP8 端到端收益的重要抵消项。若 Q/DQ、scale、layout transform 没融合，FP8 理论收益会被明显吃掉。
6. GDN 线性注意力占比基本不变，说明 hybrid attention 中这一路径没有明显从 FP8 获益，或其瓶颈不在标准 GEMM/FP8 Tensor Core。

#### R1 到 R8 的含义

图中的 `R1` 到 `R8` 表示 profile ranking bucket，也就是把端到端推理耗时按 kernel / runtime 类别聚合后得到的排名编号。

需要注意三点：

1. `R` 不是 Transformer layer 编号。`R1` 不等于第 1 层，`R2` 不等于第 2 层。
2. `R` 也不是模型前向执行顺序。比如 `R2 all_reduce / NCCL` 可能出现在多个 layer 的 attention projection、MoE 边界或 TP/EP 同步点。
3. `R` 是 profile 归因类别。一个 `R` 可能由很多 layer 上的同类 kernel 累加而成。

因此，`R1` 到 `R8` 应按下面方式理解：

| 标记 | Profile bucket | 在模型结构中的来源 | 为什么这样标 |
|---|---|---|---|
| `R1` | Fused MoE | MoE expert 的 up/gate/down grouped GEMM | 当前 profile 第一大耗时项，BF16 28.9%，FP8 23.8% |
| `R2` | all_reduce / NCCL | TP8 / EP / distributed runtime 的跨卡同步 | 第二大耗时项，不是单个 layer 内的数学算子，而是系统通信开销 |
| `R3` | dense GEMM / DG + NvJet | QKV projection、O projection、非 MoE dense linear | 普通 dense linear 的 GEMM backend，不等于 MoE expert GEMM |
| `R4` | FlashAttention | full attention 或可走 FlashAttention 的 attention path | 主要反映 attention 的 QK/softmax/V 块化计算和 HBM IO |
| `R5` | RMSNorm reduce + Triton | attention 前、MoE 前后的 RMSNorm | reduce 类小算子，FP8 后容易因 GEMM 变快而占比上升 |
| `R6` | GDN 线性注意力 | Hybrid Attention 中的 GDN / linear attention 分支 | 属于 hybrid attention 的另一条 attention kernel 路径 |
| `R7` | Quant / FP8 独有开销 | Q/DQ、scale、cast、layout transform、packing | BF16 路径没有，是 FP8 执行路径新增的转换成本 |
| `R8` | Other / elementwise / SiLU / routing | SiLU、multiply、routing、dispatch/combine、residual 等 | 单个 kernel 小，但数量多，端到端聚合后可见 |

一个更准确的读法是：

```text
Transformer Layer / Block
  -> 产生多个 kernel 和通信边界
  -> profile 把同类 kernel 跨 layer 聚合
  -> 按总耗时排序得到 R1, R2, ..., R8
```

所以图上的 `R1`、`R3`、`R4`、`R5`、`R6` 可以画在具体模块附近；`R2` 和 `R7` 则更像执行系统附加在模块边界上的成本，应该画在通信通道或 FP8 转换链路附近。

#### dense GEMM / DG + NvJet 解释

`dense GEMM / DG + nvjet` 是 profile 中的第三大项：

```text
dense GEMM / DG + nvjet:
BF16 7.9%
FP8  8.7%
```

这里的 `dense GEMM` 指非 MoE 的普通 dense linear 矩阵乘法。它和 `Fused MoE` 要区分开：

- `Fused MoE`：MoE expert 的 grouped GEMM，通常包含 top-k expert 的 up/gate/down 路径，是 R1。
- `dense GEMM`：所有 token 都会经过的 dense linear 层，是 R3。

在 Transformer block 中，dense GEMM 主要来自：

1. QKV Projection  
   将 hidden state 投影成 Query、Key、Value。

2. O Projection  
   attention 输出后的 dense projection。

3. 其他非 expert dense linear  
   例如某些辅助 projection、输出路径或 runtime 中未归到 MoE 的 dense matmul。

`DG + NvJet` 按当前 profile label 可以理解为 dense GEMM 进入的优化 kernel/backend 路径。这里不把它解释成新的模型层，而是解释成 dense GEMM 的执行后端：

```text
hidden state
  -> dense linear GEMM
  -> DG + NvJet optimized kernel/backend
  -> output activation
```

因此它在图中的映射位置是：

- Fused QKV Projection GEMM
- O Projection GEMM
- 其他 dense projection

其中，图 1 的 `Fused QKV Projection GEMM` 对应 `R3 dense GEMM / DG + NvJet`，不是 `R7`。原因是 profile 把它的主计算归到普通 dense GEMM bucket：

```text
R3 = Fused QKV Projection GEMM 本体计算
   = hidden @ fused[Wq, Wk, Wv]
```

但如果这条路径使用 FP8，GEMM 前后会额外出现量化转换成本：

```text
FP16/BF16 activation
  -> R7 quant / scale / layout transform
  -> R3 Fused QKV Projection GEMM
  -> optional R7 dequant / rescale / cast
  -> Q, K, V activation
```

所以读图时要把两件事拆开：

- `R3` 是 QKV GEMM 的主计算。
- `R7` 是为了让 QKV GEMM 走 FP8 而附着在它前后的转换手续费。

FP8 后 dense GEMM 占比从 7.9% 到 8.7%，不一定说明它绝对变慢。更常见的解释是：

1. Fused MoE 通过 FP8 得到更大收益，占比下降。
2. dense GEMM 也可能变快，但没有 MoE 降得多。
3. 因此 dense GEMM 在总耗时里的相对占比略升。

后续优化 dense GEMM 时，重点不是把它和 MoE 混在一起，而是看：

- QKV 是否真正 fused
- Q/K/V scale 是否导致 GEMM 被拆分
- FP8 layout transform 是否被提前完成
- DG/NvJet path 是否使用了目标 shape 的最优 kernel
- dense GEMM 后是否触发额外 all_reduce

#### TP8 Transformer Block 内通信解释

你的部署策略是 `TP8`，即一个 Transformer block 内的 tensor 被切到 8 张 GPU 上并行计算。以 dense projection 为例：

```text
X @ W
```

TP8 会把 `W` 或输出维度切成 8 份，每张 GPU 只算一部分：

```text
GPU0: partial_0 = X @ W_0
GPU1: partial_1 = X @ W_1
...
GPU7: partial_7 = X @ W_7
```

根据具体 parallel linear 的切法，后面可能需要：

- all_reduce
- reduce_scatter
- all_gather
- 或者在下一层保持 shard 状态，延迟同步

这就是为什么 `all_reduce / NCCL` 不画在某个单独 layer 内部，而应该画在 dense GEMM、attention projection、MoE 边界旁边。它是 TP8 执行方式带来的系统同步。

TP8 block 内通信的核心关系是：

```text
dense GEMM shard / MoE shard
  -> partial output
  -> NCCL collective
  -> next layer input
```

FP8 会减少 GEMM 的计算和权重带宽，但不一定等比例减少 all_reduce，因为 all_reduce 传的是 activation / partial output，而不是全部权重。因此 FP8 后通信占比可能上升。

#### TP8 跨机器部署解释

如果 TP8 的 8 张 GPU 跨机器部署，通信成本会比单机 TP8 更敏感。

单机 TP8 常见路径：

```text
GPU0..GPU7 within one node
  -> NVLink / NVSwitch
  -> NCCL collective
```

跨机器 TP8 路径可能变成：

```text
Machine A: GPU0..GPU3
Machine B: GPU4..GPU7
  -> intra-node NVLink / NVSwitch
  -> cross-node network
  -> NCCL collective
```

跨机器部署的问题在于：

- 跨节点带宽通常低于机内 NVLink/NVSwitch
- 跨节点延迟更高
- all_reduce 的同步等待更难隐藏
- 每个 Transformer layer 都可能重复触发 collective
- FP8 把计算压快后，跨节点通信更容易成为端到端瓶颈

因此，如果当前 `R2 all_reduce / NCCL` 已经达到 18% 到 20% 量级，TP8 跨机器部署需要重点评估：

1. TP8 是否必须跨机器，能否改成单机 TP8 或 TP4 + EP/PP。
2. 是否能减少每层 all_reduce 次数。
3. 是否能用 reduce_scatter/all_gather 替代部分 all_reduce。
4. 是否能让 NCCL 与 GEMM、MoE dispatch、attention 计算 overlap。
5. 是否存在不均衡 expert routing 导致某些 GPU 等待。
6. 跨节点网络拓扑是否匹配 NCCL ring/tree 的路径。

#### 不同 layer 之间的关系

Hybrid Attention 模型中，不同 layer 的 attention 形态可能不同。例如：

```text
Layer 0: Full Attention + MoE
Layer 1: Sliding Window Attention + MoE
Layer 2: GDN / Linear Attention + MoE
Layer N: Global / Sparse Attention + MoE
```

这意味着：

- 每层都会重复出现 dense GEMM、attention、MoE、norm、residual。
- 但每层 attention 的 KV 读取量不同。
- Full/global 层更容易放大 FlashAttention 和 KV cache 开销。
- Sliding window 层会降低 attention 开销，使 MoE 或通信更突出。
- GDN/linear attention 层可能不吃标准 FlashAttention 的收益。
- TP8 的通信可能在每层 dense projection 或 MoE 边界重复出现。

因此，分析时不要只看单层结构图，还要看 layer stack：

```text
layer pattern
  -> attention mode
  -> KV read amount
  -> GEMM shape
  -> TP8 collective frequency
  -> end-to-end profile rank
```

#### Layer 与 Transformer block 的关系

在 decoder-only LLM 中，`layer` 通常就是一个重复堆叠的 `Transformer block`。二者可以这样理解：

- layer 是堆叠编号，例如 Layer 0、Layer 1、Layer 2。
- Transformer block 是每个 layer 内部的计算结构。
- 一个 block 通常包含 attention 子层、MoE/MLP 子层、RMSNorm、residual add。

简化结构如下：

```text
Layer i / Transformer Block i
  ├─ RMSNorm
  ├─ Attention sub-layer
  │    ├─ QKV projection
  │    ├─ attention mode: full / sliding / GDN / sparse-global
  │    └─ O projection
  ├─ Residual add
  ├─ RMSNorm
  ├─ MoE / MLP sub-layer
  │    ├─ router / gate
  │    ├─ expert up / gate / down GEMM
  │    └─ dispatch / combine
  └─ Residual add
```

不同 layer 的区别主要在 attention mode：

```text
Layer 0: Full Attention + MoE
Layer 1: Sliding Window Attention + MoE
Layer 2: GDN / Linear Attention + MoE
Layer 3: Sliding Window Attention + MoE
Layer N: Sparse / Global Attention + MoE
```

因此，图中画的是一个“代表性 block”，但真实模型是多个 block 的堆叠。Profile 中的每一类 kernel 占比，是所有 layer 上相同类型 kernel 的总和。例如：

- `Fused MoE 28.9% / 23.8%` 是所有 layer 的 MoE expert GEMM 汇总。
- `dense GEMM 7.9% / 8.7%` 是所有 layer 的 QKV/O/其他 dense GEMM 汇总。
- `FlashAttention 7.9% / 8.2%` 是所有 full attention 或可走 FlashAttention 的层汇总。
- `GDN 线性注意力 9.4% / 9.5%` 是所有 GDN/linear attention 层汇总。
- `all_reduce / NCCL 18.1% / 19.9%` 是所有 layer 中 TP/EP collective 的总通信开销。

这也是为什么单个 block 图不能只画一个 attention kernel：Hybrid Attention 的 profile 要按 layer pattern 汇总理解。

#### MoE / MLP 边界上的 all_reduce / reduce_scatter / all_gather

MoE/MLP 边界的通信取决于 TP8、EP 和 parallel linear 的切分方式。核心问题是：当前张量是 sharded 还是 replicated，下一步 kernel 需要 sharded input 还是 full input。

图 7 不是新增了一个模型层，而是把图 1 的 `MoE Feed Forward Layer` 展开。对应关系如下：

| 图 1 MoE Feed Forward Layer 模块 | 图 7 展开后的含义 | Profile 映射 |
|---|---|---|
| `activation` | MoE 子层输入 hidden state | 边界张量 |
| `Router / Gate` | 为 token 选择 top-k experts，并准备 dispatch | R8 / routing，部分系统开销 |
| `Expert dispatch / permute / combine` | 将 token 送到对应 expert owner GPU，或做本地重排 | R8，跨卡时也贡献 R2 |
| `Top-k Expert up GEMM` | expert MLP 的 up projection | R1 Fused MoE |
| `Expert gate GEMM` | expert MLP 的 gate projection | R1 Fused MoE |
| `SiLU + multiply` | gate 激活与 up 输出逐元素相乘 | R8 elementwise / SiLU |
| `Expert down GEMM` | expert MLP 的 down projection，回到 hidden size | R1 Fused MoE |
| `reduce_scatter / all_reduce / all_gather` | 连接 MoE 输入、expert 输出和下一层输入的 TP/EP 同步 | R2 all_reduce / NCCL |

因此：

```text
Fused MoE Expert GEMM
  = Top-k Expert up GEMM
  + Expert gate GEMM
  + Expert down GEMM
  + grouped/fused execution wrapper
```

它对应的是图 1 中 MoE Feed Forward Layer 内部的多个 expert GEMM，不是图 1 外面单独多出来的一层。

典型路径如下：

```text
Attention output
  -> maybe all_gather
  -> router / token dispatch
  -> expert up GEMM + expert gate GEMM
  -> SiLU + multiply
  -> expert down GEMM
  -> reduce_scatter or all_reduce
  -> residual / next layer input
```

常见 collective 的含义：

| Collective | 作用 | 典型出现位置 |
|---|---|---|
| all_gather | 把各 GPU 的 shard 拼成完整 hidden | expert 或后续 kernel 需要 full hidden 时 |
| reduce_scatter | 对 partial output 求和，并把结果继续切片 | row-parallel 输出后，希望保持 sharded 状态 |
| all_reduce | 对所有 GPU 的 partial output 求和，并让每张 GPU 都拿到完整结果 | 下一步需要 replicated full output 时 |

在 TP8 中，dense GEMM 或 expert GEMM 可能按列或按行切分：

```text
Column parallel:
  each GPU computes part of output channels
  output may be concatenated or kept sharded

Row parallel:
  each GPU computes partial sum
  output usually needs all_reduce or reduce_scatter
```

MoE 增加了另一层复杂度：token 会被 router 分配到不同 expert。如果 expert parallel 跨 GPU 或跨机器，还会有 token dispatch/combine：

```text
tokens
  -> router selects experts
  -> dispatch tokens to expert owner GPUs
  -> expert grouped GEMM
  -> combine expert outputs
  -> TP collective if boundary requires synchronization
```

因此，`R2 all_reduce / NCCL` 可能来自多个边界：

1. Attention O projection 后的 TP 同步。
2. MLP/MoE down projection 后的 TP 同步。
3. MoE expert dispatch/combine 的跨 GPU token exchange。
4. 下一层输入格式要求从 sharded 变为 replicated，或反过来。

如果 TP8 跨机器部署，这些 collective 的代价会被放大。因为一部分通信从机内 NVLink/NVSwitch 变成跨节点网络，延迟和带宽都会更差。对于已经有 `18% ~ 20%` NCCL 占比的 profile，MoE/MLP 边界的 collective 是必须单独优化的对象。

优化策略：

1. 尽量让相邻 kernel 接受相同 shard layout，减少 sharded/full 来回转换。
2. 用 reduce_scatter 替代 all_reduce，如果下一步可以继续消费 sharded output。
3. 将 MoE dispatch/combine 与 grouped GEMM 做 overlap。
4. 对 TP/EP 切分做联合设计，避免 expert 分配导致跨机器 token exchange 过重。
5. 尽量避免 TP8 跨机器；如果必须跨机器，优先保证 high-frequency collective 在机内完成。
6. 检查每层 attention mode 和 MoE 边界是否触发重复 collectives。

#### TP8 层间传递与一机 8 卡 Expert 分布

图 8 新增了一个动态图，用来解释两件事：

1. TP8 下 layer 与 layer 之间传递的不是单一完整张量，而通常是按 8 张 GPU 切分后的 activation shards。
2. MoE experts 在单机 8 卡上有 owner placement；router 选中 top-k expert 后，token 需要被 dispatch 到对应 owner GPU。

简化后的 layer 间状态如下：

```text
Layer i output
  h = [h0, h1, h2, h3, h4, h5, h6, h7]
      GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7

  -> optional all_reduce / reduce_scatter / all_gather

Layer i+1 input
  consumes sharded hidden or gathered hidden
```

关键判断是：下一步 kernel 能不能直接消费 sharded hidden。

- 如果下一步 kernel 可以消费 sharded hidden，就尽量保持 shard layout，避免 all_gather。
- 如果 row-parallel GEMM 产生 partial sum，输出边界通常需要 all_reduce 或 reduce_scatter。
- 如果 router、expert 或后续 kernel 需要完整 hidden，则可能需要 all_gather。

MoE expert 在一机 8 卡上的一个示例放置如下。假设该层有 16 个 experts，每张 GPU 放 2 个 experts：

```text
GPU0 / TP0: E0,  E8
GPU1 / TP1: E1,  E9
GPU2 / TP2: E2,  E10
GPU3 / TP3: E3,  E11
GPU4 / TP4: E4,  E12
GPU5 / TP5: E5,  E13
GPU6 / TP6: E6,  E14
GPU7 / TP7: E7,  E15
```

实际 expert 数量可以不同，但读图方法不变：每个 expert 有 owner GPU，router 输出的 expert id 决定 token 要发到哪张卡。

MoE 动态路径可以写成：

```text
token hidden shard
  -> router computes top-k experts
  -> dispatch token to expert owner GPU
  -> local grouped expert GEMM
       up GEMM + gate GEMM + down GEMM
  -> combine expert outputs
  -> reduce_scatter / all_reduce if TP boundary requires
  -> next layer input
```

这和 profile 的关系是：

| 动态阶段 | 对应 profile bucket | 说明 |
|---|---|---|
| expert up/gate/down grouped GEMM | `R1 Fused MoE` | 真正的大矩阵计算主体 |
| TP boundary collective | `R2 all_reduce / NCCL` | layer 间、MoE 边界或 projection 边界的同步 |
| router top-k | `R8 Other / routing` | 控制流和小算子 |
| dispatch / combine | `R8`，跨卡时也贡献 `R2` | 本地重排偏 R8，跨 GPU token exchange 偏 R2 |
| quant / dequant / scale | `R7 Quant / FP8 独有` | 如果 FP8 路径在 MoE 前后插入转换，会形成额外开销 |

因此，一机 8 卡 TP8 的重点不是只看 `Fused MoE Expert GEMM` 是否够快，还要看：

1. expert owner 是否均匀，避免某些 GPU expert 过热。
2. top-k routing 是否导致 token 大量跨 GPU dispatch。
3. dispatch/combine 是否能和 grouped GEMM overlap。
4. layer 间是否频繁在 sharded 和 full hidden 之间切换。
5. 如果扩展到跨机器，expert owner 和 TP rank 是否会让 token exchange 走跨节点网络。

动态图中的蓝色流表示 TP8 activation shard / collective，橙色流表示 token dispatch 到 expert owner，红色流表示 expert grouped GEMM 后 combine 回 layer boundary。这个图的重点是把 `R1` 和 `R2` 的边界区分清楚：`R1` 是 expert GEMM 计算，`R2` 是 TP/EP 通信同步。

#### Quant / FP8 独有开销解释

`Quant / FP8 独有` 指的是 BF16 路径中不存在、但 FP8 推理为了使用低精度计算而额外引入的量化链路成本。它不是模型原本的 Transformer 层，而是 FP8 执行路径中的转换和准备开销。

在图 1 里，`R7` 已用黄色框画在 FP8 GEMM 前后：例如 `Fused QKV Projection GEMM` 前的 Q/DQ、scale、layout transform，`O Projection GEMM` 前的 Q/DQ、scale/cast，以及 MoE expert GEMM 前后的 expert scale、dequant、rescale、cast。它表示附着在 FP8 计算路径上的转换成本，而不是单独的模型层。

典型来源包括：

| 开销类型 | 说明 |
|---|---|
| Quantize | 将 FP16/BF16 activation、KV cache 或中间张量转成 FP8 |
| Dequantize | 将 FP8 数据还原到 FP16/BF16 或 FP32 accumulator 路径 |
| Scale apply | 应用 scale，恢复或调整动态范围 |
| Amax / absmax | 动态量化时统计 tensor/block 最大绝对值 |
| Cast | FP16/BF16 与 FP8 之间的数据类型转换 |
| Layout transform | 为 FP8 GEMM、attention kernel 或 Tensor Core tile 重排内存 |
| Rescale | 例如 Q/K/V 三套 weight scale 为了 fused QKV GEMM 合并而做 scale 对齐 |
| Padding / packing | 为 FP8 kernel 的 tile 对齐做 pack、padding 或重排 |

BF16 路径中该项为 0，是因为 BF16 推理通常直接执行：

```text
BF16 activation + BF16 weight -> BF16/FP32 accumulate GEMM
```

而 FP8 路径可能变成：

```text
BF16 activation
  -> quantize / scale
  -> FP8 activation
  -> FP8 GEMM
  -> dequant / rescale
  -> BF16 activation
```

因此，FP8 虽然减少了权重和 KV cache 的带宽与显存压力，但会额外引入 quant/dequant、scale 和 layout 相关 kernel。你给出的 profile 中：

```text
Quant:
BF16 0.0%
FP8  5.3%
```

这说明 FP8 已经带来了额外 5.3% 的执行成本。这个成本会抵消一部分 Fused MoE、dense GEMM 和 KV cache 的收益。

后续优化方向：

1. 将 quant/dequant 融合进 GEMM 或 attention kernel。
2. 避免 activation 在 FP16/BF16 和 FP8 之间反复转换。
3. 对 scale 做预处理，减少 runtime 除法、rescale 和 amax 统计。
4. 预先完成 weight layout transform，避免推理时动态重排。
5. 对 KV cache 优先采用低开销 FP8 cast 或稳定的 per-block scale。
6. 对 Fused QKV 统一或对齐 Q/K/V scale，避免因为 scale 不一致拆分 GEMM。
7. 将 padding、packing 和 Q/DQ 与主 kernel 合并，减少额外 launch。

一句话总结：

> `Quant / FP8 独有` 是 FP8 加速的转换成本。它越高，FP8 的理论收益被吃掉越多。

#### all_reduce / NCCL 开销解释

`all_reduce / NCCL` 是 profile 中的第二大项：

```text
all_reduce / NCCL:
BF16 18.1%
FP8  19.9%
```

它不是 Transformer block 内部的某个层，而是多卡并行推理引入的系统通信开销。对于 `Qwen3.5-122B-A10B` 这种 122B 级别 MoE 模型，通常需要多卡部署，常见通信来源包括：

1. Tensor Parallel all-reduce  
   dense GEMM 或 attention projection 被切到多张 GPU 后，需要聚合 partial output。

2. Expert Parallel dispatch/combine  
   MoE token 被路由到不同 expert 或不同 GPU 上，需要 token exchange、dispatch 和 combine。

3. Pipeline / runtime synchronization  
   多阶段执行、batch 调度、跨设备依赖会引入同步等待。

4. KV cache 或中间状态迁移  
   某些并行策略下，KV cache 或中间激活需要跨设备传递或重排。

FP8 后 NCCL 占比从 18.1% 升到 19.9%，通常不表示通信绝对时间一定变慢，而是因为 Fused MoE 等计算项变快后，通信在总时间中的相对占比被动上升。

这意味着 FP8 优化进入下一阶段后，瓶颈会从“纯计算”转向“计算 + 通信 + 小算子”的混合瓶颈。后续如果只继续优化 GEMM，端到端收益会被 NCCL 限制。

优化方向包括：

1. 减少 tensor parallel all-reduce 次数，例如更好的 fusion 或 reduce-scatter/all-gather 重排。
2. 优化 MoE expert parallel 的 token dispatch/combine，降低跨卡 token 交换量。
3. 做通信计算 overlap，让 NCCL 与 GEMM、attention 或 quant kernel 并行。
4. 调整 TP/EP/PP 并行切分，避免通信量过大的 shard 方案。
5. 优化 batch 和 expert load balance，减少某些 GPU 等待慢 expert。
6. 如果硬件支持，使用更高效的 NCCL topology、NVLink/NVSwitch 路径和通信 bucket 配置。

一句话总结：

> `all_reduce / NCCL` 是 FP8 后暴露出来的系统级瓶颈。它不在单层数学结构里，但会直接决定 122B MoE 模型的端到端吞吐。

### 7.1 Prefill 阶段

Prefill 阶段输入 token 较多，矩阵乘法规模较大，更接近 compute-bound。对于 `Qwen3.5-122B-A10B`，预计耗时排序如下：

1. MoE expert GEMM  
   top-k expert 的 up/gate/down grouped GEMM，A10B active 参数下通常是最大计算项。

2. QKV / O Projection GEMM  
   dense GEMM，适合 FP8 和 fusion。QKV scale 拆分会损失融合收益。

3. Hybrid Attention Core  
   Full attention 层在长上下文时占比上升；sliding window 层的 KV 读取量较小；sparse/global 层取决于 mask 与 kernel 实现。FlashAttention 的核心仍是减少 HBM IO。

4. MoE Dispatch / Combine  
   routing、token permutation、expert 分桶，多卡 expert parallel 时通信放大。

5. Norm / RoPE / Activation  
   单项小，但 kernel 碎片多，会影响尾延迟和 launch 开销。

### 7.2 Decode 阶段

Decode 阶段每次生成一个 token，GEMM 形状变瘦，KV cache 读取成为核心压力，尤其在 full attention 层和长上下文下。Hybrid Attention 会让不同层的 attention 开销差异变大。

预计耗时排序如下：

1. Hybrid KV Cache Attention  
   Full attention 读取完整历史 K/V；sliding window 只读取局部窗口；sparse/global 读取选定位置。FP8 KV cache 对容量和带宽都有收益，但 full attention 层收益最大。

2. MoE Grouped GEMM  
   batch 小时 GEMM 利用率下降，kernel 实现质量决定收益。在 sliding window 层占比较高时，MoE GEMM 可能超过 attention 成为 decode 第一瓶颈。

3. QKV / O GEMM  
   decode 下受小 batch、layout 和 fusion 影响较大。

4. TP / EP 通信  
   122B 级别模型大概率多卡部署，all-reduce、expert dispatch、KV 迁移可能进入前五。

5. Sampling / Scheduler / 小算子  
   对平均吞吐不一定最大，但会影响 P95/P99。

## 8. FP8 理论收益空间

### 8.1 权重显存

122B 参数模型：

- BF16 权重：`122B * 2 bytes ≈ 244GB`
- FP8 权重：`122B * 1 byte ≈ 122GB`

理论上权重显存约减半。实际部署还要考虑 scale、metadata、padding、tensor parallel shard 和 runtime buffer。

### 8.2 Active 参数读取

A10B active 参数：

- BF16 active 权重读取：`10B * 2 bytes ≈ 20GB/token`
- FP8 active 权重读取：`10B * 1 byte ≈ 10GB/token`

如果 decode 阶段偏 memory-bound，FP8 可以显著减少权重读取压力。但真实收益会被 dequant、MoE dispatch、通信和小 batch GEMM 利用率限制。

### 8.3 KV Cache

KV cache 从 FP16/BF16 到 FP8 后，容量和读带宽理论上约减半。

长上下文 decode 中，KV cache 可能成为最主要瓶颈，因此 FP8 KV cache 对端到端收益非常关键。

在 Hybrid Attention 下，KV cache 收益需要分层看：

- Full attention 层：收益最明显，读取长度约等于完整上下文长度。
- Sliding window 层：收益仍然存在，但读取长度约等于窗口大小，attention 占比会被压低。
- Sparse/global 层：收益取决于全局 token 数、稀疏模式和 gather/scatter kernel。

因此，如果模型中 sliding window 层占比高，FP8 KV cache 的平均收益会低于纯 full attention 模型；如果 full/global 层承担主要全局信息汇聚，长上下文 decode 中 FP8 KV cache 仍然是关键优化点。

### 8.4 工程收益区间

参考 FP8 报告：

- Opt-0 全流程 FP8：大 batch 最多约 19%，小 batch 可能负优化
- CUTLASS FP8 GEMM：平均约 1.33x
- Fused QKV FP8 GEMM：平均约 1.38x
- 加入 FP8 KV cache：平均约 1.54x
- 相比 FP16 LMDeploy：约 1.4x 到 1.86x

因此对 `Qwen3.5-122B-A10B` 更现实的端到端收益判断是：

- 普通 FP8 接入：0% 到 20%，甚至可能负优化
- FP8 GEMM + QKV fusion：约 1.3x 到 1.5x
- FP8 GEMM + Fused QKV + FP8 KV cache：约 1.4x 到 1.8x
- 长上下文 decode 或高并发场景：有机会接近 1.8x
- 超过 2x：需要非常强的 kernel、调度、通信和 attention 优化配合

## 9. 实施优先级

建议优化顺序如下：

1. MoE expert grouped GEMM FP8
2. Fused QKV GEMM FP8
3. O Projection GEMM FP8
4. FP8 KV cache，区分 full / sliding / sparse-global 层收益
5. Hybrid Attention kernel：FlashAttention / Flash-Decoding / PagedAttention / sliding-window attention FP8 路径
6. 减少 quant/dequant、scale、layout transform 的额外 kernel
7. MoE dispatch/combine 与多卡通信优化
8. 小算子融合，包括 RMSNorm、RoPE、residual 附近的 launch 开销优化

## 10. 结论

FP8 的核心价值不在于“所有层都变成 FP8”，而在于把最重、最适合低精度硬件加速的路径变成 FP8。

对于 `Qwen3.5-122B-A10B`：

- 应优先 FP8：Linear/GEMM、MoE expert GEMM、QKV/O projection、KV cache、attention matmul
- 应保留 FP16/BF16：RMSNorm、RoPE、LogN Scaling、softmax、residual、router 和调度控制流
- Hybrid Attention 会改变 attention 开销分布：full 层更吃 KV cache，sliding window 层降低 KV 读取，sparse/global 层取决于 mask 和 kernel
- 端到端收益关键：FP8 GEMM、Fused QKV、FP8 KV cache、Hybrid Attention kernel、MoE grouped GEMM 必须协同

因此，图中的分层逻辑可以概括为：

> 大矩阵乘法和大规模读写数据走 FP8；数值稳定敏感、小算子和控制流保留 FP16/BF16。
