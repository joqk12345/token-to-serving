import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const outDir = path.resolve("downloads/llmsystem2026spring/freshman_docs");
const pdfBase = "..";

const docs = [
  {
    slug: "01-llm-system-overview",
    title: "LLM 系统全景",
    module: "LLM 系统全景",
    pdf: "llmsys-01-intro-14e74a426e4a7e3ed485a026e1f65b70.pdf",
    goals: ["区分模型、系统和应用", "理解为什么 LLM 需要专门的系统支持", "建立后续课程的知识地图"],
    ideas: [
      "LLM 的核心任务可以简化为“根据上下文预测下一个 token”。系统课程关心的是怎样让这个预测过程在真实机器上可训练、可部署、可扩展。",
      "一个完整 LLM 系统通常包括数据处理、模型结构、训练框架、硬件加速、推理服务、监控和评测。任何一环太慢或太贵，都会影响最终用户体验。",
      "新生不需要一开始记住所有算法名称，先抓住三条线：模型怎样计算，硬件怎样执行，服务怎样面对很多用户请求。"
    ],
    example: "把 LLM 想成校园问答助手。模型负责生成回答，GPU 负责快速计算，服务系统负责同时处理很多同学的请求，并尽量让每个人少等。",
    questions: ["如果模型参数不变，系统层面还能优化什么？", "为什么“能跑一次”和“能服务很多用户”不是一回事？"]
  },
  {
    slug: "02-gpu-programming-basics-1",
    title: "GPU 编程基础 1",
    module: "GPU 与并行计算",
    pdf: "llmsys-02-gpu-programming-c64a0141b96a1f384db7f6717ed8e039.pdf",
    goals: ["理解 GPU 为什么适合矩阵计算", "认识线程、block 和 grid", "知道并行程序不等于自动变快"],
    ideas: [
      "GPU 由大量较简单的计算单元组成，适合同时执行很多相似的小任务。神经网络里的矩阵乘法正好具有这种结构。",
      "CUDA 程序把工作拆成许多线程，线程组成 block，block 组成 grid。程序员需要决定每个线程负责哪一小块数据。",
      "并行计算的难点是数据移动、线程同步和资源利用率。计算单元很多，但如果数据取不来，GPU 仍然会等待。"
    ],
    example: "全班 100 人一起抄写 1000 行数字，比 1 人抄快得多。但如果只有一本纸质材料，每个人都排队看原文，速度就会被“取数据”限制。",
    questions: ["为什么矩阵乘法容易并行？", "线程数量越多一定越快吗？为什么？"]
  },
  {
    slug: "03-gpu-programming-basics-2",
    title: "GPU 编程基础 2",
    module: "GPU 与并行计算",
    pdf: "llmsys-03-gpu-programming2-b82b6ffdf554494747d00ce7ac606c3b.pdf",
    goals: ["理解 GPU 内存层级", "认识 shared memory 的作用", "理解访存模式对性能的影响"],
    ideas: [
      "GPU 里的内存不是一种。全局内存容量大但慢，共享内存小但快，寄存器更快但更稀缺。",
      "高性能 kernel 会尽量复用已经取到片上的数据，减少反复访问慢速内存。",
      "访存是否连续、是否合并、是否造成 bank conflict，都会影响实际性能。很多优化本质上是在减少无效数据移动。"
    ],
    example: "做小组作业时，把常用资料放在小组桌上比每次跑到图书馆借快。shared memory 就像每个小组桌上的临时资料区。",
    questions: ["为什么同样的计算量，访存方式不同会导致速度差很多？", "shared memory 的优点和限制分别是什么？"]
  },
  {
    slug: "04-gpu-acceleration",
    title: "GPU 加速方法",
    module: "GPU 与并行计算",
    pdf: "llmsys-04-gpu-acceleration-48ffa5768ba62c54138f0a71ca2b68b8.pdf",
    goals: ["理解加速不是只增加硬件", "认识算子、kernel 和性能瓶颈", "能从计算和访存两方面分析问题"],
    ideas: [
      "GPU 加速通常先定位最耗时的算子，再考虑并行划分、数据布局、内存复用和同步开销。",
      "深度学习框架会调用底层高性能库，但理解基础原理有助于判断什么时候库函数不够好。",
      "性能优化要基于测量。没有 profiling，就很容易优化错地方。"
    ],
    example: "如果食堂排队慢，可能是窗口少，也可能是取餐路线绕、付款慢或备餐慢。系统优化也要先找真正瓶颈。",
    questions: ["什么是算子？", "为什么优化前要先测量？"]
  },
  {
    slug: "05-deep-learning-frameworks-autodiff",
    title: "深度学习框架与自动微分",
    module: "从 Token 到 Transformer",
    pdf: "llmsys-05-dlframework-fa0770d636572de3f7b48ccae0ba8848.pdf",
    goals: ["理解训练为什么需要梯度", "认识自动微分的价值", "知道框架如何连接模型和硬件"],
    ideas: [
      "训练神经网络就是不断调整参数，让模型输出更接近目标。梯度告诉我们每个参数应该往哪个方向调。",
      "自动微分让开发者写前向计算，框架自动构建反向传播所需的计算过程。",
      "PyTorch、TensorFlow 等框架不仅提供数学接口，也负责调度底层算子、管理显存和连接 GPU 库。"
    ],
    example: "自动微分像导航软件。你告诉它目的地和当前位置，它自动算出每一步该往哪边走，而不是让你手写整条路线的偏导数。",
    questions: ["为什么训练需要反向传播？", "深度学习框架除了写模型，还承担哪些系统工作？"]
  },
  {
    slug: "06-transformer",
    title: "Transformer 基础",
    module: "从 Token 到 Transformer",
    pdf: "llmsys-06-transformer-14bd7575a2f6c8bac60522354c11d691.pdf",
    goals: ["理解 attention 的直觉", "认识 Transformer 的主要组成", "知道为什么上下文长度会影响成本"],
    ideas: [
      "Transformer 的关键是 self-attention：每个 token 都可以查看上下文中其他 token，并分配不同权重。",
      "常见结构包含 embedding、位置编码、attention、前馈网络、残差连接和归一化。",
      "attention 需要比较 token 之间的关系。上下文越长，这部分计算和内存压力通常越大。"
    ],
    example: "读句子“苹果发布新手机，它很贵”时，人会知道“它”多半指手机。attention 就是在计算哪些词互相关联。",
    questions: ["attention 和普通逐词处理有什么区别？", "为什么长上下文会让系统更吃力？"]
  },
  {
    slug: "07-pretrained-llms",
    title: "预训练大语言模型",
    module: "从 Token 到 Transformer",
    pdf: "llmsys-07-llms-acf5db9438a8d9a86f86d29d9c563c00.pdf",
    goals: ["理解预训练的基本目标", "知道参数规模和数据规模的意义", "区分预训练、微调和推理"],
    ideas: [
      "预训练通常使用大量文本，让模型学习语言结构和世界知识的统计规律。",
      "模型规模增加会带来更强能力，也带来更高训练成本、显存需求和服务成本。",
      "预训练得到通用模型，微调让模型适应具体任务，推理则是在用户请求时生成答案。"
    ],
    example: "预训练像通识教育，微调像专业课程，推理像考试现场回答问题。三者目标和资源需求都不同。",
    questions: ["为什么大模型训练成本高？", "微调和推理有什么区别？"]
  },
  {
    slug: "08-tokenization-embedding",
    title: "Tokenization 与 Embedding",
    module: "从 Token 到 Transformer",
    pdf: "llmsys-08-tokenization-594dd043d7a87d8dcc91b7e7585a0e34.pdf",
    goals: ["理解 token 是模型输入单位", "认识 BPE 等分词方法的动机", "理解 embedding 的作用"],
    ideas: [
      "计算机不能直接处理自然语言文本，必须先把文本拆成 token 并映射成编号。",
      "tokenization 需要在词表大小、未知词处理、多语言支持和压缩效率之间取舍。",
      "embedding 把离散编号变成连续向量，后续 Transformer 层才能对它们做矩阵计算。"
    ],
    example: "“internationalization” 可以拆成多个子词。这样模型不用为每个罕见长词都单独准备一个完整词表项。",
    questions: ["token 太细或太粗会分别带来什么问题？", "为什么需要 embedding，而不是直接用编号计算？"]
  },
  {
    slug: "09-decoding-generation",
    title: "生成与解码策略",
    module: "从 Token 到 Transformer",
    pdf: "llmsys-09-decoding-cac2cd9402765ff5e6c24f7baffd321c.pdf",
    goals: ["理解逐 token 生成", "认识 greedy、beam search 和 sampling", "知道解码会影响质量和速度"],
    ideas: [
      "LLM 生成回答时通常一次预测一个 token，再把新 token 接回上下文继续预测。",
      "greedy 选择最可能 token，sampling 引入随机性，beam search 保留多个候选路径。",
      "解码策略影响回答多样性、稳定性、延迟和计算量。服务系统必须在质量和成本之间权衡。"
    ],
    example: "写作文时每次只选最常见的下一个词，文章可能稳但无聊；允许多个候选，表达会更丰富但也更难控制。",
    questions: ["为什么生成是一轮一轮进行的？", "temperature 调高可能带来什么效果？"]
  },
  {
    slug: "10-transformer-acceleration",
    title: "Transformer 训练与推理加速",
    module: "加速核心算子",
    pdf: "llmsys-10-transformer-acc-5ba466406bf7296f86cd244ad0405867.pdf",
    goals: ["认识 Transformer 中的热点算子", "理解训练和推理瓶颈不同", "知道 kernel 优化和融合的意义"],
    ideas: [
      "Transformer 里矩阵乘法、attention、归一化和激活函数都会消耗大量计算和内存带宽。",
      "训练需要保存中间结果用于反向传播，推理更关注逐 token 延迟和 KV cache。",
      "算子融合可以减少中间结果写回内存的次数，从而降低带宽压力。"
    ],
    example: "做菜时如果每切一次菜都洗一次砧板，会浪费时间。算子融合像把能连续做的步骤合并，减少来回搬运。",
    questions: ["训练和推理为什么瓶颈不同？", "算子融合主要减少了什么开销？"]
  },
  {
    slug: "12-jax-xla-tpu",
    title: "JAX、XLA 与 TPU",
    module: "加速核心算子",
    pdf: "llmsys-12-Introduction_to_JAX_XLA_TPU-f0450caf9e7e6707c009f7f77997a2be.pdf",
    goals: ["了解编译器在深度学习系统中的角色", "认识 TPU 的定位", "理解高层代码到硬件执行的路径"],
    ideas: [
      "JAX 提供类似 NumPy 的接口，并通过编译和自动微分把高层数学表达映射到硬件执行。",
      "XLA 是编译器层，尝试优化计算图、融合算子并生成适合目标硬件的代码。",
      "TPU 是面向机器学习工作负载设计的加速器，强调矩阵计算吞吐和大规模训练效率。"
    ],
    example: "写 Python 像写菜谱，XLA 像后厨调度系统，会决定哪些步骤合并、哪些设备执行、怎样减少搬运。",
    questions: ["为什么深度学习系统需要编译器？", "TPU 和 GPU 都是加速器，它们共同目标是什么？"]
  },
  {
    slug: "13-pallas-kernels",
    title: "Pallas Kernel 与 Splash Attention",
    module: "加速核心算子",
    pdf: "llmsys-13-pallas_splash_attention_srinath_mandalapu-b0bc7990950b84561ff9aa8e1791f727.pdf",
    goals: ["理解自定义 kernel 的动机", "认识 tile/block 思想", "知道 attention 优化要贴近硬件"],
    ideas: [
      "通用框架不一定能为每个新模型结构生成最优代码，自定义 kernel 允许开发者控制更底层的数据搬运和并行方式。",
      "tile 思想把大矩阵切成小块，让片上内存可以反复复用局部数据。",
      "Splash Attention 等方法关注如何在特定硬件上更有效地执行 attention。"
    ],
    example: "如果普通公交路线绕路，定制班车可以更快到达。自定义 kernel 就是在重要路线旁边修一条更直接的路。",
    questions: ["什么时候值得写自定义 kernel？", "tile 为什么有助于内存复用？"]
  },
  {
    slug: "14-distributed-training",
    title: "分布式训练基础",
    module: "大模型训练如何扩展",
    pdf: "llmsys-14-distributed-training-b27c3d1dc185e680c6f5cc924e9ec9d7.pdf",
    goals: ["理解为什么需要多卡多机训练", "认识通信开销", "区分计算扩展和系统扩展"],
    ideas: [
      "当模型或数据太大，单张 GPU 放不下或训练太慢，就需要把训练任务分到多张 GPU 上。",
      "分布式训练不只是增加 GPU，还要处理梯度同步、参数一致性、网络通信和故障恢复。",
      "扩展效率取决于计算和通信的比例。如果通信太多，加更多机器可能收益很低。"
    ],
    example: "多人合作写报告能变快，但如果每写一句都要开会确认，沟通成本会吃掉并行收益。",
    questions: ["为什么多卡训练会产生通信开销？", "什么情况下加 GPU 不一定有效？"]
  },
  {
    slug: "15-distributed-data-parallel",
    title: "Distributed Data Parallel",
    module: "大模型训练如何扩展",
    pdf: "llmsys-15-ddp-165dbe3873fac21eb8b339e64bcfee28.pdf",
    goals: ["理解数据并行的基本流程", "知道梯度同步的意义", "认识 all-reduce"],
    ideas: [
      "DDP 中每张 GPU 拥有一份完整模型，但处理不同 mini-batch。反向传播后，各卡同步梯度再更新参数。",
      "all-reduce 是常见同步操作，用来把多张卡的梯度求和或平均并分发回去。",
      "DDP 实现简单、适用广，但当单卡放不下完整模型时就需要其他并行方法。"
    ],
    example: "四个同学各自批改一部分作业，最后汇总平均错误趋势，再一起调整教学方案。这类似数据并行里的梯度同步。",
    questions: ["DDP 为什么每张卡都要有一份模型？", "DDP 的主要限制是什么？"]
  },
  {
    slug: "16-model-parallel-training",
    title: "模型并行训练",
    module: "大模型训练如何扩展",
    pdf: "llmsys-16-model-parallel-83b41612547620ee0e172caa1ee448ed.pdf",
    goals: ["理解模型并行的动机", "区分 tensor parallel 和 pipeline parallel", "认识并行切分带来的通信"],
    ideas: [
      "模型并行把同一个模型拆到多张 GPU 上，用来解决单卡显存放不下的问题。",
      "tensor parallel 通常切分矩阵计算，pipeline parallel 通常按层切分模型阶段。",
      "切得越细，可能越能利用资源，但通信和调度也更复杂。"
    ],
    example: "一条生产线可以按工序分给不同小组，也可以把同一个大工序分给多人同时做。模型并行也有类似切法。",
    questions: ["模型并行和数据并行解决的问题有什么不同？", "pipeline parallel 为什么可能出现空闲等待？"]
  },
  {
    slug: "17-mixture-of-experts",
    title: "Mixture-of-Experts 模型系统",
    module: "大模型训练如何扩展",
    pdf: "llmsys-17-MoE-3aa3125f9ccdd4bb7109ef077fbe9260.pdf",
    goals: ["理解 MoE 的基本思想", "认识路由和负载均衡", "知道稀疏激活为什么能扩展容量"],
    ideas: [
      "MoE 包含多个 expert，但每个 token 通常只激活其中一小部分，因此可以增加模型容量而不等比例增加每次计算量。",
      "router 决定 token 送到哪些 expert。路由质量会影响模型效果和系统负载。",
      "如果很多 token 都挤到同一个 expert，会产生负载不均衡和通信瓶颈。"
    ],
    example: "大型医院有很多专科医生。每个病人不需要见所有医生，只需要被分诊到相关科室。MoE 的 router 类似分诊台。",
    questions: ["MoE 为什么叫稀疏激活？", "路由不均衡会带来什么系统问题？"]
  },
  {
    slug: "18-zero-memory-optimization",
    title: "ZeRO 与分布式训练显存优化",
    module: "大模型训练如何扩展",
    pdf: "llmsys-18-zero-20eb6c8d8c1e7092e1b922abf03d8cdd.pdf",
    goals: ["知道训练显存花在哪里", "理解优化器状态切分", "认识显存和通信的权衡"],
    ideas: [
      "训练时显存不仅存模型参数，还存梯度、优化器状态和中间激活。优化器状态有时会比参数本身还占空间。",
      "ZeRO 把参数、梯度或优化器状态切分到不同设备上，减少每张卡的重复存储。",
      "减少显存通常会增加通信或重计算，系统设计要在容量、速度和复杂度之间取舍。"
    ],
    example: "如果每个组员都打印完整资料，纸张浪费很大。把资料分给不同人保管能省纸，但查资料时需要互相传递。",
    questions: ["训练显存除了参数还包括什么？", "ZeRO 为什么可能增加通信？"]
  },
  {
    slug: "19-llm-quantization",
    title: "LLM 量化基础",
    module: "推理、微调与在线服务",
    pdf: "llmsys-19-quantization-da7a2abad092c802b03672ce1cc7bee9.pdf",
    goals: ["理解量化解决的资源问题", "认识 bit 宽度和误差", "知道量化不等于免费加速"],
    ideas: [
      "量化用更少 bit 表示权重或激活，常见目标是减少显存占用和内存带宽压力。",
      "低 bit 会引入近似误差，需要检查模型质量是否还能接受。",
      "实际速度提升取决于硬件和 kernel 是否支持低 bit 高效计算。"
    ],
    example: "把高清照片压缩成较小文件可以省空间，但压缩太狠会失真。量化也有类似的精度和资源权衡。",
    questions: ["量化主要节省什么资源？", "为什么 4-bit 模型不一定总比 16-bit 快？"]
  },
  {
    slug: "20-gptq-quantization",
    title: "GPTQ 与后训练量化",
    module: "推理、微调与在线服务",
    pdf: "llmsys-20-quantization2-ba573d7e5d82e68027bbd3a92c3cd819.pdf",
    goals: ["理解后训练量化", "知道校准数据的作用", "认识误差补偿思想"],
    ideas: [
      "后训练量化是在模型已经训练好之后进行压缩，目标是不重新大规模训练也能降低资源需求。",
      "GPTQ 关注逐层量化时如何控制误差，让量化后的输出尽量接近原模型。",
      "校准数据用于估计模型在真实输入分布下的数值范围和误差影响。"
    ],
    example: "把一本厚教材缩写成讲义时，不能随机删句子，要保留对理解影响最大的内容。GPTQ 也是在控制压缩误差。",
    questions: ["后训练量化和重新训练有什么区别？", "校准数据为什么重要？"]
  },
  {
    slug: "21-flashattention",
    title: "FlashAttention",
    module: "加速核心算子",
    pdf: "llmsys-21-FlashAttention_tridao2026.4-50476379a6127697ae7fbf974ad28348.pdf",
    goals: ["理解 attention 的内存瓶颈", "认识 IO-aware 优化", "知道 FlashAttention 为什么重要"],
    ideas: [
      "普通 attention 可能生成很大的中间矩阵，尤其在长上下文时会占用大量显存和带宽。",
      "FlashAttention 的核心思路是分块计算并避免把完整 attention 矩阵写回慢速内存。",
      "它体现了系统优化的重要原则：算法不只看浮点计算次数，也要看数据在不同内存层级之间搬了多少次。"
    ],
    example: "做统计时不一定要先保存完整大表，可以边读边累计结果。FlashAttention 也尽量边算边合并，减少中间大表。",
    questions: ["FlashAttention 主要减少什么？", "为什么长上下文更需要 attention 优化？"]
  },
  {
    slug: "22-llm-serving-sglang",
    title: "LLM Serving 与调度",
    module: "推理、微调与在线服务",
    pdf: "llmsys-22-llm-serving-scheduler-radixattention-dfa87a4515092525676277a85bc4425d.pdf",
    goals: ["理解在线推理服务的目标", "认识请求调度和缓存复用", "区分延迟和吞吐"],
    ideas: [
      "LLM serving 面对的是持续到来的用户请求，系统既要单个请求快，也要整体吞吐高。",
      "请求长度、生成长度和到达时间都不同，调度器需要决定哪些请求一起执行。",
      "缓存复用可以减少重复计算，尤其在多个请求共享前缀或上下文时。"
    ],
    example: "奶茶店既要让单个顾客少等，也要让整队顾客流动快。批处理和调度就是在这两个目标之间平衡。",
    questions: ["延迟和吞吐有什么区别？", "为什么相同前缀可以帮助缓存复用？"]
  },
  {
    slug: "23-parameter-efficient-finetuning",
    title: "参数高效微调",
    module: "推理、微调与在线服务",
    pdf: "llmsys-23-peft-791e97c5520e2ea7491153b45a923ac7.pdf",
    goals: ["理解为什么不总是全量微调", "认识 LoRA 等 PEFT 方法", "知道微调成本来自哪里"],
    ideas: [
      "全量微调需要更新所有参数，显存、存储和训练成本都高。",
      "PEFT 方法只训练少量新增参数或低秩适配矩阵，保留大模型主体不变。",
      "这种方法适合多个任务或多个用户定制模型，因为每个任务只需保存较小的适配部分。"
    ],
    example: "不必为每门课重写整本教材，可以给通用教材加一份课程讲义。LoRA 类方法就像给基础模型加小型适配讲义。",
    questions: ["PEFT 为什么省显存和存储？", "PEFT 适合哪些应用场景？"]
  },
  {
    slug: "24-vllm-paged-attention",
    title: "PagedAttention 与 vLLM",
    module: "推理、微调与在线服务",
    pdf: "llmsys-24-vLLM_woosuk_kwon-b6a0750bb310949461ba5a635a1126eb.pdf",
    goals: ["理解 KV cache 管理难点", "认识 PagedAttention 的类比", "知道 vLLM 解决的服务问题"],
    ideas: [
      "推理时每个请求都会产生 KV cache。请求长度不同会导致显存碎片和调度困难。",
      "PagedAttention 借鉴操作系统分页思想，把 KV cache 分块管理，提高显存利用率。",
      "vLLM 通过更好的缓存管理和调度提升服务吞吐，尤其适合多请求并发。"
    ],
    example: "图书馆用固定大小书架格管理书，比给每个人随便占一大片空间更容易避免浪费。PagedAttention 对 KV cache 做类似管理。",
    questions: ["KV cache 为什么会造成显存压力？", "分页思想如何减少碎片？"]
  },
  {
    slug: "26-inference-at-scale",
    title: "大规模推理与系统机会",
    module: "推理、微调与在线服务",
    pdf: "llmsys-26-dynamo-vikram_mailthody-0ddbeb69382d5c168b6d4636a82185d0.pdf",
    goals: ["理解推理规模化的复杂性", "认识 prefill 和 decode 的差异", "知道服务系统要面向真实负载"],
    ideas: [
      "推理请求分为 prefill 和 decode 两类阶段。prefill 处理输入上下文，decode 逐 token 生成输出。",
      "这两个阶段的计算形态不同，适合的调度策略和硬件资源也可能不同。",
      "大规模服务还要面对负载波动、成本控制、实例调度和可靠性。"
    ],
    example: "餐厅备菜和出菜节奏不同。备菜可能一次处理大量材料，出菜则要按订单持续进行。prefill 和 decode 也有类似差异。",
    questions: ["prefill 和 decode 的工作负载有什么不同？", "为什么真实用户流量会让系统设计更难？"]
  },
  {
    slug: "27-lmcache",
    title: "LMCache 与 AI Memory",
    module: "推理、微调与在线服务",
    pdf: "llmsys-27-LMCache_junchenjiang-c21828fe582270cb5e08b4a21a002956.pdf",
    goals: ["理解缓存作为系统抽象", "认识跨请求复用的价值", "知道内存层级会影响服务成本"],
    ideas: [
      "LLM 服务中很多请求可能共享上下文、文档或提示模板。缓存这些中间状态可以减少重复计算。",
      "AI memory 把缓存、存储和调度结合起来，让系统更有效地管理模型运行过程中的状态。",
      "缓存并非越多越好，需要考虑命中率、占用空间、过期策略和一致性。"
    ],
    example: "老师每次讲课都重复写同一段板书很低效。把常用内容保存下来，下次直接调用，就是缓存思想。",
    questions: ["缓存命中率为什么重要？", "缓存太多会带来什么问题？"]
  },
  {
    slug: "28-heterogeneous-serving",
    title: "异构硬件上的 LLM Serving",
    module: "推理、微调与在线服务",
    pdf: "llmsys-28-mooncake-kTransformer-1243bfbecbb0c3610bf95eac030acb2a.pdf",
    goals: ["理解异构硬件的含义", "认识 CPU、GPU、存储的协作", "知道成本和速度的权衡"],
    ideas: [
      "异构服务系统会同时使用不同类型硬件，例如 GPU、CPU、内存、SSD 或专用加速器。",
      "不是所有数据都必须放在最贵最快的设备上。系统可以根据访问频率和速度需求分层放置。",
      "异构设计的难点是数据移动。如果搬运成本太高，便宜硬件也可能拖慢整体服务。"
    ],
    example: "宿舍里常用物品放桌上，不常用物品放柜子，极少用的放仓库。系统也会把数据放在不同速度和成本的层级。",
    questions: ["为什么异构硬件能降低成本？", "数据移动为什么可能成为瓶颈？"]
  },
  {
    slug: "29-disaggregated-prefill-decode",
    title: "Prefill 与 Decode 解耦",
    module: "推理、微调与在线服务",
    pdf: "llmsys-29-disaggregating_prefill_decode_hao_zhang-c0e55139d20512a2348783423397cc7f.pdf",
    goals: ["理解 disaggregation 的动机", "认识阶段化资源分配", "知道系统拆分会引入通信"],
    ideas: [
      "prefill 和 decode 对硬件资源的需求不同。把它们解耦，可以分别调度到更适合的资源池。",
      "这种设计可能提高利用率，但需要在阶段之间传递 KV cache 或其他状态。",
      "解耦系统要评估额外通信成本是否小于调度收益。"
    ],
    example: "快递分拣和末端配送用不同团队更高效，但包裹从分拣中心转到配送站也有运输成本。",
    questions: ["为什么 prefill 和 decode 可以拆开优化？", "拆开后新增的主要成本是什么？"]
  },
  {
    slug: "30-serving-stack",
    title: "应用栈与模型服务",
    module: "推理、微调与在线服务",
    pdf: "llmsys-30-serving-c4a70ab21cde01fb60068a256c6e163a.pdf",
    goals: ["理解模型服务在应用栈中的位置", "认识 Triton 等 serving 组件", "知道工程集成的重要性"],
    ideas: [
      "真实应用不直接面对一个裸模型，而是通过 API 网关、调度器、模型服务、监控和存储系统协同工作。",
      "Triton、LightLLM 等服务框架提供模型加载、批处理、并发管理和硬件后端适配。",
      "工程系统要考虑版本管理、回滚、监控、资源隔离和故障处理，这些都影响用户体验。"
    ],
    example: "校园 App 的问答功能背后不只是一个模型文件，还包括登录、请求转发、模型服务、日志和监控。",
    questions: ["为什么生产环境需要监控和回滚？", "模型服务框架解决了哪些重复工程问题？"]
  }
];

const css = `
  :root { --ink:#172033; --muted:#607089; --line:#d8e0ec; --bg:#f6f8fb; --panel:#fff; --blue:#2563eb; --green:#15803d; --amber:#b45309; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: Inter, "PingFang SC", "Microsoft YaHei", Arial, sans-serif; background: var(--bg); color: var(--ink); letter-spacing: 0; }
  main { max-width: 920px; margin: 0 auto; padding: 28px 18px 48px; }
  header, section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 22px; margin-bottom: 14px; box-shadow: 0 10px 28px rgba(23,32,51,.07); }
  a { color: var(--blue); text-decoration: none; } a:hover { text-decoration: underline; }
  .back { display: inline-flex; margin-bottom: 12px; font-weight: 700; }
  h1 { margin: 0 0 8px; font-size: clamp(28px, 4vw, 42px); line-height: 1.08; }
  h2 { margin: 0 0 12px; font-size: 21px; }
  p, li { line-height: 1.8; font-size: 16px; }
  .meta { color: var(--muted); margin: 0; }
  .tag { display:inline-flex; border-radius:999px; padding:4px 10px; background:#eaf1ff; color:#1d4ed8; font-weight:700; font-size:13px; margin-right:8px; }
  .grid { display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:14px; }
  .callout { border-left: 4px solid var(--green); background:#e9f7ef; padding:12px 14px; border-radius:8px; }
  .questions li::marker { color: var(--amber); font-weight: 800; }
  @media (max-width: 760px) { .grid { grid-template-columns:1fr; } header, section { padding:18px; } }
`;

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderDoc(doc) {
  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${escapeHtml(doc.title)} | LLM 系统新生导学</title>
  <style>${css}</style>
</head>
<body>
  <main>
    <a class="back" href="../freshman_interactive_llm_systems.html">返回互动课堂</a>
    <header>
      <span class="tag">${escapeHtml(doc.module)}</span>
      <h1>${escapeHtml(doc.title)}</h1>
      <p class="meta">这是一篇面向大学新生的导学文档，根据课程课件重新组织。原始 PDF 仍保留在文末，供深入阅读。</p>
    </header>
    <section>
      <h2>学习目标</h2>
      <ul>${doc.goals.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    </section>
    <section>
      <h2>核心解释</h2>
      ${doc.ideas.map(item => `<p>${escapeHtml(item)}</p>`).join("")}
    </section>
    <section>
      <h2>课堂例子</h2>
      <p class="callout">${escapeHtml(doc.example)}</p>
    </section>
    <section>
      <h2>检查问题</h2>
      <ol class="questions">${doc.questions.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ol>
    </section>
    <section>
      <h2>原始资料</h2>
      <p><a href="${pdfBase}/${encodeURI(doc.pdf)}" target="_blank" rel="noopener">打开原始课件 PDF</a></p>
    </section>
  </main>
</body>
</html>
`;
}

await mkdir(outDir, { recursive: true });
await Promise.all(docs.map(doc => writeFile(path.join(outDir, `${doc.slug}.html`), renderDoc(doc), "utf8")));
console.log(`Generated ${docs.length} freshman docs in ${outDir}`);
