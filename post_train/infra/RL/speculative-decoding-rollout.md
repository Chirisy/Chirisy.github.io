# Speculative Decoding Rollout

> 论文：*Accelerating RL Post-Training Rollouts via System-Integrated Speculative Decoding*  
> 主题：把 speculative decoding 作为 RL post-training 的 rollout 加速原语，集成到 NeMo RL + vLLM 中，在不改变目标策略采样分布的前提下提升 rollout 吞吐。

## 1. 一句话总结

这篇文章的核心观点是：RL post-training 的主要瓶颈正在从梯度更新转向 autoregressive rollout generation，而 speculative decoding 可以作为一种 **lossless / verifier-exact** 的系统加速手段，让 rollout 更快，同时保持轨迹仍然来自目标策略本身；难点不在算法公式，而在把 draft、verifier、权重同步、logprob 计算、异步流水线和训练语义正确接起来。

## 2. 问题背景

LLM 的 RL post-training 通常需要在线生成 rollout，再基于这些轨迹计算 reward、advantage、logprob 和 policy loss。对于 reasoning RL、数学任务、代码任务和 agentic RL 来说，rollout 往往很长，且需要反复调用当前 policy 生成 token，因此训练耗时越来越多地被生成阶段占据。

论文把一个同步 RL step 拆成：

```text
T_step = T_data + T_prepare + T_gen + T_logprob + T_train
```

其中：

- `T_prepare`：权重同步、rollout backend 准备等。
- `T_gen`：rollout generation，主要是自回归 decode。
- `T_logprob`：用当前 policy 重新计算轨迹 logprob。
- `T_train`：advantage 计算和 policy optimization。

在论文的 8B reasoning workload 实验中，generation 占 RL step 总时间的约 65% 到 72%，是最大单项开销。也就是说，即使 policy update 本身已经优化得很好，训练速度仍会被 rollout engine 卡住。

已有 rollout efficiency 方法大多通过改变训练或采样语义来换吞吐：

| 方法 | 加速来源 | 代价 |
| --- | --- | --- |
| 异步 RL | generation、logprob、training 互相 overlap | 引入 policy lag，rollout 可能来自旧策略 |
| off-policy replay | 重用旧轨迹 | 需要 importance correction，可能降低 on-policy 有效性 |
| 低精度 rollout | 用 FP8 等低精度降低推理成本 | 可能带来 distribution mismatch |
| prompt filtering | 跳过低价值 prompt | 改变采样覆盖范围 |

这些方法都有效，但它们都会以某种方式改变原始 on-policy 问题。RL post-training 的学习信号依赖 policy 自己采样出来的轨迹，如果 rollout 分布偏了，吞吐提升不一定等价于学习速度提升。

## 3. 动机

论文用一个简单分解描述 RL 训练速度：

```text
learning speed = effectiveness * throughput
```

其中 `throughput` 是系统每单位时间能完成多少 rollout 和训练工作，`effectiveness` 是这些工作带来多少有效学习信号。很多系统优化会提高 throughput，但可能损伤 effectiveness。

Speculative decoding 的吸引力在于，它理论上只提升 throughput，不改变 target policy 的输出分布。它让 draft model 一次提出多个候选 token，再由 target / verifier model 做验证和拒绝采样。只要 rejection procedure 正确，最终接受的 token 序列仍服从 verifier policy 的分布。

因此它特别适合 RL rollout：

1. **保持 on-policy 语义**：轨迹仍来自当前目标策略，不需要把 draft 当成训练策略。
2. **不改变 reward 和 loss 定义**：reward、KL、logprob、GRPO loss 仍基于 verifier policy。
3. **与异步 RL 互补**：异步训练隐藏一部分生成延迟，speculative decoding 降低每次生成本身的成本。
4. **适合长输出场景**：reasoning trace、tool-use trajectory、web agent 多轮交互都会放大 decode 成本。

但 speculative decoding 不能无条件带来端到端收益。它只加速 `T_gen` 中的 decode 部分，不加速 prefill、logprob recomputation 或 policy training。论文给出的上界可以理解为 Amdahl 定律：

```text
S_step <= 1 / (R_gen / alpha + (1 - R_gen))
```

这里 `R_gen = T_gen / T_step` 表示 generation 在 step 中的占比，`alpha` 是平均每次 speculation step 能接受的 token 数。直觉是：generation 占比越高、平均接受长度越高，收益越大；draft overhead、prefill、batching 和验证开销都会让实际收益低于理论上界。

## 4. 方法内容

### 4.1 系统集成思路

论文不是提出新的 RL objective，而是把 speculative decoding 集成成 RL infra 里的 rollout primitive。系统由三部分组成：

| 组件 | 作用 |
| --- | --- |
| vLLM rollout backend | 用 speculative decoding 生成 rollout trajectory |
| MegatronLM policy / verifier | 执行用于 GRPO loss 的 verifier forward pass |
| draft mechanism | 提前提出候选 token，可来自 EAGLE-3 外部 drafter 或模型内置 MTP heads |

关键约束是：**训练语义必须始终以 verifier policy 为准**。draft 只负责提案，不负责定义 logprob、KL 或 policy loss。否则 speculative decoding 就不再是 lossless 加速，而会改变优化目标。

### 4.2 Draft 路径

论文支持两类 draft：

| Draft 路径 | 适用场景 | 特点 |
| --- | --- | --- |
| EAGLE-3 | 任意 pretrained model | 通用性强，但需要维护一个外部 drafter |
| Native MTP heads | 模型自带 multi-token prediction heads | 系统更直接，draft 由模型内置辅助头承担 |

文章主要实验选择 EAGLE-3，因为这是更困难的情况：draft 不是模型原生能力，需要初始化、同步、可选在线更新，并在 RL 过程中持续贴近不断变化的 policy。

### 4.3 权重同步与 draft coherence

普通推理服务中的模型权重是固定的，而 RL training 中 policy 每一步都会更新。因此系统必须处理两个同步问题：

1. **Verifier 权重同步**：learner 更新后的 policy 权重要同步到 vLLM rollout backend，保证 rollout 来自当前或目标 policy。
2. **Draft 对齐**：draft model 要尽量贴近当前 policy 的 rollout 分布，否则 acceptance length 会下降，甚至被 overhead 吃掉收益。

论文强调，draft 的质量不是泛化聊天能力，而是和当前 RL rollout distribution 的匹配程度。实验里，用 DAPO 数学 post-training 数据初始化的 draft，比 UltraChat / Magpie 这类通用聊天数据初始化的 draft 更快。

### 4.4 在线 draft adaptation

论文提供可选的 online draft adaptation。做法是复用 MegatronLM 在计算 GRPO loss 时已经产生的 hidden states 和 verifier logprobs，把它们作为 EAGLE-3 draft supervision，而不是额外再跑一次 policy forward。

系统上有一个重要边界：

```text
GRPO loss path:      policy forward -> policy gradient
SpecDec loss path:   cached hidden states/logprobs -> detach -> draft loss
```

`.detach()` 的意义是让 draft training 不干扰 policy gradient。draft 可以学习当前 policy 的 token 分布，但不能通过 draft loss 反向影响 policy update。

实验结论是：如果 draft 已经用 in-domain 数据初始化得很好，online adaptation 额外收益很小；如果 draft 初始化较弱，在线更新能作为对 distribution mismatch 的保险。

### 4.5 同步与异步 RL 的组合

在同步 RL 中，rollout generation 位于 step 的关键路径上，speculative decoding 直接降低 `T_gen`，收益更容易显现。

在异步 RL 中，generation 可以和 logprob / training overlap，因此 exposed generation time 变小，speculative decoding 的端到端加速会被削弱。但两者仍然互补：

- async RL：隐藏一部分 generation 成本。
- speculative decoding：降低每次 generation 本身的成本。

论文在 16 节点异步设置中观察到，speculative decoding 把 training-side idle generation time 从 10.4s 降到 0.6s，把有效 step time 从 75.0s 降到 60.5s，约 1.24x。

## 5. 主要实验结论

实验使用 Qwen3-8B / Qwen3-8B-Base，在 DAPO-Math-17K 上做 GRPO post-training，并用 AIME-2024 验证 accuracy。

### 5.1 端到端收益

| 设置 | AR generation | Spec generation | Generation speedup | Overall step speedup |
| --- | ---: | ---: | ---: | ---: |
| RL-Zero | 100.0s | 56.6s | 1.77x | 1.41x |
| RL-Think | 133.6s | 87.0s | 1.54x | 1.35x |

验证 accuracy 曲线与 autoregressive baseline 基本重合，说明 speculative decoding 没有改变优化轨迹，只是让 rollout 更快。

### 5.2 n-gram drafting 不一定有收益

论文也比较了 `n-gram` draft。虽然它有非零 acceptance length，但在两个任务上都比 autoregressive decoding 更慢。这说明 speculative decoding 的关键不是“能接受几个 token”，而是：

- draft proposal overhead 是否足够低；
- verifier batching 是否高效；
- acceptance length 是否足以覆盖额外验证成本；
- draft 是否贴近当前 RL 分布。

### 5.3 Draft length 的权衡

实验中 `k = 3` 最好。更长的 draft length 会提高 acceptance length，但反而降低实际 speedup。RL-Think 中 `k = 5` 和 `k = 7` 甚至慢于 AR baseline。

这给 RL infra 的启示是：不能只看 acceptance length 调参，要看端到端 latency。更长 draft 会增加 speculative work、验证压力和调度开销，可能让系统吞吐变差。

### 5.4 Deployment scale projection

论文用高保真 simulator 估算大规模部署收益。对 Qwen3-235B-A22B，在有利配置下：

- rollout speedup 可超过 3x；
- projected end-to-end training speedup 约 2.5x；
- 收益受 model scale、GPU count、local batch、policy lag、draft length、acceptance length 共同影响。

更大的模型和更长的 rollout 通常更容易从 speculative decoding 中获益，因为 generation share 更高，decode-heavy 特征更明显。

## 6. 对 RL Infra 的启示

这篇文章可以理解为一个系统设计提醒：RL post-training 里的 rollout 加速不能只看推理吞吐，还必须维护训练语义。

落到工程实现，至少要检查以下问题：

1. **rollout 分布是否仍来自 verifier policy**：draft 不能进入 policy loss 的定义。
2. **logprob / KL / advantage 是否使用正确 policy**：所有训练统计都应基于 target policy。
3. **权重同步是否可观测**：需要知道 rollout backend 使用的是哪个 policy 版本。
4. **draft 是否和当前任务分布匹配**：in-domain draft 初始化通常比通用 chat draft 更重要。
5. **是否有 stage-level telemetry**：只看 tokens/s 不够，要分开看 prepare、generation、logprob、training。
6. **draft length 是否按端到端时间调参**：acceptance length 高不代表 step time 低。
7. **异步训练下要看 exposed generation time**：如果 generation 已被 pipeline overlap 隐藏，speculative decoding 的边际收益会变小。

## 7. 核心 takeaways

1. RL post-training 的瓶颈正在明显转向 rollout generation，尤其是 reasoning 和 agentic 场景。
2. Speculative decoding 是少数可以提升 rollout throughput、同时保持 target policy 采样分布的加速方式。
3. 真正困难的是系统集成：权重同步、draft coherence、verifier-exact logprob、在线 draft 更新和异步流水线组合。
4. Draft 质量要按 RL rollout distribution 评估，而不是按通用聊天能力评估。
5. Acceptance length 不是最终指标，端到端 RL step time 才是。
6. 在大模型、长输出、高 generation share 场景中，speculative decoding 有机会成为 RL infra 的基础 rollout primitive。

## 8. 参考文献

* [Better & Faster Large Language Models via Multi-token Prediction(MTP)](https://arxiv.org/abs/2404.19737)
* [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840)
* [Accelerating RL Post-Training Rollouts via System-Integrated Speculative Decoding(MTP in rollout)](https://arxiv.org/pdf/2604.26779)