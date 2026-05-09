# Speculative Decoding Rollout

> 论文：*Accelerating RL Post-Training Rollouts via System-Integrated Speculative Decoding*  
> 主题：把 speculative decoding 作为 RL post-training 的 rollout 加速原语，集成到 NeMo RL + vLLM、verl 和 slime 等训练系统中，在不改变目标策略采样分布的前提下提升 rollout 吞吐。

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

## 6. verl 中的 MTP / speculative rollout 实践

verl 文档把 MTP 分成两个问题：训练侧如何让 MTP 模块跟上主模型，以及 rollout 侧如何把 MTP 作为推测解码能力交给推理引擎使用。它的重点不是“只要打开 MTP 就一定更快”，而是明确支持范围、训练配置和硬件约束。

### 6.1 支持范围

verl 当前的 MTP RL 训练路径主要面向 mimo-7B-RL、Qwen-next、DeepSeek 系列等 MTP 架构模型。训练后端限制较强：

| 维度 | verl 文档中的约束 |
| --- | --- |
| 训练引擎 | 仅支持 `mbridge/Megatron-Bridge + megatron` 组合 |
| 推理引擎 | 原则上兼容所有引擎，但模型必须在对应推理引擎的兼容列表中 |
| Megatron | 需要支持 MTP + CP 训练的开发版本 |
| SGLang | 文档建议使用修复 MTP tensor 权重更新 OOM 的指定分支或相应 PR |

这意味着在 verl 里做 speculative rollout，第一步不是调 `num_speculative_tokens`，而是确认训练后端、模型 checkpoint、推理引擎和权重更新路径都支持 MTP 参数。

### 6.2 MTP 训练配置

verl 的核心配置都挂在 `actor_rollout_ref.model.mtp` 前缀下。常见场景可以分成三类：

| 场景 | 关键配置 | 语义 |
| --- | --- | --- |
| 只加载 MTP 参数 | `enable=True` | 显存占用增加，导出参数包含 MTP 模块，可用于部署 |
| 全参数 MTP 训练 | `enable=True`, `enable_train=True`, `mtp_loss_scaling_factor=0.1` | MTP loss 会作用到所有模型参数 |
| 只训练 MTP 参数 | `enable=True`, `enable_train=True`, `detach_encoder=True` | 冻结主干 encoder，只更新 MTP 模块 |

文档里的一个关键经验是：推荐优先使用 `detach_encoder=True` 训练 MTP。这样可以让 MTP draft 适配当前 rollout 分布，同时降低 MTP loss 对主策略学习的干扰。

典型命令行形态可以写成：

```bash
actor_rollout_ref.model.mtp.enable=True \
actor_rollout_ref.model.mtp.enable_train=True \
actor_rollout_ref.model.mtp.detach_encoder=True \
actor_rollout_ref.model.mtp.mtp_loss_scaling_factor=0.1
```

如果目标是排查 MTP 配置是否真的生效，要把几种“看起来开了 MTP 但语义不同”的场景拆开看：

- checkpoint 没有 MTP 参数：无法形成有效 MTP draft。
- MTP 参数存在但不训练：短期可用，长期可能随 policy drift 降低 accepted length。
- 设置了 `enable_train=True`，但 `mtp_loss_scaling_factor=0`：等价于不训练 MTP loss。
- `detach_encoder=True`：主要更新 MTP 模块，适合保护主策略；不要把主模型训练曲线变化当成它的主要目标。

### 6.3 Rollout 推理配置

verl 把 rollout 阶段的 MTP 加速也放在 `actor_rollout_ref.model.mtp` 下面，但 vLLM 和 SGLang 的配置不同。

vLLM 路径：

```bash
actor_rollout_ref.model.mtp.enable=True \
actor_rollout_ref.model.mtp.enable_rollout=True \
actor_rollout_ref.model.mtp.method=mtp \
actor_rollout_ref.model.mtp.num_speculative_tokens=1
```

SGLang 路径：

```bash
actor_rollout_ref.model.mtp.enable=True \
actor_rollout_ref.model.mtp.enable_rollout=True \
actor_rollout_ref.model.mtp.speculative_algorithm=EAGLE \
actor_rollout_ref.model.mtp.speculative_num_steps=2 \
actor_rollout_ref.model.mtp.speculative_eagle_topk=2 \
actor_rollout_ref.model.mtp.speculative_num_draft_tokens=4
```

这里的语义和前文一致：MTP 只负责 draft proposal，最终 rollout token 仍由 target model / verifier 验证。因此训练侧的 logprob、reward、advantage 和 policy loss 不应切到 draft policy 上。

### 6.4 性能注意事项

verl 文档给出的实验设置是 `mimo-7B-math`、`max_response_length=8k`。文档提到，启用 MTP 能把 rollout acceptance rate 提高约 14%，但在 H20 上整体吞吐没有提升，甚至可能下降；以 mimo-7B + SGLang 单独部署在 H20 为例，开启 MTP 推测解码后 rollout 吞吐下降约 50%。

这和论文的结论一致：acceptance length 或 acceptance rate 不是最终目标，端到端 step time 才是。MTP 推理收益强依赖模型大小和硬件，尤其依赖 verifier 批量验证是否足够高效。verl 文档列出的 FP16 Tensor Core 性能差异很大，H20 显著低于 H800 / H200，因此较小模型或较弱推理硬件上，MTP draft overhead 可能盖过加速收益。

工程上可以按这个顺序判断是否值得开启：

1. 先打开 stage-level profiling，分开看 `T_prepare`、`T_gen`、`T_logprob` 和 `T_train`。
2. 如果 generation 本身不是瓶颈，不要期望 MTP 改善整体 step time。
3. 如果 acceptance rate 上升但 tokens/s 下降，说明 draft/verify 调度或硬件利用率是瓶颈。
4. 对 H20 这类场景，verl 文档当前更偏向“推理阶段暂不开启 MTP 加速”，等待 rollout speculative 逻辑进一步优化。

## 7. slime 中的在线 MTP draft 实践

slime 的实践更强调 RL 过程中 draft model 的在线更新。它把问题说得很直接：随着 RL 训练推进，target model 会持续变化，如果 draft model 冻结，draft 和 target 的采样概率差异会变大，accepted length 会逐渐下降，speculative decoding 可能从正收益变成负收益。

### 7.1 推理侧开启 speculative rollout

slime 使用 SGLang 作为 rollout 推理后端时，对于带 MTP 层的模型，例如 GLM-4.6、DeepSeek-V3/R1，可以直接打开 EAGLE speculative decoding：

```bash
--sglang-speculative-algorithm EAGLE \
--sglang-speculative-num-steps 3 \
--sglang-speculative-eagle-topk 1 \
--sglang-speculative-num-draft-tokens 4
```

如果使用单独训练出来的 draft model，例如 SpecForge 训练的 draft，还需要指定路径：

```bash
--sglang-speculative-draft-model-path /your/draft/model/path
```

slime 文档当前也明确说明：外部 draft model 的训练仍在 WIP。因此比较成熟的路径是使用模型自带 MTP 层，并在 RL 流程里在线训练它。

### 7.2 Online SFT draft model

Notion 实践文章的核心方案是在 Megatron backend 内部为 MTP 层增加一条 CE loss flow，用 target model 的 hidden state 和 generated tokens 训练 MTP 层，使 draft 跟随当前 target policy。

MTP 训练目标不是普通 AR 的 `input(t) -> output(t+1)`，而更接近 EAGLE MTP 的两步预测：

```text
Input(t) + Input(t+1) -> Output(t+2)
```

假设 target model 生成序列为 `[a, b, c, d, e]`，对应 hidden state 为 `[h(a), h(b), h(c), h(d), h(e)]`，token embedding 为 `[e(a), e(b), e(c), e(d), e(e)]`。MTP 层输入来自两部分：

- target model forward 过程中得到的 hidden state。
- 将生成 token 左移一次后得到的 token embedding。

训练流可以概括为：

```text
target_model_hidden_state = [h(a), h(b), h(c), h(d), h(e)]
rolled_tokens = roll([a, b, c, d, e], shift=-1) = [b, c, d, e, pad]
token_embedding = embed(rolled_tokens) = [e(b), e(c), e(d), e(e), pad]

draft_hidden_state = mtp(concat([token_embedding, target_model_hidden_state]))
mtp_logits = shared_output_layer(draft_hidden_state)

labels = roll(rolled_tokens, shift=-1) = [c, d, e, pad, pad]
mtp_loss = cross_entropy(labels, mtp_logits)
```

Megatron 返回 target model logprobs 后，RL 框架照常计算 GRPO loss；随后一次 `backward()` 同时触发 GRPO loss 和 MTP CE loss 的反向传播。

### 7.3 梯度隔离

slime 实践里的关键安全边界是 detach：

- detach 主模型传给 MTP 的 hidden state。
- detach 主模型和 MTP 共享的 lm head。
- detach 主模型和 MTP 共享的 embedding。

这样做的目的不是让 MTP 完全孤立，而是避免 MTP CE loss 反向污染主策略更新。主模型仍由 GRPO / PPO 等 RL objective 驱动；MTP 只学习更好地预测 target policy 接下来会接受的 token，从而提高 accepted length。

对应 slime 开关是：

```bash
--mtp-num-layers 1 \
--enable-mtp-training \
--mtp-loss-scaling-factor 0.2
```

需要注意的是，MTP 训练要求 checkpoint 本身包含 MTP 权重。因此从 Hugging Face checkpoint 转成 torch dist 时，也要带上：

```bash
--mtp-num-layers 1
```

### 7.4 实验现象

slime / Notion 实践使用 H200 集群、Mimo-7B-RL、DAPO-Math-17k、`max_response_length=24k`，并设置 `mtp_loss_scaling_factor=0.35`、attention backend 为 `fa3`。对比组包括：

| 设置 | 含义 |
| --- | --- |
| 训练 MTP + speculative decoding | 启用 MTP 层推测解码，并在 RL 中训练 MTP |
| 冻结 MTP + speculative decoding | 启用 MTP 推理，但不训练 MTP |
| 无 speculative decoding | 普通 autoregressive rollout |

主要现象：

- 开启 MTP 训练后，accepted length 稳步上涨，MTP loss 下降。
- 相比不开 speculative decoding，训练 MTP 的 speculative rollout 整体约有 35% 性能提升。
- 相比冻结 MTP，训练 MTP 整体约有 14% 提升，训练后期差距可扩大到约 25%。
- 额外训练 MTP layer 会增加少量训练成本，但相对采样节省的时间，整体仍然有收益。
- 训练效果符合理论预期：speculative decoding 不改变 target model 的采样分布，因此不应改变主模型训练效果。

这套实践补上了论文中 “draft coherence” 的工程闭环：draft 不只是初始化时贴近 policy，而是在 RL 过程中持续跟随 policy。

## 8. 对 RL Infra 的启示

这篇文章可以理解为一个系统设计提醒：RL post-training 里的 rollout 加速不能只看推理吞吐，还必须维护训练语义。

落到工程实现，至少要检查以下问题：

1. **rollout 分布是否仍来自 verifier policy**：draft 不能进入 policy loss 的定义。
2. **logprob / KL / advantage 是否使用正确 policy**：所有训练统计都应基于 target policy。
3. **权重同步是否可观测**：需要知道 rollout backend 使用的是哪个 policy 版本。
4. **draft 是否和当前任务分布匹配**：in-domain draft 初始化通常比通用 chat draft 更重要。
5. **是否有 stage-level telemetry**：只看 tokens/s 不够，要分开看 prepare、generation、logprob、training。
6. **draft length 是否按端到端时间调参**：acceptance length 高不代表 step time 低。
7. **异步训练下要看 exposed generation time**：如果 generation 已被 pipeline overlap 隐藏，speculative decoding 的边际收益会变小。
8. **MTP 训练是否和主策略梯度隔离**：在线训练 draft 时，要清楚哪些 hidden state、lm head、embedding 被 detach，避免 draft loss 改写 policy objective。
9. **checkpoint 是否真的包含 MTP 权重**：只打开配置但 checkpoint 没有 MTP 参数，rollout 推测路径不会得到预期收益。
10. **硬件是否适合推测解码**：小模型、弱 Tensor Core、低 batch 或验证调度开销高时，acceptance rate 上升也可能换来吞吐下降。

## 9. 核心 takeaways

1. RL post-training 的瓶颈正在明显转向 rollout generation，尤其是 reasoning 和 agentic 场景。
2. Speculative decoding 是少数可以提升 rollout throughput、同时保持 target policy 采样分布的加速方式。
3. 真正困难的是系统集成：权重同步、draft coherence、verifier-exact logprob、在线 draft 更新和异步流水线组合。
4. Draft 质量要按 RL rollout distribution 评估，而不是按通用聊天能力评估。
5. Acceptance length 不是最终指标，端到端 RL step time 才是。
6. verl 的 MTP 实践说明：MTP 参数、训练后端、推理引擎、权重同步和硬件条件都满足时，才值得打开 speculative rollout。
7. slime 的在线 MTP 训练说明：冻结 draft 会随 policy 漂移而退化，在线 SFT draft 能长期维持 accepted length。
8. 在大模型、长输出、高 generation share 场景中，speculative decoding 有机会成为 RL infra 的基础 rollout primitive。

## 10. 参考文献

* [Better & Faster Large Language Models via Multi-token Prediction(MTP)](https://arxiv.org/abs/2404.19737)
* [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840)
* [Accelerating RL Post-Training Rollouts via System-Integrated Speculative Decoding(MTP in rollout)](https://arxiv.org/pdf/2604.26779)
* [verl: 在 SFT/RL 训练和推理中使用 MTP 指南](https://verl.org.cn/en/latest/advance/mtp.html)
* [slime: 投机采样](https://thudm.github.io/slime/zh/advanced/speculative-decoding.html)
* [Power Up Speculative Decoding In Reinforcement Learning](https://www.notion.so/jiajunli-guapisolo/Power-Up-Speculative-Decoding-In-Reinforcement-Learning-2a92d24a293b802d9c73dbae429e581e)
