# OPD 系统综述：从 On-Policy Distillation 到多专家能力整合

> 目标：Thinking Machines Lab 博客、DeepSeek-V4 技术报告、MiMo-V2-Flash 技术报告以及 `Rethinking On-Policy Distillation`，整理 OPD 作为后训练系统的核心机制、工程实现、工业案例和局限。

## 1. 一句话结论

OPD（On-Policy Distillation，在线策略蒸馏）不是简单把 teacher 的答案拿来做 SFT，而是让 student 先按自己的当前策略生成轨迹，再让 teacher 在 student 实际访问的状态上给逐 token 的概率反馈。它把 RL 的 on-policy 状态分布和蒸馏的 dense supervision 结合起来，因此特别适合做两类事情：

- 把一个强 teacher 的最终策略高效迁移给 student。
- 把多个领域专家（数学、代码、Agent、指令跟随、安全等）融合到一个统一模型里。

但 OPD 不是无条件有效。`Rethinking OPD` 的核心提醒是：teacher 更强不等于更可蒸馏。OPD 成功通常需要 teacher 和 student 的思考模式足够兼容，并且 teacher 提供 student 尚未学过的新能力。

## 2. 背景：为什么需要 OPD

大模型后训练常见路径是：

- `Pre-training`：建立语言、世界知识和通用推理的基础能力。
- `SFT / Mid-training`：补充领域数据和指令格式。
- `RL / RLHF / RLVR`：用结果奖励或偏好奖励塑造目标行为。
- `Distillation`：从更强模型或专家模型迁移能力。

问题在于，多领域后训练容易出现“跷跷板效应”：数学 RL 变强可能伤害写作，代码 Agent 训练可能影响知识问答，安全对齐可能牺牲部分任务性能。传统顺序训练或权重合并很难稳定保留每个专家的峰值能力。

OPD 的核心价值在这里：它允许每个领域先独立训练出强专家，然后通过 teacher 在 student rollout 上给 dense token-level 信号，把多个专家的行为压缩进一个统一 student，而不是直接混权重或混静态样本。

## 3. OPD 与 SFT、RL 的定位

| 方法 | 采样分布 | 反馈密度 | 主要信号 | 优点 | 典型问题 |
| --- | --- | --- | --- | --- | --- |
| SFT / off-policy distillation | teacher 或固定数据 | dense | 标准答案 token | 简单、稳定、便宜 | exposure bias，容易学 teacher 风格而非能力，长轨迹误差累积 |
| RL / RLVR | student | sparse | 结果奖励或偏好奖励 | 能探索新策略，直接优化任务目标 | 信号稀疏，credit assignment 难，rollout 成本高 |
| OPD | student | dense | teacher 对 student token 的 logprob / logits | 状态分布 on-policy，反馈逐 token，训练效率高 | 依赖 teacher-student 支持重叠，长轨迹 teacher 信号会退化 |

Thinking Machines Lab 的博客把 OPD 描述为“RL 的 on-policy 相关性 + 蒸馏的 dense reward”。类比成下棋：RL 是自己下完一盘只知道输赢，off-policy SFT 是看大师棋谱，OPD 是自己每走一步都由大师点评。

## 4. 基本算法形式

给定 prompt `x`，student `pi_theta` 自回归采样轨迹：

```text
y ~ pi_theta(. | x)
```

在每个 student 访问的前缀 `(x, y_<t)` 上，计算 student 和 teacher 的 next-token 分布：

```text
p_t(v) = pi_theta(v | x, y_<t)
q_t(v) = pi_teacher(v | x, y_<t)
```

常见 reverse KL 目标：

```text
L_OPD = E_{x, y ~ pi_theta} sum_t KL(p_t || q_t)
```

sampled-token 版本只看 student 实际采样出的 token `y_t`：

```text
reverse_kl_t = log pi_theta(y_t | x, y_<t) - log pi_teacher(y_t | x, y_<t)
advantage_t = -reverse_kl_t
```

然后复用 RL 的 importance-sampling / policy-gradient 风格 loss 更新 student。Thinking Machines 的实现思路基本是：

```text
1. student rollout，记录 sampled token 和 student logprob
2. teacher 对同一条 rollout 计算 teacher logprob
3. reward / advantage = teacher logprob - student logprob
4. 用 RL 训练框架更新 student
```

这个形式的工程优势是明显的：teacher 只需要 forward，不需要反向传播；reward 不必等整条回答结束；可以做 partial rollout；可以直接嵌入已有 GRPO / PPO 类训练框架。

## 5. 三种 OPD 粒度

### 5.1 Sampled-token OPD

只对 student 已采样 token 计算 teacher logprob。优点是成本最低，teacher forward 后只取轨迹 token 的 logprob。缺点是梯度方差大，不能利用完整 logits 分布。

适合：

- 小规模验证。
- teacher 成本高、工程预算有限。
- 已有 RL 框架快速改造。

### 5.2 Top-k OPD

只在 student 或 teacher 的 top-k token 集合上做 KL。它介于 sampled-token 和 full-vocabulary 之间。相比 sampled-token，top-k 能给更丰富的 token-level 对齐信号；相比 full-vocabulary，成本更可控。

适合：

- teacher-student overlap 需要显式监控。
- 想减少 full logits 内存压力。
- 需要在稳定性和成本之间折中。

### 5.3 Full-vocabulary OPD

对整个词表计算 KL。DeepSeek-V4 技术报告明确采用 full-vocabulary logit distillation，因为 sampled-token KL 方差高、容易训练不稳定。完整 logits 能提供更稳定的梯度和更忠实的 teacher 分布。

代价是系统复杂度显著上升：

- 长序列下 logits 体积巨大。
- 多 teacher 下 teacher 权重和 prediction head 调度困难。
- 需要专门的缓存、offload、kernel 和 batch 排序策略。

## 6. 为什么 reverse KL 适合多专家融合

OPD 通常使用 reverse KL：`KL(student || teacher)`。它的直觉是：student 在自己会生成的区域里向 teacher 靠近，而不是强迫 student 覆盖 teacher 的所有 mode。

这对多专家整合很关键：

- 专家模型可以在各自领域充分特化，不必担心自身对其他领域遗忘。
- 统一 student 只在当前任务相关的 student trajectory 上吸收对应 teacher 的局部行为。
- 相比 forward KL / SFT，reverse KL 更 mode-seeking，较不容易把 teacher 的无关风格、偏置或局部分布噪声全盘复制进 student。

知乎文章的一个有用视角是：OPD 像“先自己做题，再拿自己的解题轨迹问老师”。如果只是抄老师答案，就是 forward KL；如果先形成自己的尝试，再让专家纠正，就更接近 OPD。这个类比的重点不是把人类学习神秘化，而是强调：没有 student rollout，就没有真正针对当前能力边界的反馈。

## 7. DeepSeek-V4：把 OPD 作为多专家融合主线

DeepSeek-V4 技术报告的后训练路线可以概括为两阶段：

- 阶段一：按领域训练专家。数学、代码、Agent、指令跟随等专家分别经过 SFT 和 GRPO 类 RL，形成各自领域的强 teacher。
- 阶段二：用 multi-teacher OPD 把专家合入一个统一模型。V4 报告称混合 RL 整合阶段被 OPD 替代，OPD 成为最终统一模型的主要能力融合方式。

DeepSeek-V4 的关键设计点：

- 多 teacher：超过 10 个不同领域 teacher 参与蒸馏。
- 任务路由：根据 prompt 所属领域对齐到相应 expert teacher，例如数学题对齐数学 teacher，编程任务对齐代码 teacher。
- reverse KL：统一 student 在自己的 on-policy trajectory 上对齐 expert 分布。
- full-vocabulary logit distillation：报告认为 sampled-token 估计方差较大，full logits 更稳定。
- 工程目标：在百万 token 上下文和多 teacher 场景下可扩展运行 OPD。

DeepSeek-V4 的系统实现尤其值得关注：

- FP4 量化：rollout、teacher forward、reference forward 等 inference-only 路径使用 FP4，降低内存流量和采样延迟。
- teacher 权重 offload：teacher 权重放在中心化分布式存储，按需加载，并用 ZeRO-like sharding 缓解 I/O 和 DRAM 压力。
- hidden state 缓存：不直接物化所有 teacher 的 full logits，而是缓存最后一层 hidden states，训练时再接对应 prediction head 重建 logits。
- teacher head 调度：按 teacher index 对样本排序，让每个 mini-batch 尽量只加载一个 teacher head，减少 GPU 显存占用。
- 专用 KL kernel：用 TileLang kernel 计算 exact KL，减少动态内存分配和 kernel overhead。
- token-granular WAL：rollout 服务支持抢占和故障恢复，每生成一个 token 就写 WAL，避免中断后重采样导致长度偏差。
- 长上下文数据格式：把 rollout 数据拆成轻量 metadata 和重 per-token fields，metadata 做全局 shuffle 和 packing，重字段通过 shared-memory loader 按 mini-batch 读取和释放。
- Agent sandbox：为 Agent 后训练和评测准备统一 sandbox，支持函数调用、容器、microVM、fullVM 等执行 substrate。

DeepSeek-V4 的意义在于：它把 OPD 从“一个 loss”推进成“发布级后训练基础设施”。多专家 OPD 的瓶颈不在公式，而在 teacher scheduling、logits/hidden state 存储、rollout 容错、长上下文 I/O 和 Agent 环境管理。

## 8. MiMo-V2-Flash：MOPD 与 RL/ORM 混合

MiMo-V2-Flash 技术报告提出 MOPD（Multi-Teacher On-Policy Distillation），同样把多 teacher OPD 作为后训练核心范式。

MiMo-V2 的三阶段 pipeline：

- 阶段一：通用 SFT，建立指令跟随和基础 assistant 行为。
- 阶段二：领域专家训练，通过专门的 RL / SFT 训练数学、代码、搜索 Agent、通用工具、安全、推理等 teacher。
- 阶段三：MOPD，student 从自己的 rollout 分布采样，并在 token 级别接收领域 teacher 的 reverse KL 信号。

MiMo-V2 的公式上更明确地把 OPD 写成 RL surrogate loss。teacher-student reverse KL 给出 token-level advantage，同时可以叠加 outcome reward model（ORM）的序列级 advantage：

```text
A_t = log pi_teacher(y_t | x, y_<t) - log pi_theta(y_t | x, y_<t) + alpha * A_ORM
```

这个设计表达了一个重要方向：OPD 不必替代 RL。更实际的系统可能是“dense teacher reward + sparse outcome reward”的混合：

- teacher reward 负责局部过程对齐，降低 credit assignment 难度。
- ORM / verifier reward 负责最终正确性，避免只模仿 teacher 的局部偏好。
- 对 Agent、代码、数学这类长链任务，混合信号通常更稳健。

MiMo-V2 的工程侧也有几个关键点：

- SGLang 作为 inference engine，Megatron-LM 作为 training engine。
- FP8 用于训练和推理。
- Rollout Routing Replay（R3）解决 MoE rollout 和 training 之间 expert routing 不一致问题。
- request-level prefix cache 在多轮 Agent rollout 中复用 KV cache 和 MoE routed experts，同时避免跨请求输出 cache 共享带来的采样不一致。
- Data Scheduler 按细粒度 sequence 调度，而不是只调度 micro-batch，降低长尾 rollout 造成的 GPU 空转。
- partial rollout 与 staleness-aware truncated importance sampling 用于加速长轨迹训练，同时控制策略陈旧度。
- Toolbox / Tool Manager 用 Ray 管理工具资源、QPS、环境预热、异步 reward computation、timeout recovery 和监控。
- MTP 被复用为推测解码 draft model，在推理和 rollout 中提升吞吐。

MiMo-V2 的意义在于：它把 MOPD 放进完整的 Agent/RL 基础设施中，并强调 teacher-student 的持续共进化。新一代 student 可以再次进入领域 RL 阶段产生更强 teacher，再回到 MOPD 融合。

## 9. Thinking Machines Lab：OPD 的实用配方

Thinking Machines Lab 的博客更偏工程实践和成本分析，给出几个可操作结论：

- OPD 可以直接复用 RL 框架，把 KL regularizer/reference model 换成 teacher model。
- teacher 只需要 `compute_logprobs`，不用训练 reward model。
- 如果 student 已经经过 SFT / mid-training，teacher 行为更可能落在 student 的 support 内，OPD 更容易有效。
- 相比继续 SFT 或 RL，OPD 在部分实验中显著减少训练成本，因为每条 trajectory 提供 `O(N)` token-level 信息，而 RL 的结果奖励更接近 `O(1)` episode-level 信息。
- OPD 适合 continual learning：先用 SFT / mid-training 注入新知识，再用旧模型或强 teacher 做 OPD 恢复指令跟随等已学行为。
- RL 更像搜索新语义策略，OPD 更像把已经搜索到的最终策略快速教给 student。

这个视角能帮助区分 OPD 和 RL 的边界：

- 如果你还没有强 teacher，也不知道好策略在哪里，仍然需要 RL / search。
- 如果强 teacher 已经存在，OPD 是更便宜的策略压缩和能力迁移路径。
- 如果 student 完全没有 teacher 所需 token/support，先做 SFT cold start 往往必要。

## 10. Rethinking OPD：OPD 为什么会失败

`Rethinking On-Policy Distillation` 系统研究了 OPD 的成功/失败机制。它的核心结论可以压缩成三句话：

- OPD 成功需要 teacher 和 student 的 thinking pattern 兼容。
- teacher 分数更高不代表能给 student 新信息。
- 成功 OPD 的有效梯度集中在 student 和 teacher 高概率 token 的重叠区域。

### 10.1 Thinking-pattern consistency

OPD 在 student 访问的状态上学习。如果 teacher 和 student 在这些状态下的 top-k token 分布几乎不重叠，teacher 的局部信号就很难转化成有效梯度。

这解释了一个反直觉现象：更强 teacher 可能蒸馏失败，较弱但同源、同思考模式的 teacher 反而有效。早期的 thinking-pattern mismatch 会造成收益损失，并且后续训练不一定能补回来。

### 10.2 Higher scores 不等于 new knowledge

即使 teacher 分数更高、和 student 同源，也可能没有带来 student 没见过的新知识。论文的 reverse distillation 实验表明，同 family 的 1.5B 和 7B 模型可能在 student 视角下分布差异很小，导致 OPD 没有足够的额外信息增益。

所以选 teacher 时不能只看 benchmark：

- 要看 teacher 是否经过额外 RL / 数据 / 工具训练。
- 要看 teacher 是否包含 student 未见过的能力。
- 要看 teacher 与 student 在目标 prompt 上的 top-k overlap 和 entropy gap。

### 10.3 高概率 overlap token 是主要学习载体

论文发现，成功 OPD 表现为 student 和 teacher 在 student-visited states 上逐渐对齐高概率 token。一个很小的共享 token 集合会承载大部分概率质量，实验中 overlap token 可覆盖约 97% 到 99% 的概率质量。

这说明 OPD 的有效学习不是平均发生在整个词表，而是集中在少数“分叉 token”和高概率局部决策上。训练监控时应重点看：

- `overlap ratio`：student top-k 与 teacher top-k 的交集比例。
- `overlap mass`：交集 token 覆盖的概率质量。
- `overlap-token advantage`：重叠 token 上 teacher 相对 student 的优势。
- `entropy gap`：teacher 和 student 的不确定性差异。

### 10.4 两个失败恢复方法

`Rethinking OPD` 给出两个 practical recipe：

- Off-policy cold start：先用 teacher 生成数据做 SFT，把 student 拉近 teacher 分布，再切到 OPD。
- Teacher-aligned prompt selection：使用更贴近 teacher post-training 分布的 prompt，提高 student rollout 与 teacher 分布的局部兼容性。

但第二种方法有风险：如果 prompt 过于 teacher-aligned，student entropy 可能过低，探索能力塌缩。因此需要混入 OOD prompts 保持多样性。

### 10.5 长轨迹限制

论文还指出，OPD 的 dense token-level reward 并非免费午餐。随着 trajectory depth 增加，teacher 对 student 长前缀的局部 reward 质量会系统性退化，后段 token 的不稳定会向前传播。这对长链推理和 Agent 任务尤其关键。

工程上更稳妥的方向可能是：

- 短段用 dense OPD。
- 长程用 sparse outcome reward / verifier。
- 对长轨迹做 curriculum，逐步拉长 OPD supervised horizon。
- 使用 partial rollout 和 segment-level teacher scoring，避免 teacher 在过长错误前缀上提供低可靠局部信号。

## 11. OPD 系统架构

一个可扩展 OPD 系统不只是 loss function，至少包含以下模块：

| 模块 | 作用 | 关键设计 |
| --- | --- | --- |
| Prompt/Data Scheduler | 选择训练 prompt，控制领域配比、长度、难度、温度 | teacher-aligned 与 OOD 混合，领域 quota，pass-rate aware sampling |
| Student Rollout Engine | 生成 on-policy trajectory | batch packing，partial rollout，prefix cache，deterministic sampling |
| Teacher Service | 在 student trajectory 上计算 logprob / logits | 多 teacher 路由，权重 offload，teacher head 调度，异步 forward |
| Reward/Advantage Builder | 计算 reverse KL、ORM advantage、importance weights | sampled-token / top-k / full-vocab，clipping，normalization |
| Training Engine | 根据 advantage 更新 student | GRPO/PPO/importance-sampling loss，MoE routing replay，mixed precision |
| Storage & Cache | 存轨迹、logprob、hidden state、metadata | WAL，hidden state cache，shared memory loader，KV cache 保存 |
| Fault Tolerance | 应对抢占和硬件故障 | token-level WAL，checkpoint，rollout resume |
| Eval & Monitoring | 判断 OPD 是否有效 | task metrics，overlap ratio，entropy gap，KL，length bias，teacher-student gap recovery |
| Agent Environment | 为工具使用和代码任务提供执行反馈 | sandbox，tool manager，QPS quota，timeout recovery，async reward |

## 12. 工程落地清单

如果要实现一个 OPD/MOPD 系统，建议按以下顺序推进。

### 12.1 先做最小 sampled-token OPD

- 复用现有 RL rollout pipeline。
- rollout 时保存 student logprob。
- teacher 对同一 trajectory 调 `compute_logprobs`。
- `advantage = teacher_logprob - student_logprob`。
- 用原 RL loss 更新 student。

最小版本的目标不是追求 SOTA，而是验证 teacher-student 是否可蒸馏。

### 12.2 加入可蒸馏性诊断

- 采样固定 validation prompts。
- 计算 student/teacher top-k overlap ratio。
- 计算 entropy gap。
- 比较 teacher 分数、student 分数、蒸馏后分数，统计 gap recovery。
- 如果 early overlap 很低，优先做 cold start，而不是盲目加大训练。

### 12.3 做冷启动和 prompt 对齐

- teacher rollout 生成一批 SFT 数据。
- SFT 不追求最终能力，只负责把 student 拉到 teacher support 附近。
- OPD prompts 混合 teacher-aligned、in-domain dedup、OOD prompts。
- 避免只用 teacher 训练分布导致 entropy collapse。

### 12.4 从 sampled-token 走向 top-k/full-vocab

- sampled-token 版本出现高方差或不稳定时，考虑 top-k KL。
- 多 teacher、重要发布模型、能力融合阶段可考虑 full-vocabulary OPD。
- full-vocab 必须配套 logits/hidden state 缓存、teacher head scheduling 和专用 KL kernel。

### 12.5 加入 outcome reward

- 对数学、代码、Agent 任务，teacher reward 不应完全替代 verifier/ORM。
- 用 `A = A_OPD + alpha * A_ORM` 混合局部过程监督和最终结果监督。
- 监控 reward hacking、答案格式、长度偏差和工具调用失败率。

### 12.6 系统稳定性优先级

- rollout 必须支持抢占恢复，否则中断重采样会引入长度偏差。
- MoE 模型要处理 rollout/training routing 不一致。
- 长上下文任务要拆 metadata 和 per-token heavy fields，避免 CPU/GPU 内存爆炸。
- Agent 任务要把 sandbox、tool quota、环境预热和 timeout recovery 纳入训练系统，而不是当外部脚本处理。

## 13. 多专家 OPD 的 teacher 选择原则

| 维度 | 建议 | 原因 |
| --- | --- | --- |
| 能力 | teacher 应在目标领域显著强于 student | 否则没有能力增益 |
| 新信息 | teacher 最好有额外 RL、数据或工具经验 | benchmark 高不一定有新知识 |
| 兼容性 | 优先同 tokenizer、同 family、同 thinking template | 提高 top-k overlap，降低局部优化困难 |
| 专家化 | teacher 可以强烈偏科 | 最终由 MOPD 融合，不要求单 teacher 通用 |
| 稳定性 | teacher 输出分布应可控，避免极低 entropy 或格式漂移 | 防止 student entropy collapse |
| 成本 | 大 teacher 要有 offload、batching 和缓存方案 | 多 teacher OPD 的瓶颈通常是系统吞吐 |

## 14. OPD 的典型失败模式

| 失败现象 | 可能原因 | 处理方法 |
| --- | --- | --- |
| teacher 分数高但 student 不涨 | thinking pattern mismatch | off-policy cold start，换同源 teacher，teacher-aligned prompts |
| early KL 很大且梯度噪声高 | sampled-token 方差大或 support 不重叠 | top-k/full-vocab OPD，增大 batch，先 SFT |
| 初期涨分后退化 | prompt 过窄、entropy collapse、过拟合 teacher 分布 | 混入 OOD prompts，加 entropy/KL 监控 |
| 长链任务后段错误变多 | teacher 在 student 错误长前缀上信号退化 | segment OPD，partial rollout，叠加 outcome reward |
| 多领域能力互相伤害 | teacher 路由、领域配比或权重不合理 | 领域 quota，动态 teacher weighting，per-domain eval |
| rollout 中断后长度分布异常 | 失败恢复时重采样 | token-level WAL，保存 KV cache，确定性 resume |
| MoE 训练不稳定 | rollout 和 training expert routing 不一致 | routing replay，batch-invariant deterministic kernels |

## 15. 对 OPD 的系统性理解

可以把 OPD 看成后训练系统里的“策略压缩层”：

- RL 负责探索，把 sparse outcome reward 转化成强专家策略。
- OPD 负责压缩，把专家策略以 dense token reward 的方式迁移给 student。
- MOPD 负责融合，把多个专家策略合到同一参数空间。
- SFT / cold start 负责铺路，把 student 拉进 teacher 可教的支持域。
- ORM / verifier 负责校准，防止局部 token 模仿偏离最终任务目标。

因此 OPD 最适合的场景不是“从零创造能力”，而是“已有强策略后，高效复制、恢复、融合、迭代”。

## 16. 未来方向

- 更可靠的长轨迹 OPD：解决 teacher reward 随深度退化的问题。
- 自适应 supervised horizon：短问题 full OPD，长问题分段 OPD + outcome reward。
- teacher 可蒸馏性预测：训练前用 overlap/entropy/trajectory diagnostics 预测收益。
- 多 teacher 自动路由：按 prompt、trajectory prefix、uncertainty 动态选择 teacher。
- full-vocab OPD 系统优化：hidden state cache、head scheduling、低精度 logits KL kernel。
- self-distillation：同一模型在 privileged context、工具增强或更高 test-time compute 下作为 teacher。
- Agent OPD：把 tool trace、sandbox state、执行结果和 token-level teacher reward 统一建模。

## 17. 参考资料



- Thinking Machines Lab：On-Policy Distillation：https://thinkingmachines.ai/blog/on-policy-distillation/
- Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe：https://arxiv.org/abs/2604.13016
- DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence：https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf
- DeepSeek-V4 Hugging Face model card：https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
- MiMo-V2-Flash Technical Report：https://arxiv.org/abs/2601.02780
- On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes：https://arxiv.org/abs/2306.13649

