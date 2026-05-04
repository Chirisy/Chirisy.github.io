# 从 Reasoning RL 到 Agentic RL：Credit Assignment 的系统理解

> 主题：基于论文 **From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models**，系统梳理 Reasoning RL 与 Agentic RL 的核心差异，以及为什么 Credit Assignment 是二者训练范式变化中的关键问题。  
> 综述 arXiv `2604.09459v1` 和用户给出的飞书阅读笔记
> 关键词：Reasoning RL、Agentic RL、Credit Assignment、GRPO、PRM、LLM-as-Critic、Hindsight Credit、Counterfactual Credit。

## 1. 一句话总结

Reasoning RL 主要是在**单次生成的思维链内部**回答“哪一步推理对最终答案有贡献”；Agentic RL 则是在**多轮环境交互轨迹中**回答“哪一次工具调用、规划、搜索、执行或沟通真正改变了任务结果”。二者都面对稀疏终局奖励，但 Agentic RL 的环境随机性、部分可观测、超长轨迹和异质动作，使 Credit Assignment 从一个优化效率问题变成训练能否成立的核心问题。

| 维度 | Reasoning RL | Agentic RL |
| --- | --- | --- |
| 典型任务 | 数学、代码、逻辑推理、可验证 CoT | 网页浏览、工具使用、软件工程、桌面自动化、多智能体协作 |
| 轨迹形态 | 一次生成，一个回答 | 多轮交互，动作后有环境反馈 |
| 状态 | prompt + 已生成 tokens，基本完全可见 | 历史上下文 + 局部观察 + 隐藏环境状态，部分可观测 |
| 动作 | 下一个 token、一个推理段或一步 CoT | 一整轮回复、工具调用、搜索、代码执行、沟通 |
| 转移 | 自回归生成，近似确定 | API、网页、代码执行、外部环境，随机且不可微 |
| 奖励 | 通常是最终答案对错 | 通常是任务成功/失败，且中间步骤难验证 |
| 归因粒度 | token、segment、step | turn、action、critical action、agent |
| 主流方向 | PRM、token/step 级优势、critic-free group comparison | turn-level critic、hindsight/counterfactual、hierarchical CA、LLM-as-Critic |

## 2. 为什么要从 Credit Assignment 看 LLM RL

强化学习训练 LLM 时，经常只有一个终局奖励，例如“答案是否正确”“任务是否完成”。但策略更新发生在许多 token、步骤或轮次上，因此必须把一个稀疏的结果信号分配给轨迹中的局部决策。

Credit Assignment 关心的是：

```text
给定一条轨迹 τ 和终局奖励 R(τ)，
如何为轨迹中的 token、step、turn 或 agent 分配 credit c_i，
使策略优化更接近真正导致成功或失败的决策？
```

从这个视角看，常见 LLM RL 算法的差异可以重新解释：

| 算法 | Credit Assignment 方式 | 问题 |
| --- | --- | --- |
| REINFORCE | 把整条轨迹的回报分给所有动作 | 最粗糙，方差高 |
| GRPO | 同一 prompt 下多条轨迹做组内相对比较，但一条轨迹内所有 token 仍共享同一 advantage | 省掉 critic，但仍是 episode-level credit |
| PPO | 用 value function / critic 估计 token-level advantage | 粒度更细，但 critic 难训练、成本高 |
| DPO | 偏好优化中隐式学习 token-level Q/value 结构 | 理论优雅，但显式提取和控制 credit 不直接 |

论文的核心判断是：随着 LLM 从回答问题走向操作环境，Credit Assignment 的难度不是线性增加，而是问题形态发生了变化。

## 3. Reasoning RL：单次生成里的推理归因

Reasoning RL 可以建模为 token-level MDP：

- `state`：prompt 加上已经生成的 token。
- `action`：下一个 token，或聚合后的 reasoning step。
- `transition`：自回归生成，给定前缀后状态转移基本确定。
- `reward`：通常只在最终答案处给出，例如数学题正确/错误。

它的核心问题是：一条 CoT 可能有几百到三万多个 token，最终答案正确并不代表每个 token 都有正贡献；最终答案错误也不代表所有步骤都错。因此，只用 GRPO/REINFORCE 把同一个 advantage 分给整条回答，会把“真正关键的推导步骤”和“格式性 token”混在一起。

### 3.1 Token-level 方法

Token-level 方法希望把 credit 分配到最细粒度。

| 方法 | 思路 | 优点 | 局限 |
| --- | --- | --- | --- |
| VinePPO | 从中间 prefix 分叉采样多个 continuation，用 Monte Carlo 估计每个 token 位置的 value/advantage | 不依赖 learned critic，credit 更准 | 额外 forward 成本随序列长度上升 |
| RED | 利用已有 Reward Model 的隐藏表征，训练轻量 probe 估计 token 对总奖励的贡献 | 便宜，能复用 RM 信息 | credit 信号依赖 RM 表征质量 |
| T-REG | 让模型生成正确/错误解，通过 token logprob 差异找出区分性 token | 不需要外部 RM/critic | 更像启发式自监督信号 |
| DPO implicit credit | DPO 中 policy/ref logprob ratio 可解释为隐式 token-level Q 信息 | 可能从偏好模型中“免费”获得 credit | 难直接转成稳定的显式训练信号 |

Token-level 方法适合长 CoT 中寻找细粒度错误，但在 hard math 或长代码推理里，逐 token 估计往往过贵。

### 3.2 Segment-level 方法

Segment-level 方法把连续 token 聚成语义片段，例如“建立方程”“代入求解”“验证边界条件”。这比 token-level 便宜，也比 episode-level 更有信息量。

| 方法 | 思路 | 适用直觉 |
| --- | --- | --- |
| SPO | 在 reasoning chain 的语义 cutpoint 处分段，对共享 prefix 的轨迹做 MC 比较 | 数学推理的自然步骤通常是片段级 |
| TEMPO | 把推理路径建成树，对分支进行 MC + TD 式传播 | 适合存在多条推理路径的问题 |
| SCAR | 用 Shapley value 估计每个片段对最终结果的边际贡献 | 理论上公平，但 coalition 评估成本高 |

Segment-level 的本质是承认：对语言模型来说，真正的决策单位往往不是一个 token，而是一段语义完整的推理动作。

### 3.3 Step-level 与 PRM

Process Reward Model（PRM）可以看作 step-level Credit Assignment。它不是单纯的奖励模型，而是在把终局 reward 拆给中间推理步骤。

典型路径是：

1. 把 CoT 拆成多个 reasoning steps。
2. 为每一步估计“从这里继续下去能否到达正确答案”。
3. 将 step score 用作 policy update 的局部 reward 或 advantage。

代表方法包括：

| 方法 | 核心贡献 |
| --- | --- |
| Math-Shepherd / OmegaPRM | 通过采样 continuation 自动标注 step correctness，为 PRM 提供训练信号 |
| PURE | 认为普通 sum-form PRM 容易被 reward hacking，提出 min-form credit 让错误更难被后续高分步骤掩盖 |
| SPRO | mask 掉某一步，观察结果能力下降多少，以 leave-one-out 方式估计该步必要性 |
| HICRA | 区分 planning token 与 procedural token，让不同类型的推理内容获得不同 credit |
| CAPO / LLM-as-Critic 类方法 | 让 LLM 对中间步骤进行语义评价，生成 step-level credit |

Reasoning RL 的 Credit Assignment 正在成熟，原因是它有三个有利假设：

1. **确定性转移**：从 prefix 继续生成很容易，MC 分叉成本相对可控。
2. **单次生成轨迹**：没有真实环境状态需要恢复。
3. **结果可验证**：数学、代码、选择题等任务有明确正确性信号。

这些假设一旦失效，Reasoning RL 的方法就很难直接搬到 Agentic RL。

## 4. Agentic RL：多轮环境交互里的行动归因

Agentic RL 更适合建模为 turn-level POMDP：

- `state`：完整环境状态、历史交互、检索上下文等，但其中很多对模型不可见。
- `observation`：模型实际看到的文本、网页、工具返回、日志片段。
- `action`：一整轮响应，可能包含工具调用、计划、代码、搜索 query、对其他 agent 的消息。
- `transition`：由环境执行动作后产生，可能随机、延迟、失败、不可微。
- `reward`：通常只在最终任务完成时给出。

Agentic RL 的 Credit Assignment 是双层问题：

1. 哪一轮 turn 或哪一个 action 是关键的？
2. 在该 turn 内，哪些 token 或子动作真正重要？

这比 Reasoning RL 更难，因为“最后成功”可能由很早的一次搜索 query 决定；“最后失败”也可能是环境变化、工具超时或信息不可见造成的，并不一定说明某个动作本身不好。

### 4.1 Agentic RL 的六个难点

| 难点 | 说明 | 对 Credit Assignment 的影响 |
| --- | --- | --- |
| 环境随机性 | API 失败、网页变化、代码执行非确定 | 从某一步重新 rollout 不再便宜，MC 分叉困难 |
| 部分可观测 | agent 只能看到观察文本，看不到完整环境状态 | outcome 不能直接等同于动作质量 |
| 超长 horizon | SWE-bench、OSWorld 等可能有 50-100+ turns、100K-1M tokens | episode-level gradient 方差急剧增大 |
| 动作异质性 | 搜索、计划、写代码、执行、沟通的性质不同 | 统一 token credit 无法表达不同动作类型 |
| 中间状态难验证 | 搜索 query、点击、规划是否“好”通常要事后才知道 | PRM 式 step correctness 难直接标注 |
| bifurcation point 稀疏但关键 | 少数决策点决定成败，大量动作只是常规执行 | 均匀 credit 会浪费训练信号并放大噪声 |

论文特别强调，Agentic RL 中的关键不是给每个动作都分一个精确分数，而是找到那些“真的改变轨迹走向”的关键动作。

## 5. Agentic RL 的主要 Credit Assignment 路线

### 5.1 Turn-level PRM / Critic

这类方法把一整轮交互当成 credit 原子单位，而不是逐 token 处理。

| 方法 | 核心思路 |
| --- | --- |
| AgentPRM | 用 TD + GAE 训练 step/turn-level critic，避免每一步都重新执行环境做 MC 标注 |
| SWEET-RL | 使用 privileged critic：训练时让 critic 看到 ground truth、未来轨迹或额外环境信息，actor 推理时仍只看正常 observation |
| Turn-Level Reward Design | 对可验证动作使用程序化验证，对主观动作使用 LLM-as-judge |
| Turn-PPO | 将多轮 agent 轨迹改写成 turn-level MDP，使用 turn-level value 和 importance ratio |
| SORL | 用 turn-level importance sampling 与归一化机制稳定长程 off-policy 训练 |
| TARL / ITPO | 通过 LLM judge 或隐式 turn-level reward 给多轮交互提供过程监督 |

Turn-level 是 Agentic RL 的自然粒度：一次工具调用或一次环境反馈之后，任务状态才真正发生变化。

### 5.2 Hindsight 与 Counterfactual

Agentic RL 中，事前很难判断一个动作好坏，但事后可以根据完整轨迹进行回看。

| 方法 | 核心思路 |
| --- | --- |
| HCAPO | 轨迹完成后，让 LLM critic 基于结果回看每个 turn 的贡献，并生成反事实分析 |
| C3 | 用 leave-one-out：如果去掉某个 turn/agent，结果会怎样，从差值估计 credit |
| CCPO | 将 counterfactual credit 用于 policy optimization |
| CriticSearch | 针对搜索型 agent 进行回溯式 critic 评估 |

Hindsight 的优势是信息更多：知道最后发生了什么，就能区分“动作本身好”与“碰巧成功/碰巧失败”。代价是会引入额外延迟，也可能产生 hindsight bias。

### 5.3 Critic-free Step-level 方法

Agentic RL 的训练成本很高，因此不训练额外 critic 的方法很有吸引力。

| 方法 | 核心思路 |
| --- | --- |
| GiGPO | 在 GRPO 的组间比较之外，引入轨迹内部 step grouping，形成 group-in-group 的 step-level advantage |
| CARL | 用 action entropy 找 critical actions，只对关键高熵动作更新，避免给常规动作浪费 credit |
| iStar / ITPO | 从 DPO 或 policy logprob 变化中提取隐式 step/turn reward |
| SPA-RL | 用轻量 MLP 预测 progress，credit 是 progress 的增量 |
| RAGEN / StarPO | 识别 episode-level credit 下的 echo trap，并用不确定性过滤降低噪声 |

这一路线的直觉是：Agentic RL 的瓶颈往往不是 credit 最精确，而是能否低成本、低噪声地告诉模型“哪些动作值得学”。

### 5.4 Hierarchical 与 Action Decomposition

Agentic task 往往天然包含层级：计划、执行、验证、修复。层级化 credit 可以避免把所有动作摊平成 token 序列。

| 方法 | 核心思路 |
| --- | --- |
| ArCHer | 高层 critic 负责 turn-level Q，低层 actor 负责 turn 内 token policy |
| PilotRL | 先做 plan-level RL，再做 step-level RL，最后细化到 token-level |
| POAD | 同时处理 inter-action credit 与 intra-action token credit |
| HICRA 的迁移价值 | planning/procedural 区分可迁移到 agent 的 plan-execute 结构 |

这类方法说明：Agentic RL 不应该简单套用单层 PPO/GRPO，而要让优化粒度贴合 agent 的实际行动结构。

## 6. 两类 RL 的本质差异

| 问题 | Reasoning RL 的回答 | Agentic RL 的回答 |
| --- | --- | --- |
| 什么是一个 action？ | token、推理片段、一步 CoT | 一整轮回复、工具调用、环境操作、agent 消息 |
| 什么是 state？ | prompt + 已生成内容 | 部分观察到的环境、历史、外部状态 |
| credit 应该给谁？ | 正确推理步骤、关键 token | 改变环境走向的 turn/action/agent |
| 为什么 GRPO 不够？ | 长 CoT 下同一 advantage 太粗 | 多轮任务中关键动作和无关动作被混为一谈，SNR 崩塌 |
| 最自然的细化方向 | token/segment/step PRM | turn-level、hindsight、counterfactual、critical action |
| 最大工程约束 | 额外 forward / PRM 标注成本 | 环境 reset、沙箱执行、工具调用成本、安全与异步 rollout |

一句话区分：

- Reasoning RL 的 credit 问题是“哪一步推理导向正确答案”。
- Agentic RL 的 credit 问题是“哪一个行动在真实或模拟环境中改变了任务状态”。

## 7. 方法选型建议

结合论文的 practical guidance，可以按任务形态选方法：

| 场景 | 特征 | 优先考虑 |
| --- | --- | --- |
| GSM8K/MATH 等短 CoT | 可验证、确定性、轨迹短 | GRPO 作为 baseline；PURE、SPO、SPRO 做更细 step/segment credit |
| AIME/IMO 等长 CoT | 10K-30K tokens，可验证但 credit 更难 | VinePPO、HICRA、CAPO；关键看额外 compute 是否可接受 |
| WebShop/ALFWorld 等工具使用 | 5-20 turns，部分动作可验证 | GiGPO、AgentPRM、Turn-PPO；低成本时偏 critic-free |
| WebArena 等网页导航 | 10-30 turns，随机、部分可观测 | SWEET-RL、HCAPO、IGPO；可利用 privileged critic 或 hindsight |
| SWE-bench 等软件工程 | 50-100+ turns，长上下文，中间状态难验证 | CARL、HCAPO、C3/CCPO、ArCHer；重点是 sparse critical action 与反事实分析 |
| 多智能体系统 | team reward、跨 agent 协作 | M-GRPO、LLM-MCA、C3、SHARP、MAPPA；重点是跨 agent credit decomposition |
| 算力受限训练 | GPU/rollout 成本有限 | GRPO、CARL、iStar、GiGPO；少用大 critic 和大量 MC 分叉 |

实用判断顺序可以简化为：

1. 如果是短、可验证、单次生成任务，先用 GRPO/RLOO 类 baseline，再叠加 PRM 或 step-level credit。
2. 如果是长 CoT 且可验证，优先考虑 segment/step credit，而不是逐 token credit。
3. 如果是多轮工具或网页任务，把 turn 当作最小决策单位。
4. 如果中间过程不可验证，优先考虑 hindsight、counterfactual 或 privileged critic。
5. 如果 rollout 极贵，优先选择 critic-free、低 overhead、critical-action focused 方法。

## 8. 放到 Agentic RL 训练流水线里看

论文把 Agentic RL 训练拆成五个阶段：

```text
环境构建 -> rollout 生成 -> 终局奖励计算 -> credit assignment -> policy update
```

Credit Assignment 不只是 reward 后处理，它会影响整个系统：

- **rollout efficiency**：更好的 credit 能降低梯度方差，用更少 rollout 达到同等学习效果。
- **reward design**：PRM、progress reward、information gain reward 都在把稀疏终局奖励改写成更可学习的过程信号。
- **exploration**：理想情况下，credit 不确定性可以反过来指导 agent 去探索最有信息量的状态。
- **infrastructure**：Agentic RL 中环境 reset、浏览器/容器沙箱、API 调用、异步 rollout 和 policy lag 都会改变 CA 方法的可用性。

因此，在 Agentic RL 中选择 CA 方法不能只看算法公式，还要看环境能不能 checkpoint、能不能重放、rollout 是否安全、critic 是否能看到额外训练信息，以及 policy update 是否允许 off-policy 修正。

## 9. 未来问题

论文指出几个值得继续跟踪的方向：

1. **Ultra-long horizon agents**：当 agent 进入 100+ turns、数十万 token 的任务，turn-level credit 也可能过细或过贵，需要更深的动态层级。
2. **Open-world reward**：真实助手、研究 agent、创作 agent 的终局 reward 本身不明确，credit assignment 必须和 reward modeling 一起重做。
3. **统一 benchmark**：当前不同论文的 base model、任务和训练配置差异大，需要可控 bifurcation point、已知 ground-truth credit 的基准。
4. **Memory credit**：长期记忆的写入、检索、总结可能在很多轮之后才体现价值，现有 CA 方法难处理这种长延迟贡献。
5. **Reasoning 到 Agentic 的迁移**：VinePPO、PURE、HICRA 等 reasoning 方法可以迁移到 turn-level，但需要环境 checkpoint、反事实重放或新的层级结构。
6. **Multi-agent credit**：未来 agent 系统会越来越多地由多个角色协作，credit 不只要按时间分，还要按 agent 分。

## 10. 核心 takeaway

1. Credit Assignment 是 LLM RL 的中心问题。模型越从单次回答走向真实交互，稀疏终局奖励就越难直接用于策略更新。
2. Reasoning RL 的 CA 已经形成较清晰的技术谱系：token-level、segment-level、step-level PRM、critic-free group comparison。
3. Agentic RL 的 CA 仍处在早期，但已经出现不同于 Reasoning RL 的新方向：hindsight/counterfactual、privileged critic、critical action identification、turn-level MDP。
4. LLM-as-Critic 是 LLM 时代特有的 credit 方法。它能进行语义评价和自然语言解释，但成本、偏差和一致性仍是问题。
5. 从 Reasoning RL 到 Agentic RL，不是“把轨迹变长”这么简单，而是从“推理步骤归因”转向“环境行动归因”。

## 资料来源

- arXiv PDF：<https://arxiv.org/pdf/2604.09459v1>
- arXiv HTML：<https://arxiv.org/html/2604.09459v1>
- Hugging Face paper page：<https://huggingface.co/papers/2604.09459>
