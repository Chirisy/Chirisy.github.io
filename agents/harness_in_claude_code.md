# Agent Harness in Claude Code

> LLM 拥有强大的文本生成能力。在 RL 和工具调用的 setting 下，将模型放进多轮交互、文件系统、浏览器、解释器和权限系统组成的环境里，它才真正成为一个 agent。实现生产级 agent 的关键，不只是模型本身，而是围绕模型搭建的 Agent Harness。
>
> 本文整理自播客：[十字路口 Crossing：探秘 Claude Code，搞懂 Agent Harness｜对谈来新璐](https://www.xiaoyuzhoufm.com/episode/69f2e83fbb3ffa11e59dec82?s=eyJ1IjoiNjlmYzhmZGIyNTRmYmNhNDcwNGYwOTAxIn0%3D)
>
> 参考项目：[Learn Claude Code](https://learn.shareai.run/zh/s01/)

## 将收获什么

1. 从工程语言理解 Agent Harness：模型以外的执行环境、上下文、状态、权限、记忆和编排系统，都属于 Harness。
2. 用三层框架拆解生产级 Agent：会跑的执行能力层、跑久的上下文与状态层、跑稳的治理与编排层。
3. 理解 Claude Code 代表的 Agent Native 思路：更多 context、更少 control，让模型拥有足够好的 action space，而不是用 Prompt Flow 硬控每一步。
4. 梳理 Claude Code 的上下文压缩、接力交接、forked agent、auto-dream 记忆维护等机制。
5. 观察 Agent Infra 的创业方向：轻量级 Agent Computer、Agent 组网、个性化 Agent 模型训练与推理基础设施。

## Concept: 模型以外都是 Harness

如果要用一句话解释 Agent Harness，可以说：模型以外都是 Harness。

模型像一个聪明的大脑，但只有大脑无法行动。它需要身体、手脚、工作台、工具箱、记忆系统和管理机制，才能把“会想”变成“会做”。在 Agent 系统里，这些模型之外的工程部分共同构成 Harness：包括工具调用、文件系统、CLI、浏览器、代码解释器、system prompt、skills、memory、上下文压缩、权限隔离、多 Agent 编排、观测和反馈闭环。

因此，“Agent 的上限由 Harness 决定”这句话并不是否认模型智能的重要性。模型仍然是第一性因素，强模型会直接提升任务表现。但当模型能力进入可用区间后，Agent 能不能完成长程任务、能不能持续工作、能不能稳定迭代，就越来越取决于 Harness 的设计。

播客里用了一个很形象的比喻：Harness 像机甲。它不直接提升驾驶员的智力，却极大扩展了驾驶员能完成的动作、能进入的环境和能承受的任务复杂度。今天的前沿模型已经足够聪明，很多 Agent 产品之间的差距，不再只是“模型 IQ 差多少”，而是有没有给模型一个合适的行动环境。

## Tool: Bash/CLI is all you need

播客里反复强调的一个共识是：CLI is all you need，或者更极端地说，Bash is all you need。

这背后不是复古情绪，而是和模型训练分布有关。Unix、Linux、Shell、CLI 命令已经存在了几十年，互联网上有极其丰富的语料。LLM 在预训练阶段见过大量命令行样本，因此对命令组合、错误修复、参数查找、管道调用和脚本化自动化非常熟悉。

相比之下，MCP 是近两年才出现的新协议，预训练语料里的占比很低。它作为工具接入协议有价值，但模型对它的熟悉程度、组合空间和二次编程能力，通常不如 CLI。新璐举了 GitHub 的例子：早期使用 GitHub MCP 能明显解放手动操作，但切到 GitHub CLI 后，很多任务的完成率和灵活性反而更好。

这也解释了为什么很多 Agent 框架正在从 Prompt Flow / Node Graph 的范式转向 Agent Native 范式。过去的 LangChain、LangGraph 等框架擅长把任务拆成 prompt 节点、状态图和路由规则，开发者觉得每一步都可控。但这种方式越到强模型时代，越容易把模型锁进开发者预设的流程里。

Agent Native 的思路正好相反：模型才是 agent。开发者要做的不是替模型规划每一步，而是给它足够好的上下文空间和行动空间，让它自己决策、调用工具、检查结果、修正错误。

## Framework: Harness 三层拆解

可以把 Agent Harness 拆成三层：执行能力层、上下文与状态层、治理与编排层。

### 第一层：执行能力层

执行能力层解决的是 agent “能做什么”。

最基础的是文件系统能力：创建、删除、读取、写入、搜索文件。无论是 coding agent、research agent，还是文档处理 agent，文件系统都是最低限度的工作环境。其次是浏览器能力，agent 需要能访问用户世界中的网页、表单、系统和应用。再往上是语言解释器和运行环境，例如 Python、Node.js、Bash、数据库 CLI、GitHub CLI、云服务 CLI 等。

这一层听起来像“给工具”这么简单，但真正的难点在权限和角色绑定。比如只负责探索代码库的 agent，应该只拥有无副作用的读操作，不应该有删除、重写或提交代码的权限。负责测试的 agent 应该能运行测试、读取结果，但不应该一边测试一边修改代码，否则它可能为了让测试通过而 hack 掉真实问题。

工具不是越多越好，而是要和 agent 的角色、任务边界和风险等级绑定。

### 第二层：上下文与状态层

上下文与状态层解决的是 agent “知道什么、记得什么、如何接着做”。

这包括 system prompt、skills、memory、工作目录、依赖环境、项目结构、历史任务进展，以及上下文窗口满了之后的交接策略。长程任务不可能永远塞在一个上下文窗口里完成，agent 需要在某个阶段写下当前进展、未完成事项、已尝试路径和后续计划，再交给下一个 agent 接力。

播客里提到的案例是：多个 agent 在两周内协作，从零构建一个复杂项目。这个过程不是一个模型拿到工具就能完成的，它必须知道当前项目已经到哪一步、哪些模块已经完成、哪些模块可以并行推进、哪些决策不能重复推翻。

因此，状态层的关键不是把所有信息都塞给模型，而是决定哪些信息进入当前上下文，哪些信息留在文件里按需读取，哪些信息要被压缩，哪些信息要在交接时明确写出来。

### 第三层：治理与编排层

治理与编排层解决的是 agent “如何组织起来稳定工作”。

当系统里只有一个 agent 时，编排问题还不明显。但一旦任务需要几十、几百甚至上千个 agent 协作，就必须处理并行、串行、权限、隔离、依赖、冲突、测试、验收和观测。

例如，哪些模块可以并行开发？不同 agent 如何交接上下文？测试 agent 和写代码 agent 的权限如何隔离？探索 agent 是否允许访问外网？修改 agent 是否允许删除文件？如果多个 agent 同时改同一个目录，谁来合并冲突？这些都属于治理层问题。

这也是生产级 Agent 和 demo 级 Agent 的分水岭。demo 只要看起来会调用工具即可，生产级系统则必须可控、可观测、可追责、可恢复。

## Memory: External 和 Internal

记忆是 Agent Harness 里最容易被混淆的部分。播客中把 memory 分成三类：结构化、半结构化和模型内化。

第一类是完全结构化的记忆，例如知识图谱、向量数据库、节点关系和规则式检索。这种方案的优点是可查询、可解释、适合 pipeline 化推理；缺点是过于依赖预设结构，维护成本高，也不一定符合模型自然工作的方式。

第二类是半结构化记忆，也就是 Unix Files + Markdown + Agent 驱动更新。Claude Code 和一些新项目更偏向这种路线：用文件夹和 Markdown 保存记忆，让人和模型都能读懂；再通过 agent 定期整理、纠错、合并和更新。这种方式既保留了结构，又没有把信息压成过度僵硬的 schema。

第三类是模型内化，也就是把经验真正训练进模型参数里。它的长期价值很大，但短期还不够实用：模型记忆难以批量提取、无损迁移和独立维护，也容易被单一模型提供商绑定。新璐判断，真正可生产落地的模型内化记忆，可能还需要数年时间。

这里还涉及 memory 和 skill 的边界。Claude Code 早期的 Insights 功能会分析最近一段时间的对话，判断任务为什么成功或失败、模型犯了什么错，再生成报告指导 skill 的生成。也就是说，记忆、技能和自我迭代之间存在重叠区。与其纠结标签，不如看目标：能否让 agent 从过去的轨迹中学习，并在未来任务中表现得更好。

## Claude Code: Frontier Context Engineering

Claude Code 之所以值得研究，是因为它展示了一套更贴合强模型的 Agent Harness 思路：更多 context，更少 control。

传统 Prompt Flow 的做法，是开发者预设各种节点和条件，让模型按流程流转。Claude Code 的核心哲学则是：模型才是 agent，Harness 应该给模型提供合适的工具、上下文和记忆，而不是替模型决定每一步。

这套思路在上下文工程里体现得很明显。

第一，Claude Code 的上下文压缩不是简单裁剪。它会判断工具 output 什么时候可以删除，哪些信息必须保留，哪些内容可以让下一个 agent 自己去文件里查。上下文窗口满了以后，系统需要把任务进展、用户目标、当前状态和后续计划交接给下一个 agent，而不是粗暴丢弃历史。

第二，好的上下文管理要和模型推理机制自洽。随意修改 system prompt、随意裁剪前文、频繁破坏 KV cache，会造成重算和性能浪费，也会让模型失去连续性。播客里甚至提到：“最好的管理就是不要做管理”，意思不是完全不管上下文，而是不要用不理解模型运行机制的方式强行干预。

第三，Claude Code 的记忆机制和 skills 保持同一套哲学。记忆文件不会一开始全文加载，而是先读取文件名和 description，再由模型判断是否需要进一步打开。skill 的组织方式也是类似的：通过简短元信息让模型先完成路由，再按需读取细节。

第四，Claude Code 有两类记忆更新机制。一类发生在每轮交互结束后，通过 stop hook 触发 forked agent。forked agent 会复用上一轮的系统提示和上下文，用后台任务更新记忆。另一类是 auto-dream：大约每天触发一次，在满足会话数量等条件后，让后台 agent 回放最近的 sessions，提取有用信息、纠正过时事实、合并重复记忆。这个过程像人类睡眠中的整理和重放，所以被称为 dreaming。

这些机制共同说明，Claude Code 的关键不是“泄露源码里有什么神秘 prompt”，而是它围绕模型推理逻辑做了大量工程兜底。

## Agent Infra: Komputer Blue, 代号 KB

新璐团队正在做的方向可以理解为 Agent Computer 工具链。他们的核心判断是：要让 agent 工作得好，就应该给每个 agent 一个熟悉、轻量、可组合的计算机环境。

这套工具链被拆成几层。

Komputer 是最底层的 Agent Computer。它用 TypeScript 重新实现一套类似 Unix 的文件系统和 Bash 环境，让 agent 在浏览器、插件、App、Electron、小程序、静态网页、全栈 SaaS 等任何能跑 JavaScript 的场景中，都能拥有一致的执行环境。在支持 WebAssembly 的场景中，可以切换到 Rust/WASM 实现；不支持时则 fallback 到纯 JavaScript。

Kruntime 是 Agent Runtime 层，给开发者提供创建、运行和派生 agent 的接口。它既服务人类开发者，也服务 agent 创建 agent 的未来场景。

Kwatch 是观测层，用来分析 agent 在哪些任务、哪些阶段、哪些工具调用上卡住。观测结果可以反向指导 CLI 设计、skill 补充、memory 更新和模型选择。

KRL 则面向更长期的数据闭环：把 agent 在 runtime 上沉淀的轨迹、good cases 和 bad cases 用于强化学习、上下文优化或个性化模型训练。

这一路线和云厂商提供的全量沙箱不同。E2B、Daytona、Agent Bay 这类方案更像“给 agent 一台真实或接近真实的云端机器”，而 KB 追求的是极致轻量：不把完整 Linux、浏览器、GCC 编译器都塞进每个 agent 的环境，而是用数据结构模拟最小 Unix 文件系统、最小 Bash、局域网通信、共享磁盘、虚拟时钟和后台进程能力。

它的取舍是明确的：不适合在内部真实跑重型编译器或浏览器，但可以用极低成本给大量 agent 提供一致的生活和协作环境。重型能力则应该作为外部 infra 服务被调用。

## What Makes a Good Harness

一个好的 Harness，首先要和模型的 inference 逻辑自洽。

不好的 Harness 会随意裁剪上下文、频繁改写系统提示、破坏 KV cache，或者用 Prompt Graph 硬控每一步决策。这样做在弱模型时代或许能提高可控感，但在强模型时代会压制模型的自主推理和工具使用能力。

好的 Harness 可以用一个公式概括：

> good context space + good action space + less prompt control

好的 context space 让模型知道任务目标、环境状态、历史进展和可用知识；好的 action space 让模型能通过 CLI、文件系统、浏览器、解释器和外部服务完成真实动作；less prompt control 则意味着开发者不要过度设计流程，把决策权还给模型。

这也是为什么 Harness 必须和模型进步方向正交。模型越强，好的 Harness 应该让整体 Agent 系统越强，而不是因为流程写死、工具僵硬、上下文裁剪粗暴，导致强模型也被束缚住。

## Conclusion: Agent 的未来

播客最后提到三个值得关注的创业方向。

第一是 Agent Harness 工具链，也就是围绕执行环境、runtime、memory、skills、观测和数据闭环做基础设施。新璐团队的 KB 属于这一类。

第二是 Agent 组网。未来 agent 不只运行在云端，也会运行在 Mac、手机、NAS、路由器、浏览器和各种边缘设备上。它们需要混合组网、高通量上下文交换和更 agent-native 的控制能力，而不只是发 IM 或邮件。现有的 Tailscale 等工具给了启发，但还不是专门为 agent 设计。

第三是 Agent 模型的集约化训练和推理基础设施。未来不一定所有人都调用同一个基础模型 ID，而是可能拥有面向自己场景微调过的个性化 agent 模型。类似 LoRA 热插拔推理的方式，可以让用户在请求 header 中引用自己的个性化参数，以较低成本获得更适合自己 ERP、CRM 或特定工作流的模型。

对更远的未来，新璐给了一个很激进的判断：公司可能会越来越像一种理财产品。今天大家讨论一人公司，但他认为真正本质的趋势是零人公司。公司内部本来就是一个黑盒，只要它能接收输入、产出结果、创造收入，未来就可能由 agent 组成、自管理、自进化。

YoYo Agent 是一个早期例子：创作者把 agent 放出去后不再改代码，也不给它持续供给资金，它需要自己想办法赚钱、获得 token、进化自己，并以超越 Claude Code 为目标。

这听起来还很早期，甚至有些科幻。但如果单 agent 会进化成 agent swarm，再进化成 agent 自管理和 agent 创造 agent，那么 Agent Harness 就不只是开发工具问题，而会成为未来自动化组织、零人公司和数字劳动力的基础设施。
