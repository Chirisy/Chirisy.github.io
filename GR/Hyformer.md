# HyFormer 模型与源码解读

本文档基于 `Paper/Hyformer.pdf`、`train/` 与 `eval/` 目录源码整理，目标是把论文中的 HyFormer 思路映射到当前工程实现，并逐个解释主要结构的源码职责、输入输出和数据流。

## 1. 阅读范围

- 论文: `Paper/Hyformer.pdf`
- 训练入口: `train/train.py`
- 训练模型: `train/model.py`
- 训练数据: `train/dataset.py`
- 训练器: `train/trainer.py`
- 训练工具: `train/utils.py`
- 推理入口: `eval/infer.py`
- 推理模型与数据: `eval/model.py`, `eval/dataset.py`
- 运行脚本: `train/run.sh`

当前 `train/model.py` 与 `eval/model.py` 是镜像实现，`train/dataset.py` 与 `eval/dataset.py` 也是镜像实现。训练侧额外包含优化器、验证、保存 checkpoint 等逻辑；评估侧额外包含从 checkpoint 目录恢复配置、构造模型、加载权重、输出预测文件等逻辑。

## 2. 论文核心思想

论文提出 HyFormer，用于工业推荐/CTR 预估中的长序列建模和非序列特征交互。传统大模型推荐结构常见流程是先用 LONGER 或 Transformer 压缩用户行为序列，再把压缩后的序列 token 和非序列特征 token 输入 RankMixer 之类的交互模块。论文认为这种“两阶段、晚融合、单向”的方式限制了表达能力。

HyFormer 的核心做法是把长序列建模和特征交互放入同一个 backbone 中，层层交替执行两个动作:

1. Query Decoding: 用由非序列特征和序列摘要生成的 query token，对每条行为序列做 cross-attention，从序列 K/V 中抽取目标相关信息。
2. Query Boosting: 把 decoded query 和非序列 token 拼在一起，通过 RankMixer/MLP-Mixer 风格 token mixing 做跨 query、跨序列、跨非序列特征交互。

多层堆叠后，query token 不断带着更丰富的全局语义去重新询问长序列。论文还强调多序列场景不直接 merge 不同序列，而是在每个 HyFormer block 内对每条序列独立编码和解码，再在 query 层做跨序列混合。

## 3. 论文概念到源码的映射

| 论文概念 | 当前源码结构 | 说明 |
|---|---|---|
| Input Tokenization | `GroupNSTokenizer`, `RankMixerNSTokenizer` | 把 user/item 离散特征变成 NS token。 |
| Non-sequential tokens | `PCVRHyFormer.forward()` 中的 `user_ns`, `item_ns`, `user_dense_tok`, `item_dense_tok` | 离散和 dense 特征合并成 `ns_tokens`。 |
| Query Generation | `MultiSeqQueryGenerator` | 每条序列独立生成 `num_queries` 个 query token。 |
| Sequence Representation Encoding | `SwiGLUEncoder`, `TransformerEncoder`, `LongerEncoder` | 对每条行为序列做 layer-wise 编码。 |
| Query Decoding | `CrossAttention` | query token 对当前序列编码结果做 cross-attention。 |
| Query Boosting | `RankMixerBlock` | 拼接所有 decoded query 和 NS token 后做 token mixing 和 FFN。 |
| HyFormer layer | `MultiSeqHyFormerBlock` | 一个 block = sequence evolution + query decoding + query boosting。 |
| HyFormer stack | `PCVRHyFormer._run_multi_seq_blocks()` | 多层 `MultiSeqHyFormerBlock` 堆叠。 |
| Multi-sequence modeling | `seq_domains`, `seq_tokens_list`, `q_tokens_list` | 每个 domain 单独编码/解码，Boosting 时统一交互。 |
| BCE objective | `PCVRHyFormerRankingTrainer._train_step()` | `binary_cross_entropy_with_logits` 或 focal loss。 |

## 4. 数据结构与输入格式

### 4.1 `ModelInput`

源码位置: `train/model.py:11`, `eval/model.py:11`

`ModelInput` 是模型 forward 的统一输入容器:

- `user_int_feats`: 用户侧离散特征，形状大致为 `[B, total_user_int_dim]`
- `item_int_feats`: 物品侧离散特征，形状为 `[B, total_item_int_dim]`
- `user_dense_feats`: 用户侧 dense 特征，形状为 `[B, user_dense_dim]`
- `item_dense_feats`: 物品侧 dense 特征，当前数据集实现中通常为空张量 `[B, 0]`
- `seq_data`: 多序列字典，每个 domain 是 `[B, S, L]`，其中 `S` 是该序列的 side-info 特征数，`L` 是截断后的最大序列长度
- `seq_lens`: 每条序列的有效长度 `[B]`
- `seq_time_buckets`: 每条序列每个位置的时间桶 `[B, L]`

训练器和推理脚本都先把 batch dict 转成 `ModelInput`，再交给模型。

### 4.2 `FeatureSchema`

源码位置: `train/dataset.py:43`, `eval/dataset.py:43`

`FeatureSchema` 记录每个 feature id 在扁平 tensor 中的 `(offset, length)`。离散和 dense 特征都会先被展开到一个连续向量中，后续 tokenizer 通过 schema 知道每个 fid 对应哪段切片。

关键字段:

- `entries`: 按插入顺序保存 `(feature_id, offset, length)`
- `total_dim`: 展开后的总维度
- `_fid_to_entry`: fid 到 `(offset, length)` 的快速查表

### 4.3 `PCVRParquetDataset`

源码位置: `train/dataset.py:135`, `eval/dataset.py:135`

该类直接读取 Parquet 和 `schema.json`，把原始多列数据转换成模型 batch。主要步骤:

1. 读取 `schema.json`，构建用户离散、物品离散、用户 dense、多序列字段的 schema。
2. 枚举 Parquet row group，用 `row_group_range` 划分 train/valid。
3. 预分配 numpy buffer，减少每个 batch 的临时分配。
4. 读取 scalar/list 离散特征，缺失值和非正值统一置 0，0 也是 embedding padding id。
5. 检查离散 id 是否超出 vocab size，默认越界会被 clip 为 0。
6. 读取 dense list<float>，补齐到 schema 指定维度。
7. 读取每个序列 domain 的 side-info 特征，写入 `[B, S, L]`。
8. 根据 `timestamp` 和序列时间戳生成 `time_bucket`，供模型中的 `time_embedding` 使用。

### 4.4 `get_pcvr_data`

源码位置: `train/dataset.py:672`, `eval/dataset.py:672`

`get_pcvr_data()` 负责构造训练和验证 DataLoader:

- 按 row group 切分训练集和验证集。
- 训练集使用 `IterableDataset`，支持多 worker 和 shuffle buffer。
- 验证集关闭 shuffle，worker 数为 0。
- 返回 `(train_loader, valid_loader, train_dataset)`，其中 `train_dataset` 用来提供 schema 信息给模型构造过程。

## 5. 模型基础组件源码解读

### 5.1 `RotaryEmbedding`

源码位置: `train/model.py:26`, `eval/model.py:26`

该结构实现 RoPE 的 cos/sin 缓存。初始化时计算 `inv_freq`，再在 `_build_cache()` 中预生成最大长度的 cos/sin。forward 时按当前序列长度切片返回。

源码特点:

- `register_buffer(..., persistent=False)` 表示这些缓存不写入 checkpoint。
- 缓存一次构建，不在 forward 中动态扩容，有利于 `torch.compile`。
- 当前只在 `use_rope=True` 时由 `PCVRHyFormer` 创建。

### 5.2 `rotate_half` 与 `apply_rope_to_tensor`

源码位置: `train/model.py:67`, `train/model.py:74`

`rotate_half()` 把最后一维拆成两半并做旋转；`apply_rope_to_tensor()` 将 RoPE 应用到多头注意力的 Q 或 K 上。

输入输出约定:

- 输入 `x`: `[B, num_heads, L, head_dim]`
- `cos/sin`: `[1, L, head_dim]` 或 `[B, L, head_dim]`
- 输出仍是 `[B, num_heads, L, head_dim]`

### 5.3 `SwiGLU`

源码位置: `train/model.py:100`

这是轻量 FFN 单元。先把输入投影到 `2 * hidden_dim`，拆成 `x1, x2`，计算 `x1 * silu(x2)`，再投回 `d_model`。论文中 decoder-style lightweight encoding 使用 SwiGLU，这里对应 `SwiGLUEncoder` 的核心算子。

### 5.4 `RoPEMultiheadAttention`

源码位置: `train/model.py:117`

这是工程里统一使用的多头注意力实现，支持 self-attention 和 cross-attention。

主要流程:

1. 线性投影得到 Q/K/V。
2. reshape 到 `[B, num_heads, L, head_dim]`。
3. 可选对 K 和 Q 加 RoPE。`rope_on_q=False` 时只对 K 加，适合 Query Decoding 的 cross-attention。
4. 把 `key_padding_mask` 转成 SDPA 使用的 bool attention mask。
5. 调用 `F.scaled_dot_product_attention`。
6. 对全 padding 产生的 NaN 做 `torch.nan_to_num`。
7. 使用 `W_g` 生成门控，`out * sigmoid(G)` 后接 `W_o`。

源码中 `W_g.weight` 初始化为 0，`W_g.bias` 初始化为 1。也就是说训练初期门控值约为 `sigmoid(1)`，不是完全关闭注意力输出。

### 5.5 `CrossAttention`

源码位置: `train/model.py:244`

对应论文中的 Query Decoding。输入 query 是全局 query token，key/value 是某条序列的当前层表示。

实现细节:

- 默认 `ln_mode='pre'`，先对 query 和 key/value 分别做 LayerNorm。
- 内部使用 `RoPEMultiheadAttention(..., rope_on_q=False)`，所以 RoPE 只作用在序列 K 侧，query 不加序列位置。
- 输出和 query residual 相加。

该模块的效果是让非序列特征生成的 query 直接读取长序列信息。

### 5.6 `RankMixerBlock`

源码位置: `train/model.py:315`

对应论文中的 Query Boosting。输入是 `[decoded_qs, ns_tokens]` 拼接后的 token 序列，形状 `[B, T, D]`。

三种模式:

- `full`: 执行 token mixing + per-token FFN，要求 `D % T == 0`。
- `ffn_only`: 不做 token mixing，只做 per-token FFN。
- `none`: 直接返回输入。

`full` 模式的 token mixing 与论文中的子空间交换一致:

1. `[B, T, D] -> [B, T, T, D/T]`
2. transpose token 维和子空间维
3. reshape 回 `[B, T, D]`

随后经过 LayerNorm、Linear、GELU、Dropout、Linear，再与原始 Q residual 相加，最后做 post LayerNorm。

### 5.7 `MultiSeqQueryGenerator`

源码位置: `train/model.py:415`

对应论文的 Query Generation。该模块为每条序列独立生成 query token。

输入:

- `ns_tokens`: `[B, M, D]`
- `seq_tokens_list`: S 条序列，每条 `[B, L_i, D]`
- `seq_padding_masks`: S 条序列，每条 `[B, L_i]`

流程:

1. 将 NS token flatten 为 `[B, M*D]`。
2. 对每条序列用 mask 做 mean pooling，得到 `seq_pooled`。
3. 拼接 `global_info = Concat(NS_flat, seq_pooled)`。
4. 对每条序列使用独立的 `num_queries` 个 FFN，生成 `[B, Nq, D]`。

论文说第一层通过 MLP 生成 queries，后续层复用上一层更新后的 query。源码也符合这个逻辑: query 只在 `PCVRHyFormer.forward()` 进入 block 前生成一次，之后每个 `MultiSeqHyFormerBlock` 返回更新后的 `next_q_list`。

## 6. 序列编码器源码解读

论文 3.4.1 支持三种序列表示策略，源码正好对应三个 encoder。

### 6.1 `SwiGLUEncoder`

源码位置: `train/model.py:502`

轻量 attention-free 编码器。结构是:

```text
x -> LayerNorm -> SwiGLU -> Dropout -> residual add
```

优点是延迟低，适合 latency-critical 场景；缺点是序列内部 token 间没有显式 attention 交互。

### 6.2 `TransformerEncoder`

源码位置: `train/model.py:544`

标准 Pre-LN Transformer encoder layer:

1. `LayerNorm`
2. self-attention，可选 RoPE
3. residual
4. `LayerNorm`
5. FFN
6. residual

这是论文中的 full Transformer encoding，表达能力最强，但复杂度最高，长序列时成本是主要问题。

### 6.3 `LongerEncoder`

源码位置: `train/model.py:616`

对应论文中 LONGER-style efficient encoding。它根据当前序列长度自适应选择两种模式:

- 如果 `L > top_k`: 从每个样本中取最近的 `top_k` 个有效 token 作为 query，原长序列作为 K/V 做 cross-attention，复杂度约为 `top_k * L`。
- 如果 `L <= top_k`: 后续层已经被压缩到短序列，执行 self-attention。

`_gather_top_k()` 会按 `key_padding_mask` 计算有效长度，选择最近的有效 token，并生成新的 padding mask。若启用 RoPE，还会为 gather 出来的 query token 从原序列位置中取对应的 `q_rope_cos/sin`。

### 6.4 `create_sequence_encoder`

源码位置: `train/model.py:811`

简单工厂函数，根据 `seq_encoder_type` 返回 `SwiGLUEncoder`、`TransformerEncoder` 或 `LongerEncoder`。该参数由 `train/train.py` 的 `--seq_encoder_type` 控制。

## 7. HyFormer Block 源码解读

### 7.1 `MultiSeqHyFormerBlock`

源码位置: `train/model.py:850`

这是单个 HyFormer layer 的工程实现。它处理多序列场景。

构造阶段:

- 为每条序列创建一个独立 sequence encoder。
- 为每条序列创建一个独立 CrossAttention。
- 创建一个共享 `RankMixerBlock`，token 数为 `num_queries * num_sequences + num_ns`。

forward 阶段:

1. Sequence Evolution: 每条序列独立调用对应 encoder，得到 `next_seqs` 和 `next_masks`。
2. Query Decoding: 每条序列使用自己的 query 对该序列的 `next_seq_i` 做 cross-attention，得到 `decoded_q_i`。
3. Token Fusion: 拼接所有序列的 `decoded_qs` 和共享 `ns_tokens`。
4. Query Boosting: 调用 `RankMixerBlock` 做跨 query、跨序列、跨 NS token 交互。
5. Split: 把 boosted token 前半部分按序列拆回 `next_q_list`，剩余部分作为下一层的 `next_ns`。

这个结构和论文 3.6、3.7 的设计基本一致: 不合并原始多序列，而是在 query 层完成跨序列交互。

## 8. NS Tokenizer 源码解读

### 8.1 `GroupNSTokenizer`

源码位置: `train/model.py:988`

语义分组 tokenizer。输入是 feature specs 和 groups，每个 group 对应一个 NS token。

处理逻辑:

- 每个 fid 对应一个 embedding table。
- 单值特征直接 embedding lookup。
- 多值特征先 lookup，再对非 0 位置做 mean pooling。
- 如果 `emb_skip_threshold` 触发过滤，则该 fid 输出全 0 向量。
- 一个 group 内多个 fid 的 embedding 拼接后，经 `Linear + LayerNorm + SiLU` 得到一个 token。

`train/ns_groups.json` 给出了示例分组: user 侧 7 组，item 侧 4 组。该文件中的值是 fid，训练入口会转换成 schema entry 的 positional index。

### 8.2 `RankMixerNSTokenizer`

源码位置: `train/model.py:1070`

自动切分 tokenizer。它先按 groups 顺序把所有 fid embedding 展平成一个长向量，再均匀切成 `num_ns_tokens` 段，每段通过 `Linear + LayerNorm + SiLU` 投影成一个 NS token。

这对应论文中提到的 auto-split 方式，也对应当前 `train/run.sh` 默认配置:

```text
--ns_tokenizer_type rankmixer
--user_ns_tokens 5
--item_ns_tokens 2
--num_queries 2
--ns_groups_json ""
```

当前默认配置不使用 `ns_groups.json`，但仍会用 fallback 的 singleton groups 提供 fid 顺序，再由 RankMixer tokenizer flatten/split。

## 9. `PCVRHyFormer` 主模型源码解读

源码位置: `train/model.py:1192`

### 9.1 初始化

`PCVRHyFormer.__init__()` 负责把 schema 和超参数组装成完整模型。

主要参数:

- `user_int_feature_specs`, `item_int_feature_specs`: 离散特征 schema。
- `user_dense_dim`, `item_dense_dim`: dense 维度。
- `seq_vocab_sizes`: 每条序列每个 side-info fid 的 vocab size。
- `user_ns_groups`, `item_ns_groups`: NS 分组。
- `d_model`: token hidden size。
- `emb_dim`: 每个 embedding table 的输出维度。
- `num_queries`: 每条序列生成多少 query token。
- `num_hyformer_blocks`: 堆叠多少层 HyFormer block。
- `seq_encoder_type`: `swiglu`, `transformer`, `longer`。
- `rank_mixer_mode`: `full`, `ffn_only`, `none`。
- `use_rope`: 是否启用 RoPE。

初始化过程:

1. 创建 user/item NS tokenizer。
2. dense 特征如果存在，则用 `Linear + LayerNorm` 投影成一个 NS token。
3. 计算 `num_ns`。
4. 在 `rank_mixer_mode='full'` 时检查 `d_model % T == 0`，其中 `T = num_queries * num_sequences + num_ns`。
5. 为每个序列 domain 创建 embedding tables 和投影层。
6. 创建可选 time bucket embedding。
7. 创建 `MultiSeqQueryGenerator`。
8. 创建 `num_hyformer_blocks` 个 `MultiSeqHyFormerBlock`。
9. 创建可选 RoPE。
10. 创建输出投影和最终 classifier。

### 9.2 序列 embedding

源码位置: `train/model.py:1544`

`_embed_seq_domain()` 把某个序列 domain 的 `[B, S, L]` side-info id 转成 `[B, L, D]` token 表示。

步骤:

1. 对每个 side-info feature 做 embedding lookup。
2. 对高基数 id 特征在训练时额外 dropout，dropout rate 是 `dropout_rate * 2`。
3. 将所有 side-info embedding 沿最后一维拼接。
4. 经 `Linear + LayerNorm + GELU` 投影到 `d_model`。
5. 如果启用 time bucket，将 `time_embedding(time_bucket_ids)` 加到 token 上。

### 9.3 padding mask

源码位置: `train/model.py:1576`

`_make_padding_mask()` 根据序列长度生成 `[B, L]` bool mask。`True` 表示 padding，后续 attention 会将其屏蔽。

### 9.4 多层 block 执行

源码位置: `train/model.py:1584`

`_run_multi_seq_blocks()` 是 HyFormer stack:

1. 训练时对 q、ns、seq token 做 dropout。
2. 每层 block 前，如果启用 RoPE，则为每条当前序列长度计算 cos/sin。
3. 调用 block，得到更新后的 q、ns、seq、mask。
4. 最后一层后只拼接所有序列的 query token，展平后输入 `output_proj`。

注意: 当前代码的最终 readout 只使用 query token，不直接拼接最终 NS token。论文公式中 Query Boosting 会持续使用 NS tokens 做交互，源码确实在每层 block 中使用 NS tokens，但最后分类头只读 `curr_qs`。

### 9.5 forward

源码位置: `train/model.py:1634`

训练时 forward 链路:

```text
user/item int -> NS tokenizer
user/item dense -> dense projection token
concat -> ns_tokens
each seq domain -> _embed_seq_domain -> seq_tokens_list
seq_lens -> padding masks
MultiSeqQueryGenerator(ns_tokens, seq_tokens_list, masks) -> q_tokens_list
_run_multi_seq_blocks(q, ns, seq, masks) -> output
classifier(output) -> logits
```

输出是 raw logits，训练器使用 `binary_cross_entropy_with_logits`，因此 forward 不做 sigmoid。

### 9.6 predict

源码位置: `train/model.py:1677`

`predict()` 基本复用 forward 逻辑，但 `_run_multi_seq_blocks(..., apply_dropout=False)`，返回 `(logits, output_embedding)`。评估和推理都使用这个接口。

## 10. 训练入口与训练器源码解读

### 10.1 `train/train.py`

训练入口主要负责配置解析和模型装配。

关键结构:

- `build_feature_specs()`，源码位置 `train/train.py:27`: 把 `FeatureSchema` 和 per-position vocab size 合成 `(vocab_size, offset, length)`。
- `parse_args()`，源码位置 `train/train.py:41`: 定义数据、模型、loss、优化器、NS tokenizer 等参数。
- `main()`，源码位置 `train/train.py:208`: 完成数据加载、NS groups 加载、模型构造和 trainer 构造。

训练入口的模型装配顺序:

1. 从环境变量或 CLI 获取数据路径、checkpoint 路径、日志路径。
2. 解析 `seq_max_lens`。
3. 调用 `get_pcvr_data()` 读取数据。
4. 如果存在 `ns_groups_json`，把 fid 分组转换成 schema index 分组，否则每个 feature 单独成组。
5. 通过 schema 构造 user/item feature specs。
6. 组装 `model_args` 并创建 `PCVRHyFormer`。
7. 创建 `EarlyStopping` 和 `PCVRHyFormerRankingTrainer`。

### 10.2 `PCVRHyFormerRankingTrainer`

源码位置: `train/trainer.py:25`

虽然类名有 Ranking，但当前训练是 pointwise binary classification。

初始化逻辑:

- 如果模型提供 `get_sparse_params()`，则把 embedding 参数交给 Adagrad。
- 其他参数交给 AdamW。
- 保存 checkpoint sidecar 文件所需的 schema、ns_groups、train_config。

训练 step:

源码位置: `train/trainer.py:402`

```text
batch -> device
batch -> ModelInput
model(model_input) -> logits
loss = BCEWithLogits 或 focal loss
backward
clip_grad_norm
dense_optimizer.step
sparse_optimizer.step
```

验证:

- `evaluate()` 汇总所有验证 logits 和 labels。
- AUC 使用 sklearn 的 `roc_auc_score`。
- logloss 使用 `binary_cross_entropy_with_logits`。
- 如果验证输出存在 NaN，会过滤 NaN prediction 后再算指标。

checkpoint:

- `_handle_validation_result()` 会把新最佳模型保存到 `global_step*.best_model/model.pt`。
- `_write_sidecar_files()` 会复制 `schema.json`、`ns_groups.json`，并写入 `train_config.json`。
- 这些 sidecar 是 eval 阶段严格重建模型结构的依据。

### 10.3 `train/utils.py`

主要结构:

- `LogFormatter`: 日志带绝对时间和 elapsed time。
- `create_logger()`: 配置文件日志和控制台日志。
- `EarlyStopping`: 按 AUC 这类 higher-is-better 指标保存最佳模型。
- `set_seed()`: 设置 Python、NumPy、PyTorch、CUDA 随机种子。
- `sigmoid_focal_loss()`: focal loss 实现，基于 logits 计算。

## 11. 推理入口源码解读

### 11.1 `eval/infer.py` 的职责

推理脚本的核心原则是: 模型结构必须从 checkpoint sidecar 中恢复，而不是手写猜测。

关键函数:

- `load_train_config()`，源码位置 `eval/infer.py:106`: 从 checkpoint 目录读取 `train_config.json`。
- `resolve_model_cfg()`，源码位置 `eval/infer.py:125`: 从 train config 提取结构超参数，缺失时使用 fallback。
- `build_model()`，源码位置 `eval/infer.py:162`: 使用 schema、NS groups 和 model cfg 重建 `PCVRHyFormer`。
- `load_model_state_strict()`，源码位置 `eval/infer.py:238`: strict 模式加载权重，结构不匹配则直接失败。
- `_batch_to_model_input()`，源码位置 `eval/infer.py:271`: 将 eval dataset batch 转为 `ModelInput`。
- `main()`，源码位置 `eval/infer.py:306`: 完成加载数据、构造模型、加载 checkpoint、循环预测、输出 `predictions.json`。

### 11.2 推理数据流

```text
MODEL_OUTPUT_PATH -> model.pt / train_config.json / schema.json / ns_groups.json
EVAL_DATA_PATH -> test parquet
PCVRParquetDataset -> batch
_batch_to_model_input -> ModelInput
model.predict -> logits
sigmoid(logits) -> probability
user_id -> probability dict
write EVAL_RESULT_PATH/predictions.json
```

### 11.3 eval 与 train 的一致性

`eval/model.py` 与 `train/model.py` 应保持结构一致，否则 strict load 可能 shape mismatch。`eval/infer.py` 的 fallback 配置必须与 `train/train.py` 默认参数保持一致，源码注释也明确提醒这一点。

## 12. 当前实现与论文的对应和差异

### 已实现且与论文高度对应

- Query Generation: `MultiSeqQueryGenerator`
- Query Decoding: `CrossAttention`
- Query Boosting: `RankMixerBlock`
- 多序列独立编码和解码: `MultiSeqHyFormerBlock`
- 三类 sequence encoder: `SwiGLUEncoder`, `TransformerEncoder`, `LongerEncoder`
- NS tokenization 的语义分组和自动切分: `GroupNSTokenizer`, `RankMixerNSTokenizer`
- 多层 query 复用和逐层增强: `_run_multi_seq_blocks()`

### 工程实现中的简化

- 论文提到 GPU Pooling for Long-Sequence，当前代码没有专门的 GPU pooling 自定义算子。
- 论文提到 Asynchronous AllReduce，当前单机训练器没有异步 allreduce 逻辑。
- 论文中的线上部署 KV cache/M-Falcon 相关服务优化没有在该工程中实现。
- 当前最终 readout 只使用 query token，不直接读最终 NS token。
- 当前 dataset 只显式实现 user dense，item dense schema 初始化为空，但模型接口保留了 item dense token。
- 当前 `train/run.sh` 默认使用 RankMixer NS tokenizer，不使用 `ns_groups.json` 的语义分组。

## 13. 关键张量形状速查

| 名称 | 形状 | 来源 |
|---|---|---|
| `user_int_feats` | `[B, total_user_int_dim]` | dataset |
| `item_int_feats` | `[B, total_item_int_dim]` | dataset |
| `user_dense_feats` | `[B, user_dense_dim]` | dataset |
| `seq_data[domain]` | `[B, S_domain, L_domain]` | dataset |
| `seq_lens[domain]` | `[B]` | dataset |
| `seq_time_buckets[domain]` | `[B, L_domain]` | dataset |
| `ns_tokens` | `[B, num_ns, D]` | tokenizer + dense projection |
| `seq_tokens` | `[B, L_domain, D]` | `_embed_seq_domain()` |
| `q_tokens` | `[B, num_queries, D]` per sequence | `MultiSeqQueryGenerator` |
| `combined` | `[B, num_queries*num_sequences + num_ns, D]` | `MultiSeqHyFormerBlock` |
| `output` | `[B, D]` | `_run_multi_seq_blocks()` |
| `logits` | `[B, action_num]` | classifier |

## 14. 一句话总结

这份代码实现的是一个面向 PCVR/CTR 的 HyFormer baseline: 它把 user/item/dense 非序列特征编码成 NS tokens，为每条行为序列生成 query tokens，再通过多层“序列编码 + query cross-attention + RankMixer boosting”不断融合长序列和非序列信息，最终用 query 表示完成二分类 logits 预测。相比论文完整工业系统，当前工程更偏模型结构和离线训练/评估实现，未包含 GPU pooling、异步 allreduce、线上 serving 优化等生产级组件。
