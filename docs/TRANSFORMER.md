# Transformer 架构详解：从论文原文到实现细节

本文以一手资料为准整理，凡是论文原文、权威实现和常见解读存在差别的地方，都明确标注出处，不把二手博客的结论当作事实。

## 0. 资料来源与可信度分级

| 级别 | 来源 | 用途 |
| --- | --- | --- |
| 一手 | Vaswani et al., *Attention Is All You Need*, arXiv:1706.03762v7（2017-06-12 首发，2023-08 最后修订），全文经 ar5iv HTML 版核对 | 结构、公式、超参数、训练配置、实验结果、复杂度表的唯一权威依据 |
| 一手 | Harvard NLP, *The Annotated Transformer*（nlp.seas.harvard.edu/annotated-transformer） | 实现层面的张量形状、掩码构造、teacher forcing、与论文文字不一致的细节 |
| 一手 | Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv:2104.09864v5 | RoPE 的公式、相对位置性质、长程衰减 |
| 二手（仅交叉核对） | 多源检索汇总 | 用于发现流传说法，其中与一手冲突者已在第 14 节纠正 |

下文标记约定：**[论文]** 表示原文明确写出；**[实现]** 表示哈佛注解中的具体代码行为；**[解读]** 表示学界常见解释，论文未直接论证。

---

## 1. 它解决了什么问题

2017 年之前，序列转换（如机器翻译）的主流是 encoder–decoder + RNN/CNN。RNN 的递归结构把第 t 步的计算绑死在第 t−1 步上，无法并行；两个相距很远的 token 之间信息要穿过整条链，路径长为 O(n)。

Transformer 的主张是：**丢掉 recurrence 和 convolution，只用 attention**。这一步带来的直接后果是：一层内所有位置同时计算（顺序操作数 O(1)），任意两个位置直接相连（最大路径长 O(1)）。

注意它的出身是**机器翻译**，所以原始论文是完整的 encoder–decoder 结构，不是今天大家熟悉的 decoder-only 大模型。

---

## 2. 全景结构

```
源序列                                          目标序列（右移一位）
  │                                                    │
  ▼                                                    ▼
Tokenize → Embedding×√d → +PE              Tokenize → Embedding×√d → +PE
  │                                                    │
┌─┴────────────── Encoder × N ──────────────┐   ┌──────┴── Decoder × N ──────────────┐
│  ① 多头自注意力 (Q=K=V)                    │   │  ① 掩码多头自注意力 (Q=K=V)         │
│  ② Add & Norm                             │   │  ② Add & Norm                      │
│  ③ 逐位置 FFN                              │   │  ③ 交叉注意力 (Q=dec, K/V=enc)      │
│  ④ Add & Norm                             │   │  ④ Add & Norm                      │
└──────────────────┬─────────────────────────┘   │  ⑤ 逐位置 FFN                      │
                   │  Z ∈ R^{n×d_model}          │  ⑥ Add & Norm                      │
                   └──────────────► 交叉注意力的 K/V └──────────────┬─────────────────┘
                                                                    ▼
                                                        Linear → Softmax → 下一 token 概率
```

**关键维度，全部来自论文 Table 3 [论文]：**

| 模型 | N | d_model | d_ff | h | d_k | d_v | P_drop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | 6 | 512 | 2048 | 8 | 64 | 64 | 0.1 |
| big | 6 | 1024 | 4096 | 16 | 64 | 64 | 0.3 |

big 在英法任务上例外地使用 P_drop = 0.1。**所有子层和嵌入层的输出维度都必须是 d_model**，这是残差连接的要求（见第 6 节）。

---

## 3. 输入环节：从 token 到向量

### 3.1 分词

[论文] 英德任务用 BPE，共享源目标词表约 37000；英法用 32000 的 word-piece 词表。

### 3.2 词嵌入与 √d_model 缩放

[论文] 每个 token 查表得到 d_model 维向量，`Embedding ∈ R^{V×d_model}`。[论文] 原文明确写：

> In the embedding layers, we multiply those weights by √d_model.

[解读] 常见解释是：正弦位置编码每个位置的方差约 0.5，而初始化后的 embedding 数值尺度不同，乘 √d_model 让两者量级匹配，避免位置信号被淹没或反过来压过语义。**论文本身只给出操作，没有论证原因**，所以这条属于推断而非定论。

[实现] 哈佛代码：`return self.lut(x) * math.sqrt(self.d_model)`。

### 3.3 位置编码

**为什么必须有**：自注意力是置换不变的——把输入 token 的顺序打乱，输出作为集合完全相同。语言有顺序，所以必须显式注入位置信息。

[论文] 原始方案是固定的正弦/余弦函数：

$$
PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \qquad
PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

其中 pos 是位置，i 是维度对索引（i = 0 … d_model/2−1）。

三个要点：

1. **多频率指纹**。i 小时分母小、振荡快，i 大时振荡慢；d_model 个维度合起来给每个位置一个唯一的向量。
2. **相对位置可学习**。[论文] 原文给出的理由是：对任意固定偏移 k，`PE(pos+k)` 可以表示成 `PE(pos)` 的线性函数——这是三角函数和角公式的直接结果。
3. **可外推到更长序列**。它是确定性函数而非查表，理论上不受训练时最大长度限制（实际外推效果有限，属于实践问题）。

[实现] 哈佛代码把 PE 预计算成 shape `(1, max_len, d_model)` 的 buffer，`register_buffer` 注册、`requires_grad_(False)`，不参与训练；`max_len=5000`。加法顺序是 `x = embedding + PE`，**之后**再 dropout。

[实现细节] 论文说 embedding 和 PE 相加后 dropout，哈佛代码把 dropout 放在 `PositionalEncoding.forward` 内，等价。

---

## 4. 核心：注意力

### 4.1 Q、K、V 的含义

把每个位置的向量投影成三个角色：

- **Query（查询）**：我在找什么信息
- **Key（键）**：我提供什么信息的索引
- **Value（值）**：我实际提供的内容

[解读] 可以类比数据库检索：Query 是查询条件，Key 用于匹配，Value 是返回内容。但注意注意力的匹配是“软”的——所有位置都按权重贡献，不是硬选取。

### 4.2 缩放点积注意力

[论文] 公式 (1)：

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right) V
$$

逐步展开（设 Q ∈ R^{n×d_k}，K ∈ R^{m×d_k}，V ∈ R^{m×d_v}）：

1. `S = QKᵀ` → R^{n×m}。S[i][j] = 第 i 个 query 与第 j 个 key 的内积，即“i 该从 j 取多少信息”。
2. `S = S / √d_k`。
3. （可选）掩码：非法位置的分数置 −∞ [论文] / −1e9 [实现]。
4. `A = softmax(S)`，沿最后一维（key 维）归一化，每行和为 1。
5. `Output = AV` → R^{n×d_v}。每个位置的输出是所有位置 value 的**凸组合**。

这就是注意力的本质：**内容相关的动态信息聚合**。RNN 用固定隐状态一步步传递，注意力让每个位置直接从全序列加权取信息。

### 4.3 为什么除以 √d_k（最容易讲错的一处）

[论文] 脚注给出严格理由：假设 q、k 的各分量独立、均值 0、方差 1，则内积

$$
q \cdot k = \sum_{i=1}^{d_k} q_i k_i
$$

的方差是 d_k。d_k 越大，点积幅值越大，softmax 被推进**梯度极小的饱和区**，训练变慢甚至停滞。

[论文] 原文还给出实验观察：

> While for small values of d_k the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of d_k.

也就是说，不缩放时，d_k 大的情况下加性注意力反而更好——缩放因子是点积注意力能work的必要条件，不是可有可无的装饰。

[解读] 所以准确说法是“控制方差、防止 softmax 饱和导致梯度消失”，而不是笼统的“防止数值溢出”。

### 4.4 掩码

两类掩码，作用不同：

- **padding mask**：屏蔽填充位，让模型不把 `<pad>` 当信息。源序列和目标序列都要用。
- **causal mask（因果掩码）**：只用于 decoder 的自注意力，屏蔽“未来”位置。

[实现] 哈佛代码构造因果掩码：

```python
subsequent_mask = torch.triu(torch.ones((1, size, size)), diagonal=1)
return subsequent_mask == 0
```

得到一个下三角布尔矩阵（含对角线），`True` = 允许关注。

[实现] 掩码在 softmax **之前**生效：

```python
scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
scores = scores.masked_fill(mask == 0, -1e9)
p_attn = scores.softmax(dim=-1)
```

论文写的是 −∞，代码用 −1e9 作为数值近似，softmax 后该位置权重近似为 0。注意 attention dropout 施加在 `p_attn` 上，即 softmax **之后**。

### 4.5 多头注意力

[论文] 单头只能做一次“软选择”，多头让模型在多个表示子空间并行关注：

$$
\mathrm{head}_i = \mathrm{Attention}(QW_i^{Q}, KW_i^{K}, VW_i^{V})
$$
$$
\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h)\, W^{O}
$$

投影矩阵维度 [论文]：

$$
W_i^{Q}\in\mathbb{R}^{d_{model}\times d_k},\;
W_i^{K}\in\mathbb{R}^{d_{model}\times d_k},\;
W_i^{V}\in\mathbb{R}^{d_{model}\times d_v},\;
W^{O}\in\mathbb{R}^{h d_v\times d_{model}}
$$

base 模型：h = 8，d_k = d_v = d_model/h = 64。

**为什么计算量没有爆炸**：每头维度缩小到 1/h，8 个头合起来的计算量与“单头用完整 512 维”基本持平。这是近乎免费的多视角。

[论文] 原文给出动机：

> Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

[解读] 实践观察到不同头会分化出不同功能（语法依存、指代、局部/长程模式），但这是**观察性结论，不是架构保证**。

[实现] 张量形状的完整变化（B = batch，L = 序列长）：

```
(B, L, 512)  ──线性投影──►  (B, L, 512)
             ──view─────►  (B, L, 8, 64)
             ──transpose►  (B, 8, L, 64)
             ──attention─► scores (B, 8, L, L), context (B, 8, L, 64)
             ──transpose─► (B, L, 8, 64)
             ──view──────► (B, L, 512)
             ──W^O───────► (B, L, 512)
```

`W^O` 不是多余的：它把 h 个子空间的结果融合回统一表示。去掉它会破坏表达能力。

### 4.6 三种注意力的对照

| 位置 | Q 来源 | K/V 来源 | 掩码 | 作用 |
| --- | --- | --- | --- | --- |
| Encoder 自注意力 | encoder 上一层 | encoder 上一层 | 无（双向） | 源序列内部建模 |
| Decoder 掩码自注意力 | decoder 上一层 | decoder 上一层 | causal + padding | 已生成前缀内部建模 |
| Encoder–Decoder 交叉注意力 | decoder 当前层 | **encoder 最终输出 Z** | 仅 padding（无因果掩码） | 源→目标对齐与检索 |

第三行最容易被记反：**Q 来自解码器，K/V 来自编码器**。注意力矩阵形状是 `m×n`（目标长 × 源长）。它是源序列信息进入解码器的唯一通道。

---

## 5. 逐位置前馈网络 FFN

[论文] 公式 (2)：

$$
\mathrm{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

两个线性层 + 中间 ReLU；`W_1 ∈ R^{d_model×d_ff}`，`W_2 ∈ R^{d_ff×d_model}`，d_ff = 2048 = 4×d_model。

**“逐位置”的含义**：同一个 FFN 独立作用于每个位置，位置之间不交互。序列内部的信息混合完全由注意力完成，FFN 只做单点的非线性变换。实现上等价于对 `n×d_model` 做两次矩阵乘。

**参数量占比**：每层 attention 约 4·d_model² = 1.05M，FFN 为 2·d_model·d_ff = 2.10M，FFN 占每层参数的 2/3——它是参数的主要承载者。

[解读] 一种有影响力的解释把 FFN 看作 key–value 记忆：第一层神经元充当模式匹配的“键”，第二层对应列是被写回的“值”，事实性知识主要存在 FFN 中。这是主流解释框架，不是严格定理。

[实现] 哈佛代码在 ReLU 之后、W_2 之前还有一层 dropout。

---

## 6. 残差连接与层归一化

[论文] 每个子层统一封装：

$$
y = \mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))
$$

这叫 **Post-Norm**（先残差相加，再归一化）。

**残差的两个作用**：

1. 梯度有恒等通路，可无损回传，使深层堆叠可训练。
2. **结构约束**：因为要做 `x + Sublayer(x)`，两者维度必须相同——这就是 d_model 贯穿全模型（含嵌入、每个子层输出）的原因。

**LayerNorm vs BatchNorm**：LayerNorm 在单个样本的单个位置上、沿特征维做标准化（减均值、除标准差，再乘 γ 加 β）。它不依赖 batch 统计，因此不受 batch 大小和变长序列影响，训练与推理行为一致。

### 一处必须澄清的差异

[论文] 写的是 Post-Norm；但 [实现] 哈佛注解的 `SublayerConnection` 实际是 **Pre-Norm**：

```python
return x + self.dropout(sublayer(self.norm(x)))   # x + Dropout(Sublayer(LayerNorm(x)))
```

代码注释也承认：

> Note for code simplicity the norm is first as opposed to last.

同时 encoder/decoder 栈末尾再补一个 final LayerNorm。现代大模型普遍采用 Pre-Norm，因为深层训练更稳定。**所以“Transformer 用 Post-Norm”只对原论文成立，不对今天的实际实现成立。**

---

## 7. 编码器：完整流程

单层编码器：

```
x (n×d_model)
 → 多头自注意力（Q=K=V=x，无掩码，每个位置可见全序列）
 → Add & Norm:  x1 = LN(x + MHA(x))
 → FFN
 → Add & Norm:  x2 = LN(x1 + FFN(x1))
```

N = 6 层串联，每层输入输出维度不变，只有表征逐层抽象（低层偏词形和局部句法，高层偏语义和篇章）。最终输出 Z ∈ R^{n×d_model}，供解码器交叉注意力使用。

**编码器是双向的**：第 i 个位置的表示融合了整个源序列。这正是 BERT 类掩码语言模型能直接用于理解任务的结构基础。

---

## 8. 解码器：完整流程

单层解码器有三个子层：

```
y (m×d_model)
 → ① 掩码多头自注意力（causal mask）
 → Add & Norm
 → ② 交叉注意力（Q=y, K/V=encoder 输出 Z）
 → Add & Norm
 → ③ 逐位置 FFN
 → Add & Norm
```

### 8.1 因果掩码 + 右移：如何做到“不偷看未来”

[论文] decoder 的输入是目标序列**右移一位**（shifted right）。[实现] 哈佛代码：

```python
self.tgt = tgt[:, :-1]     # decoder 输入：[<s>, y1, y2, ...]
self.tgt_y = tgt[:, 1:]    # 标签：      [y1, y2, ..., </s>]
```

两者配合保证自回归性：

- 训练时可以把整句答案**并行**喂进去；
- 因果掩码让第 i 个位置在计算时只能看到 i 及之前的位置，无法看到 y_{i+1}；
- 因此每个位置的损失都只基于已知前缀，和真实推理时的条件完全一致。

### 8.2 训练 vs 推理

| | 训练 | 推理 |
| --- | --- | --- |
| 输入 | 整句右移目标，teacher forcing（前文是真实 token） | 只有 `<s>`，之后每步喂自己上一步的预测 |
| 并行性 | 所有位置一次前向，完全并行 | 逐 token 串行生成 |
| 取输出 | 所有位置的 logits 都算损失 | 只取最后一个位置的 logits |
| 优化 | — | KV Cache：缓存历史 K/V，避免每步重算全前缀 |

**训练时并行不是因为模型“同时生成所有 token”**，而是因为掩码保证每个位置的计算互不泄漏未来——这个区别常被误解。

[实现] 哈佛的贪心解码：

```python
memory = model.encode(src, src_mask)
ys = [[start_symbol]]
for i in range(max_len - 1):
    out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)))
    prob = model.generator(out[:, -1])
    next_word = argmax(prob)
    ys = concat([ys, next_word])
```

---

## 9. 输出层

解码器栈输出 H ∈ R^{m×d_model} → 线性层映射到词表维 V → softmax 得下一 token 概率。

[论文] **权重共享**：两个 embedding 层和 pre-softmax 线性变换共享同一个权重矩阵。这减少参数量并改善泛化。[论文] embedding 层的权重乘 √d_model。

[实现差异] 哈佛代码默认**没有**共享：`Embeddings` 和 `Generator` 是独立参数；共享代码写在 `if False:` 块里，默认不启用。这是教学实现为了清晰做的简化，不是论文原样。

**label smoothing** [论文]：训练时用 ε_ls = 0.1。原文的观察很值得注意：

> This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.

即困惑度变差但翻译质量变好——说明困惑度不是唯一指标。

[实现] `KLDivLoss`，正确词概率为 `1−smoothing`，其余非 padding 词平分 `smoothing`（即 `smoothing/(size−2)`），padding 位置整行置 0。

---

## 10. 完整前向数据流（张量形状）

记 n = 源长，m = 目标长，d = d_model，V = 词表，h = 头数，d_k = d/h：

| 步骤 | 形状 |
| --- | --- |
| 源/目标 embedding + PE | n×d / m×d |
| Q/K/V 投影 | n×d（单头视角 d_k） |
| 自注意力分数 QKᵀ | n×n |
| 交叉注意力分数 | m×n |
| 注意力输出 AV | n×d_v |
| 多头拼接 | n×h·d_v = n×d |
| FFN 中间层 | n×d_ff |
| decoder 输出 H | m×d |
| 输出 logits | m×V |

---

## 11. 训练配置（论文全部关键数字）

[论文] 均来自原论文第 5 节：

- **数据**：WMT 2014 英德约 450 万句对，BPE 共享词表约 37000；英法 3600 万句，32000 word-piece 词表。
- **批大小**：每个 batch 约含 25000 个源 token 和 25000 个目标 token（按句长近似分桶）。
- **硬件与时间**：单机 8 张 NVIDIA P100。base 每步约 0.4 秒，10 万步 = 12 小时；big 每步 1.0 秒，30 万步 = 3.5 天。
- **优化器**：Adam，β₁ = 0.9，β₂ = 0.98，ε = 1e−9。
- **学习率调度** [论文] 公式 (3)：

$$
lrate = d_{model}^{-0.5} \cdot \min\left(step\_num^{-0.5},\; step\_num \cdot warmup\_steps^{-1.5}\right)
$$

即前 4000 步（warmup_steps = 4000）线性上升，之后按步数的平方根倒数下降。

[实现] 代码对 `step == 0` 做保护（置为 1），避免 0 的负次幂。

- **Dropout**：base 0.1，big 0.3（英法 big 例外用 0.1）。施加在每个子层输出（残差相加之前）以及 embedding+PE 的和上。
- **Label smoothing**：ε = 0.1。
- **参数初始化** [实现]：维度大于 1 的参数用 Xavier/Glorot uniform。

---

## 12. 它为什么赢：复杂度与实测

[论文] Table 1 原文（n 序列长，d 表示维度，k 卷积核，r 受限注意力邻域）：

| 层类型 | 每层复杂度 | 顺序操作数 | 最大路径长 |
| --- | --- | --- | --- |
| Self-Attention | O(n²·d) | **O(1)** | **O(1)** |
| Recurrent | O(n·d²) | O(n) | O(n) |
| Convolutional | O(k·n·d²) | O(1) | O(log_k n) |
| Self-Attention (restricted) | O(r·n·d) | O(1) | O(n/r) |

**三个结论**：

1. **并行性**：顺序操作数从 RNN 的 O(n) 降到 O(1)，这是几小时而不是几周训完的根因。
2. **长程依赖**：任意两 token 最大路径 O(1)，梯度无需穿过长链；RNN 是 O(n)。
3. **代价**：O(n²) 对长序列不友好。论文自己把“受限/局部注意力”列为未来工作——后来的稀疏注意力、线性注意力、FlashAttention 都源于这条。

[论文] 结果：英德 big 28.4 BLEU（比此前包括集成模型的最好结果高 2 BLEU 以上）；英法单模型 41.8 BLEU。base 模型分别为 27.3 / 38.1。

[严谨性提示] 论文自身存在一处内部不一致：摘要和 Table 2 写英法 big 为 **41.8**，但正文 6.1 节写 **41.0**。引用时应以摘要/表格的 41.8 为准并知晓此差异。

---

## 13. 从原始论文到今天的主流实现

原始结构是“教材原型”，今天的大模型（GPT / LLaMA 系）在细节上已明显演化：

| 环节 | 原论文 2017 | 现代主流 |
| --- | --- | --- |
| 整体 | Encoder–Decoder | 多为 **Decoder-only**，去掉交叉注意力 |
| 归一化位置 | Post-Norm | **Pre-Norm**，深层更稳定 |
| 归一化类型 | LayerNorm | **RMSNorm**（去掉均值中心化，只做缩放） |
| 激活 | ReLU | **SwiGLU / GeGLU** 门控线性单元 |
| 位置编码 | 正弦绝对编码 | **RoPE** 旋转位置编码；也有 ALiBi |
| 注意力偏置 | Q/K/V 带 bias | 多数去掉 bias |
| 推理优化 | 无 | KV Cache、**GQA/MQA**（压缩 KV 显存）、**FlashAttention**（IO 感知实现） |
| 规模 | 6 层 / 512 维 | 32–120+ 层 / 4096–16384 维 |

### RoPE 的数学（来自原始论文 arXiv:2104.09864）

RoPE 的目标：用绝对位置的方式编码，但让 q·k 只依赖**相对位置**。

[一手] 二维情形，把位置 m 的 query / 位置 n 的 key 旋转：

$$
f_q(x_m, m) = (W_q x_m)\, e^{im\theta}, \qquad
f_k(x_n, n) = (W_k x_n)\, e^{in\theta}
$$

对应的实数旋转矩阵：

$$
R_m = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix}
$$

**核心性质**（因为 R_mᵀ = R_{−m}）：

$$
(R_m q)^{\top}(R_n k) = q^{\top} R_m^{\top} R_n k = q^{\top} R_{n-m} k
$$

内积只通过相对位移 n−m 进入，绝对位置 m、n 都消掉了。

[一手] d 维推广：把 d 维分成 d/2 个二维子空间，各自旋转，频率

$$
\theta_i = 10000^{-2i/d}, \quad i = 0,1,\ldots,d/2-1
$$

[一手] 论文还证明长程衰减：当相对距离 |m−n| 增大时，位置相关的内积项平均趋于衰减，符合“距离远的 token 连接更弱”的直觉。

[严谨性提示] RoPE 原论文在 Sec. 3.1 用 m−n、Sec. 3.4 用 n−m 描述相对位移，方向约定不统一；核心结论（只依赖相对位移）不受影响。

---

## 14. 常见误解澄清（对照一手来源后）

1. **“注意力权重就是模型在关注什么”** — 权重只是一次软路由，不等于因果重要性，不能直接当解释性证据。
2. **“多头是把模型拆成 h 个小模型”** — 各头共享输入、只是投影不同，且必须经 `W^O` 融合。
3. **“缩放是为了防止数值溢出”** — 直接原因是控制点积方差、避免 softmax 饱和导致梯度消失；溢出只是次要后果。
4. **“交叉注意力的 Q 来自编码器”** — 反了。Q 来自 decoder，K/V 来自 encoder。
5. **“FFN 负责序列交互”** — 不。FFN 逐位置、不跨位置；序列混合全在注意力子层。
6. **“Transformer 用 Post-Norm”** — 只对原论文成立；哈佛实现和现代大模型是 Pre-Norm。
7. **“训练时并行 = 同时生成所有 token”** — 不是。并行计算损失靠因果掩码 + teacher forcing 保证不泄漏未来；推理仍逐 token 串行。
8. **“论文实现里 embedding 与输出层共享权重”** — 论文要求共享，但常见教学实现（哈佛注解）默认不共享。
9. **“正弦位置编码能完美外推到任意长度”** — 理论上不依赖查表，但实践中超过训练长度后效果明显下降，长上下文需要专门的外推方案。
10. **“d_ff = 4×d_model 是理论必需”** — 是经验配置，不是定理。

---

## 15. 建议的复现顺序

想真正弄懂，按这个顺序动手最有效：

1. 用纯 NumPy 实现单头缩放点积注意力，在 4×4 的小矩阵上打印 S、A、AV，观察 softmax 行和为 1。
2. 加上 causal mask，对比加掩码前后 A 的上三角是否为 0。
3. 实现多头拆分/合并，验证 `view + transpose` 后形状变化。
4. 实现单层 encoder，确认输入输出形状一致（残差要求）。
5. 实现 decoder 的 `tgt[:, :-1]` / `tgt[:, 1:]` 切分，理解 teacher forcing。
6. 跑一个 toy copy task，看 loss 下降与注意力矩阵逐渐形成对齐。

---

## 参考文献

1. Vaswani, A. et al. (2017). *Attention Is All You Need*. arXiv:1706.03762. https://arxiv.org/abs/1706.03762
2. Harvard NLP (2018). *The Annotated Transformer*. https://nlp.seas.harvard.edu/annotated-transformer/
3. Su, J. et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv:2104.09864. https://arxiv.org/abs/2104.09864
4. Bahdanau, D. et al. (2014). *Neural Machine Translation by Jointly Learning to Align and Translate*. arXiv:1409.0473（论文中引用的加性注意力对照）
