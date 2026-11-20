# 进阶实验室说明

进阶实验室是主界面的第九块内容，用十一个主题回答一个问题：**2017 年的论文结构，怎样走到今天的模型、推理系统和可解释性讨论？**

所有主题都基于当前页面配置或明确标注的 toy 示例，不连接外部模型，不做训练。界面中保留了三级来源标注：

- **[论文]** — 原论文摘要或正文可核对的结论。
- **[实现]** — 主流实现中的做法，可能与论文原文不同。
- **[解读]** — 学界或社区常见解释，不是论文证明的结论。

## 十一个主题一览

| 主题 | 能看到什么 | 主要来源 |
| --- | --- | --- |
| 架构族对照 | Encoder-only / Decoder-only / Encoder–Decoder 的数据流差异；点击卡片切换主实验台 | BERT、GPT、Vaswani et al. 2017 |
| Modern Block | 2017 原版 Block 与现代常见 decoder-only Block 的组件对照 | Xiong 2020、Zhang 2019、Shazeer 2020、Ainslie 2023 |
| 注意力≠解释 | 不同注意力矩阵产生完全相同输出的数学构造；论文争论与四种检验 | Jain & Wallace 2019、Wiegreffe & Pinter 2019 |
| FlashAttention | 8×8 矩阵分块、online softmax 四步演示；精确注意力与 O(n²) 的澄清 | Dao et al. 2022 |
| KV Cache | 生成第 t 步时缓存了哪些 K/V；累计 K/V 计算次数与缓存字节数 | Shazeer 2019、Ainslie et al. 2023 |
| RoPE / ALiBi | 旋转后点积只依赖相对位置；给分数加距离线性偏置 | Su et al. 2021、Press et al. 2021 |
| QK/OV 电路 | 当前第 1 个头的 QK 与 OV 有效矩阵热力图 | Elhage et al. 2021 |
| 归纳头 | `[A][B] … [A] → [B]` 的两层组合过程；证据强度分级 | Olsson et al. 2022 |
| Logit Lens | 每层隐藏状态投影回词表后的 top-1 预测与下一真实词概率 | nostalgebraist 2020（社区方法） |
| 批注四问 | 分词 / 编码 / 重要性 / 解码器四问，按[论文/事实]/[解读]/[边界]分层，可跳对应实验 | PatchTST、Anomaly Transformer、Jain & Wallace、BERT |
| 实现错误对照 | 五例实测错误（编译失败、掩码反向、padding 维、NaN、恒等映射）的错误→实测→正确对照 | 附件代码实测审读，`docs/REVIEW-materials.md` |

## 1. 架构族对照

三种架构族共享注意力与 FFN 组件，区别在于“谁能看见谁”：

- **Encoder-only**：双向自注意力，无因果掩码。每个位置能看到整句，适合分类、抽取等表示任务。BERT 是代表。
- **Decoder-only**：因果掩码，每个位置只看前缀。训练时可并行，推理时逐 token 生成。GPT、Llama、Qwen 等主流 LLM 属于这一族。
- **Encoder–Decoder**：编码器输出 Z 通过交叉注意力进入解码器；Q 来自解码器，K/V 来自编码器。原始 Transformer 与 T5 属于这一族。

本实验室默认用 cross 模式把编码器和解码器都展示出来，这是教学选择，不代表主流 LLM 的结构。点击卡片会把主实验台切到对应模式。

## 2. Modern Block

原论文 Block 是 Post-LN + 正弦绝对编码 + MHA + ReLU FFN + 4d 中间维度。现代常见 decoder-only Block 则把这些组件逐一替换：

| 组件 | 2017 原论文 | 现代常见 | 来源 |
| --- | --- | --- | --- |
| 归一化位置 | Post-LN | Pre-LN | Xiong et al. 2020 |
| 归一化函数 | LayerNorm | RMSNorm（去均值、去 bias） | Zhang & Sennrich 2019 |
| 位置信息 | 正弦绝对编码，加一次 | RoPE，每层旋转 Q/K | Su et al. 2021 |
| 注意力 | MHA，每头独立 K/V | GQA，共享少量 KV 头 | Ainslie et al. 2023 |
| FFN | ReLU，d_ff = 4d | SwiGLU，d_ff ≈ 8d/3 | Shazeer 2020 |
| 线性层 bias | 有 | 常见做法：无 | 模型实现约定 |

[论文] Xiong et al. 用平均场理论说明 Post-LN 在初始化时输出层附近梯度较大，需要 warm-up；Pre-LN 梯度更良态。RMSNorm 摘要报告在不同模型上运行时间减少 **7%~64%**。GQA 摘要报告可用原预训练 **5%** 算力把 MHA checkpoint uptrain 成 GQA，质量接近 MHA、速度接近 MQA。

这些组件在主引擎中只实现了一部分：Pre/Post-Norm 与 ReLU/GELU 可选；RMSNorm、SwiGLU、RoPE、GQA 只在本主题中演示，不参与主前向计算。

## 3. 注意力≠解释

页面构造一个 toy 例子：Value 的前两列相同（V₀ = V₁），因此把注意力权重在两列之间交换，输出 `O = A·V` 不变。拖动 λ 滑块在两种注意力矩阵之间插值，输出始终相同，`max|AλV − AV| = 0`。

这说明注意力权重不是输出的唯一解释，但不能推出“训练模型里随便换注意力都不影响输出”。

- [论文] Jain & Wallace（NAACL 2019）：注意力权重常与梯度重要性不相关，且可构造差异很大但预测等价的替代注意力。
- [论文] Wiegreffe & Pinter（EMNLP 2019）：是否算解释取决于定义，提出均匀权重基线、多种子方差校准、冻结权重诊断、端到端对抗训练四种检验。

## 4. FlashAttention

[论文] Dao et al. 2022 摘要明确称其为 “IO-aware exact attention algorithm”：

- 用 **tiling** 把 Q/K/V 切成小块，只在 SRAM 中计算一个 tile；
- 用 **online softmax** 维护运行最大值 m 与归一化分母 ℓ，无需等整行算完；
- 减少 HBM ↔ SRAM 的读写，但 FLOPs 仍是 `O(n²·d)`；
- 论文里的 block-sparse FlashAttention 才是近似版本。

页面用 8×8 矩阵和四步播放器展示分块顺序与 online softmax 更新式。它解释的是访存优化，不是把二次复杂度变成线性。

## 5. KV Cache

自回归生成时，历史 token 的 K/V 不会随新 token 改变。缓存后每步只算新 token 的一行 K/V：

- 无缓存：累计 K/V 向量计算次数 `n(n+1)/2`
- 有缓存：`n`
- 注意力分数计算次数仍是 `n(n+1)/2`，因为新 query 仍要和全部历史位置打分

页面按当前 `layers / heads / d_k / 序列长度` 估算缓存字节数：`2 · N · h_kv · d_k · n · bytes`。默认按 fp32（4 字节）计算，仅用于量级直觉。

[论文] MQA（Shazeer 2019）只保留 1 个 KV 头；GQA（Ainslie et al. 2023）使用介于 1 和查询头数之间的 KV 头数。本主题用图示说明这一族优化，不改动主引擎的 MHA 结构。

## 6. RoPE / ALiBi

- **RoPE**：把位置 m 的 q、k 分别旋转 mθ，点积只依赖相对位移 m−n。页面用二维向量演示 `q·k = cos(m−n)`；完整 RoPE 对每个维度对使用不同频率 `θ^(2i/d)`。
- **ALiBi**：不给 embedding 加位置向量，而是给注意力分数加 `−α|i−j|` 偏置。页面用 6 个位置和可调 α 展示分数如何向近处倾斜。

[论文] Su et al. 2021 给出 RoPE 的相对位置性质。Press et al. 2021 报告：在 1024 长度训练的 1.3B 模型可外推到 2048，达到与 2048 长度训练的正弦编码模型相同困惑度，且训练快 11%、显存少 11%。外推来自归纳偏置，不是无条件保证。

## 7. QK/OV 电路

[论文] Anthropic《A Mathematical Framework for Transformer Circuits》把一个头拆成两段：

- **QK 回路**：决定“看向哪里”。行向量约定下有效矩阵是 `W_Q · W_Kᵀ`。
- **OV 回路**：决定“看中之后写回什么”。行向量约定下是 `W_V · W_O⁽头⁾`，其中 `W_O⁽头⁾` 是输出投影中属于该头的 d_k 行。

页面展示当前第 1 个头的两张 8×8 热力图。这是分析框架，不是另一种前向实现；QK 与 OV 只能近似拆分，完整模型还有 MLP、层间组合与残差流。

## 8. 归纳头

[论文] Olsson et al. 2022 定义归纳头完成 `[A][B] … [A] → [B]`：

1. 前一个 token 头把“我的前驱是谁”写进当前位置；
2. 归纳头在当前 token 为 A 时，寻找前驱也是 A 的位置；
3. 它复制那个位置后面的 token B，形成模式补全。

页面用 `A B C A` 四步演示这个过程。证据强度必须分清：小型 attention-only 模型上有强因果证据；带 MLP 的较大模型上是相关证据。页面展示的是机制示意，不是训练出来的权重。

## 9. Logit Lens

Logit Lens 是社区方法（nostalgebraist 2020），把每层隐藏状态直接用输出 embedding 投影回词表，看“如果现在就输出，模型会猜什么”。页面表格显示：

- 每个位置在各层的 top-1 token 与概率；
- 下一真实词出现在词表中时的概率。

[解读] 这能看出预测如何逐层收敛，但不等于模型真的在那一层做了决定。Pre-Norm 栈末还有一次额外 LayerNorm，因此最后一层快照与最终输出 logits 可能不同。

## 10. 批注四问

来自五份材料页边的四条手写批注，是整套材料真正的主线。界面把每问的答复按三级来源拆开，避免把推论当论文结论：

- **为什么分词**：文本是符号索引化；时序的对应物是"把窗口切成多少位置"，即 patch 切分。粒度决定注意力分辨率与 O(n²) 计算量。
- **怎么编码才富含信息**：可学习 embedding + 位置/时间特征 + 多层上下文三者叠加，不是给每一维贴语义标签。
- **注意力≠重要性**：softmax 行和为 1，只表示读取预算分配。重要性更可靠的量是消融、梯度×输入、OV 写回量；对角元最不可靠。异常检测推荐用关联差异。
- **要不要解码器**：判别/打分用 encoder-only 就够；只有重建式需要重构头，自回归解码器不需要。

每问都附一手来源链接与可跳转的对应实验（复杂度、位置编码、关联差异、架构族）。本主题属 `[解读] + [论文]` 组合，非新事实。

## 11. 实现错误对照

五例均来自对所附 `transformer代码.docx` 的实测审读，不是虚构案例：

1. **docstring 缺引号** — `py_compile` 实测第 1 段第 35 行 `SyntaxError: invalid character '。'`，该段无法运行；第 2 段 452 行编译通过。
2. **因果掩码整体反向** — `triu(·, diagonal=1)` 与 `mask==0 → −inf` 组合后，位置 t 反而屏蔽了 ≤ t 的位置，对角线自己也看不到。正确应为 `~triu(·, diagonal=1)`。
3. **padding 掩码落在 query 维** — `(tgt!=0).unsqueeze(1).unsqueeze(3)` 得 `(B,1,L,1)`，标记的是 query；应为 `.unsqueeze(2)` 得到 `(B,1,1,L)` 按 key 屏蔽。
4. **全 −inf 行 + softmax = NaN** — numpy 实测 `-inf - (-inf)` 得到 nan；标准做法用 `-1e9`，并去掉多余的 `[:, None, :, :]`。
5. **训练任务是恒等映射** — 70% 样本 `tgt_output = tgt_input.clone()` 且 src 独立随机，模型学成恒等映射即可压低 loss，交叉注意力白训；应让 tgt 由 src 决定并做右移 teacher forcing。

每例按"错误写法 → 实测/后果 → 正确写法 → 根因"四段呈现。本主题的实测记录见仓库外 `docs/REVIEW-materials.md`（附件审读报告），本实验台引擎已修正对应问题。

## 已知边界

- 主引擎始终按 2017 论文结构运行；进阶主题里的 RMSNorm、SwiGLU、RoPE、GQA、KV Cache 是独立演示，不改变前向结果。
- 所有示例都是玩具规模；归纳头与 Logit Lens 不是从训练模型中提取的真实电路。
- 论文数字均来自摘要或正文，并标明出处；本页没有复现那些训练实验。

## 参考文献

1. Vaswani et al. 2017. *Attention Is All You Need*. https://arxiv.org/abs/1706.03762
2. Xiong et al. 2020. *On Layer Normalization in the Transformer Architecture*. https://arxiv.org/abs/2002.04745
3. Zhang & Sennrich 2019. *Root Mean Square Layer Normalization*. https://arxiv.org/abs/1910.07467
4. Shazeer 2020. *GLU Variants Improve Transformer*. https://arxiv.org/abs/2002.05202
5. Su et al. 2021. *RoFormer*. https://arxiv.org/abs/2104.09864
6. Ainslie et al. 2023. *GQA*. https://arxiv.org/abs/2305.13245
7. Jain & Wallace 2019. *Attention is not Explanation*. https://arxiv.org/abs/1902.10186
8. Wiegreffe & Pinter 2019. *Attention is not not Explanation*. https://arxiv.org/abs/1908.04626
9. Dao et al. 2022. *FlashAttention*. https://arxiv.org/abs/2205.14135
10. Press et al. 2021. *Train Short, Test Long*（ALiBi）. https://arxiv.org/abs/2108.12409
11. Elhage et al. 2021. *A Mathematical Framework for Transformer Circuits*. https://transformer-circuits.pub/2021/framework/index.html
12. Olsson et al. 2022. *In-context Learning and Induction Heads*. https://arxiv.org/abs/2209.11895
13. nostalgebraist 2020. *Interpreting GPT: the Logit Lens*. https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
14. Nie et al. 2023. *A Time Series is Worth 64 Words*（PatchTST）. https://arxiv.org/abs/2211.14730
15. Xu et al. 2022. *Anomaly Transformer*（关联差异）. https://arxiv.org/abs/2110.02642
16. Devlin et al. 2018. *BERT*. https://arxiv.org/abs/1810.04805
