# Transformer 相关论文一手素材（结构化事实）

采集日期：2026-11-19
适用范围：transformer-lab 项目论文依据补充；同时覆盖供水/管道漏水异常检测方向的期刊一手研究。

---

## 【采集概览】

本次采集围绕四个主题：Transformer 原始架构、时序异常检测方法、注意力解释性边界、现代架构组件，并补充供水与管道漏水检测方向的期刊论文。共纳入 17 篇论文（14 篇 arXiv/会议论文、3 篇期刊论文），全部来自 arXiv 官方摘要页、会议官方页面或 ScienceDirect 期刊官方页面。所有标题、作者、日期、数字均逐条核对原文页面；会议信息无法从 arXiv 页面确认的，改查会议官方页面，仍无法确认的填"公开暂未披露"。

- 一级信源：arXiv 摘要页（14 篇）、NeurIPS 2017 官方 proceedings、ICLR 2022 官方 virtual poster、AAAI 2022 官方 poster 页、ScienceDirect 期刊官方页（3 篇）。
- 二级信源：论文作者官方 GitHub 仓库（仅用于补充会议等级与复现条件，已单独标注）。
- 未采信：无明确发布主体的转载文章、AI 问答拼接内容、个人博客转述数字。

---

## 【核心信息摘要】

1. 原始 Transformer 论文的两个官方版本摘要数字不同：arXiv v7 摘要为 28.4 BLEU（En-De）/ 41.8（En-Fr），NeurIPS 2017 会议版摘要为 27.5 BLEU（En-De，165M 参数）/ 41.1（En-Fr）。两处均为原文表述，差异如实并列登记，未合并。
2. Anomaly Transformer 的异常判据是"关联差异"：论文原文表述为异常点难以与整个序列建立非平凡关联，其关联主要集中在相邻时间点；该相邻集中偏置被用作区分正常与异常的判据。
3. Anomaly Transformer 与"严谨评测"论文之间存在公开张力：前者官方 GitHub 明确写明评估使用了 Xu et al. 2018 提出的 adjustment 操作；后者指出 point adjustment 可能使随机异常分数成为"最先进方法"。
4. 供水系统漏损方向的一手基准研究给出与直觉相反的结果：二分类"是否漏水"所有模型准确率均超过 95%，而泄漏定位仍是难点，集成 ML 方法 Top-1 准确率约 75–77%，且在鲁棒性对比中集成 ML 最稳定，Transformer 与 LSTM 表现为"有竞争力"，未报告其超过集成方法。
5. 注意力权重的解释性存在两篇立场相反的一手论文：一篇论证注意力权重与梯度重要性常不相关且可构造等价预测的不同注意力分布；另一篇提出四种替代检验并论证前者的检验设计依赖"解释"的定义。

---

## 【分类详细素材】

### 一、Transformer 原始架构

**1. Attention Is All You Need**

- 作者：Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
- 首次提交：2017-06-12（v1）；最新版本：2023-08-02（v7）
- 发表：Advances in Neural Information Processing Systems 30（NIPS 2017）
- arXiv 摘要原文关键句："We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."；"Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task... establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs."
- NeurIPS 2017 会议版摘要原文关键句："Our single model with 165 million parameters, achieves 27.5 BLEU on English-to-German translation... achieving a BLEU score of 41.1."
- 交叉核验：两版摘要的 BLEU 数字不同（28.4/41.8 对 27.5/41.1），差异未在任何一页说明原因。按原文并列登记，不做取舍。
- [来源：arXiv 官方摘要页 | 发布时间：2017-06-12 (v1) / 2023-08-02 (v7) | 链接：https://arxiv.org/abs/1706.03762]
- [来源：NeurIPS 官方 proceedings | 发布时间：2017 | 链接：https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html]

---

### 二、时序异常检测方法

**2. Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy**

- 作者：Jiehui Xu, Haixu Wu, Jianmin Wang, Mingsheng Long
- 首次提交：2021-10-06（v1）；最新版本：2022-06-29（v5）
- 发表：ICLR 2022（ICLR 官方 virtual poster 页面确认接收，标注为 Poster）
- 摘要原文关键句："Our key observation is that due to the rarity of anomalies, it is extremely difficult to build nontrivial associations from abnormal points to the whole series, thereby, the anomalies' associations shall mainly concentrate on their adjacent time points."；"A minimax strategy is devised to amplify the normal-abnormal distinguishability of the association discrepancy."；"The Anomaly Transformer achieves state-of-the-art results on six unsupervised time series anomaly detection benchmarks of three applications: service monitoring, space & earth exploration, and water treatment."
- 会议等级：Spotlight。arXiv 与 ICLR 官方 poster 页面均未标注 Spotlight；论文作者官方 GitHub README 标注为 "ICLR 2022 Spotlight"。该等级属二级信源自述，公开核验路径：OpenReview 页面 https://openreview.net/forum?id=LzQQ89U1qm_ （本轮访问触发浏览器验证，未完成核验）。
- 数据集数量差异：摘要写六个基准（six benchmarks）；作者官方 GitHub 提供四个数据集的下载与脚本（SMD、MSL、SMAP、PSM），并注明 SWaT 需按官方教程申请。两处数字分属不同表述口径，并列登记。
- 评估协议：作者官方 GitHub README 原文："Especially, we use the adjustment operation proposed by Xu et al, 2018 for model evaluation."，并指向 issue #14 与 arXiv:1802.03903。
- [来源：arXiv 官方摘要页 | 发布时间：2021-10-06 (v1) / 2022-06-29 (v5) | 链接：https://arxiv.org/abs/2110.02642]
- [来源：ICLR 2022 官方 virtual poster | 发布时间：2022 | 链接：https://iclr.cc/virtual/2022/poster/7023]
- [来源：作者官方 GitHub README（二级信源） | 发布时间：页面未标注 | 链接：https://github.com/thuml/Anomaly-Transformer]

**3. Towards a Rigorous Evaluation of Time-series Anomaly Detection**

- 作者：Siwon Kim, Kukjin Choi, Hyun-Soo Choi, Byunghan Lee, Sungroh Yoon
- 首次提交：2021-09-11（v1）；最新版本：2022-01-04（v2）
- 发表：AAAI 2022, Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 36, Issue 7, pp. 7194–7201
- 摘要原文关键句："we theoretically and experimentally reveal that the PA protocol has a great possibility of overestimating the detection performance; that is, even a random anomaly score can easily turn into a state-of-the-art TAD method."；"an untrained model obtains comparable detection performance to the existing methods even when PA is forbidden."
- 提出的新协议：PA%K（要求异常片段中达到给定比例的检测点才触发 adjustment）。该名称来自第三方论文解读页面，非 arXiv 摘要原文；原文摘要仅写 "we propose a new baseline and an evaluation protocol"。PA%K 具体定义未在本次采集中核对原文，标注待核验；公开核验路径：arXiv 全文 https://arxiv.org/pdf/2109.05257 第 4 节。
- 交叉核验：该文结论与 Anomaly Transformer 官方仓库声明的 evaluation adjustment 使用形成公开张力，两方表述并列登记，未做取舍。
- [来源：arXiv 官方摘要页 | 发布时间：2021-09-11 (v1) / 2022-01-04 (v2) | 链接：https://arxiv.org/abs/2109.05257]
- [来源：AAAI 2022 官方 poster 页 | 发布时间：2022 | 链接：https://aaai-2022.virtualchair.net/poster_aaai2239]

**4. TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data**

- 作者：Shreshth Tuli, Giuliano Casale, Nicholas R. Jennings
- 首次提交：2022-01-18（v1）；最新版本：2022-05-14（v6）
- 发表：Accepted in VLDB 2022（arXiv 页面 Comments 字段原文）
- 摘要原文关键句："TranAD uses focus score-based self-conditioning to enable robust multi-modal feature extraction and adversarial training to gain stability."；"TranAD increases F1 scores by up to 17%, reducing training times by up to 99% compared to the baselines."
- 原文未说明实验在何种硬件上测得训练时间，该口径公开暂未披露；公开核验路径：arXiv 全文 https://arxiv.org/pdf/2201.07284 实验章节。
- [来源：arXiv 官方摘要页 | 发布时间：2022-01-18 (v1) / 2022-05-14 (v6) | 链接：https://arxiv.org/abs/2201.07284]

**5. Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting**

- 作者：Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, Wancai Zhang
- 首次提交：2020-12-14（v1）；最新版本：2021-03-28（v3）
- 发表：AAAI 2021（arXiv 页面 Comments 字段原文 "to be appeared in AAAI2021"）
- 摘要原文关键句："a ProbSparse self-attention mechanism, which achieves O(LlogL) in time complexity and memory usage"；"the generative style decoder... predicts the long time-series sequences at one forward operation rather than a step-by-step way".
- [来源：arXiv 官方摘要页 | 发布时间：2020-12-14 (v1) / 2021-03-28 (v3) | 链接：https://arxiv.org/abs/2012.07436]

---

### 三、时间序列的 token 化

**6. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers（PatchTST）**

- 作者：Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam
- 首次提交：2022-11-27（v1）；最新版本：2023-03-05（v2）
- 发表：Accepted by ICLR 2023（arXiv 页面 Comments 字段原文）
- 摘要原文关键句："segmentation of time series into subseries-level patches which are served as input tokens to Transformer"；"channel-independence where each channel contains a single univariate time series that shares the same embedding and Transformer weights across all the series"；"computation and memory usage of the attention maps are quadratically reduced given the same look-back window; and the model can attend longer history."
- 摘要未给出具体准确率或压缩倍数，该口径公开暂未披露；公开核验路径：arXiv 全文 https://arxiv.org/pdf/2211.14730 实验表格。
- [来源：arXiv 官方摘要页 | 发布时间：2022-11-27 (v1) / 2023-03-05 (v2) | 链接：https://arxiv.org/abs/2211.14730]

---

### 四、注意力解释性边界

**7. Attention is not Explanation**

- 作者：Sarthak Jain, Byron C. Wallace
- 首次提交：2019-02-26（v1）；最新版本：2019-05-08（v3）
- 发表：Accepted as NAACL 2019 Long Paper（arXiv 页面 Comments 字段原文）
- 摘要原文关键句："learned attention weights are frequently uncorrelated with gradient-based measures of feature importance, and one can identify very different attention distributions that nonetheless yield equivalent predictions."；"standard attention modules do not provide meaningful explanations and should not be treated as though they do."
- [来源：arXiv 官方摘要页 | 发布时间：2019-02-26 (v1) / 2019-05-08 (v3) | 链接：https://arxiv.org/abs/1902.10186]

**8. Attention is not not Explanation**

- 作者：Sarah Wiegreffe, Yuval Pinter
- 首次提交：2019-08-13（v1）；最新版本：2019-09-05（v2）
- 发表：Accepted to EMNLP 2019（arXiv 页面 Comments 字段原文）
- 摘要原文关键句："such a claim depends on one's definition of explanation, and that testing it needs to take into account all elements of the model, using a rigorous experimental design."；提出四种检验："a simple uniform-weights baseline; a variance calibration based on multiple random seed runs; a diagnostic framework using frozen weights from pretrained models; and an end-to-end adversarial attention training protocol."
- 交叉核验：该文明确以 Jain & Wallace 2019 为反驳对象，两篇结论并列登记，未判定谁正确。
- [来源：arXiv 官方摘要页 | 发布时间：2019-08-13 (v1) / 2019-09-05 (v2) | 链接：https://arxiv.org/abs/1908.04626]

---

### 五、现代架构组件

**9. RoFormer: Enhanced Transformer with Rotary Position Embedding（RoPE）**

- 作者：Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu
- 首次提交：2021-04-20（v1）；最新版本：2023-11-08（v5）
- 发表：会议信息公开暂未披露（arXiv 页面 Comments 仅写 "fixed some typos"）；公开核验路径：arXiv 全文 https://arxiv.org/pdf/2104.09864 首页页眉，或 ACL Anthology 站点检索标题。
- 摘要原文关键句："RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation."；"the flexibility of sequence length, decaying inter-token dependency with increasing relative distances".
- [来源：arXiv 官方摘要页 | 发布时间：2021-04-20 (v1) / 2023-11-08 (v5) | 链接：https://arxiv.org/abs/2104.09864]

**10. Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation（ALiBi）**

- 作者：Ofir Press, Noah A. Smith, Mike Lewis
- 首次提交：2021-08-27（v1）；最新版本：2022-04-22（v2）
- 发表：会议信息公开暂未披露（arXiv 页面未列出）；公开核验路径：OpenReview 检索标题，或 arXiv 全文 https://arxiv.org/pdf/2108.12409 首页页眉。
- 摘要原文关键句："ALiBi does not add positional embeddings to word embeddings; instead, it biases query-key attention scores with a penalty that is proportional to their distance."；"trains a 1.3 billion parameter model on input sequences of length 1024 that extrapolates to input sequences of length 2048, achieving the same perplexity as a sinusoidal position embedding model trained on inputs of length 2048 but training 11% faster and using 11% less memory."
- [来源：arXiv 官方摘要页 | 发布时间：2021-08-27 (v1) / 2022-04-22 (v2) | 链接：https://arxiv.org/abs/2108.12409]

**11. Root Mean Square Layer Normalization（RMSNorm）**

- 作者：Biao Zhang, Rico Sennrich
- 提交：2019-10-16（v1）
- 发表：NeurIPS 2019（arXiv 页面 Comments 字段原文）
- 摘要原文关键句："RMSNorm regularizes the summed inputs to a neuron in one layer according to root mean square (RMS), giving the model re-scaling invariance property and implicit learning rate adaptation ability."；"achieves comparable performance against LayerNorm but reduces the running time by 7%~64% on different models."
- [来源：arXiv 官方摘要页 | 发布时间：2019-10-16 | 链接：https://arxiv.org/abs/1910.07467]

**12. GLU Variants Improve Transformer**

- 作者：Noam Shazeer
- 提交：2020-02-12（v1）
- 发表：会议信息公开暂未披露（arXiv 页面未列出）；公开核验路径：arXiv 全文 https://arxiv.org/pdf/2002.05202 首页页眉。
- 摘要原文关键句："We test these variants in the feed-forward sublayers of the Transformer... and find that some of them yield quality improvements over the typically-used ReLU or GELU activations."
- 摘要未给出具体提升数字，该口径公开暂未披露；公开核验路径：arXiv 全文实验表格。
- [来源：arXiv 官方摘要页 | 发布时间：2020-02-12 | 链接：https://arxiv.org/abs/2002.05202]

**13. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints**

- 作者：Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai
- 首次提交：2023-05-22（v1）；最新版本：2023-12-23（v3）
- 发表：Accepted at EMNLP 2023（arXiv 页面 Comments 字段原文）
- 摘要原文关键句："we propose a recipe for uptraining existing multi-head language model checkpoints into models with MQA using 5% of original pre-training compute"；"uptrained GQA achieves quality close to multi-head attention with comparable speed to MQA."
- [来源：arXiv 官方摘要页 | 发布时间：2023-05-22 (v1) / 2023-12-23 (v3) | 链接：https://arxiv.org/abs/2305.13245]

**14. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**

- 作者：Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
- 首次提交：2022-05-27（v1）；最新版本：2022-06-23（v2）
- 发表：Advances in Neural Information Processing Systems 35, NeurIPS 2022, pp. 16344–16359
- 摘要原文关键句："We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM."；"We argue that a missing principle is making attention algorithms IO-aware".
- 摘要列出加速与质量数字，注意各数字的对照基线不同，逐项登记见数据汇总表。
- [来源：arXiv 官方摘要页 | 发布时间：2022-05-27 (v1) / 2022-06-23 (v2) | 链接：https://arxiv.org/abs/2205.14135]
- [来源：NeurIPS 论文元数据（机构机构库检索页） | 发布时间：2022 | 链接：https://researchconnect.buffalo.edu/en/publications/flashattention-fast-and-memory-efficient-exact-attention-with-io-]

---

### 六、供水与管道漏水检测（期刊一手研究）

**15. Benchmarking Diverse Learning Architectures for Leak Detection and Localization in Water Distribution Systems**

- 作者：Ranganathan S, Vinothkumar N, Muthuramalingam E
- 发布时间：May 2026（ScienceDirect 页面标注）
- 期刊名称/卷/DOI：公开暂未披露（该页面仅给出日期与文章号 PII S2214714426006720）；公开核验路径：ScienceDirect 文章页 https://sciencedirect.com/science/article/pii/S2214714426006720 的 "Article info" 栏目。
- 摘要原文关键句："Using the Net3 hydraulic model, a large-scale dataset was generated with over 47,000 samples"；"binary leak detection is a largely solved problem, with all models achieving over 95% accuracy. Localization, however, remains challenging—ensemble ML methods (Random Forest, XGBoost, LightGBM) achieved Top 1 accuracies around 75–77%, while Transformers and LSTM models showed competitive results."；"A comprehensive robustness analysis across leak size, sensor coverage, and noise revealed ensemble ML to be the most stable."
- 数据口径：47,000+ 为 Net3 水力模型仿真样本数，非现场实测样本；>95% 为二分类准确率；75–77% 为定位 Top-1 准确率。三项口径不同，不可互相比较。
- [来源：ScienceDirect 期刊官方页 | 发布时间：May 2026 | 链接：https://sciencedirect.com/science/article/pii/S2214714426006720]

**16. Dynamic Causal Graph Network for Reliable Pipeline Leak Detection**

- 作者：Yuxuan Zhang, Yuchen Lu, Dun Li, Sebastian Bader
- 发布时间：公开暂未披露（页面未给出具体日期；PII 编号 S095183202600606X 含 2026）；公开核验路径：ScienceDirect 文章页 "Article info" 栏目。
- 期刊名称/卷/DOI：公开暂未披露（同上核验路径）。
- 摘要原文关键句："A Transformer-based Causal Inference and Reasoning (CIR) module aggregates long-range causal evidence for robust state inference."；"DCGN achieves 96.32% and 96.60% accuracy respectively"；"Under extreme conditions including 5-dB signal-to-noise ratio (SNR) and strong impulse noise, DCGN maintains over 86% accuracy, and achieves above 81% accuracy with only 10% training samples".
- 数据口径：96.32% 对应枝状（branched）拓扑，96.60% 对应环状（looped）拓扑；>86% 为 5 dB SNR 与强脉冲噪声条件下；>81% 为仅用 10% 训练样本条件下。该文实验对象为实验室尺度管道（作者页另述四传感器、51.2 kHz 采样、30 s 记录），与篇 15 的管网仿真不是同一数据基础，准确率不可直接比较。
- [来源：ScienceDirect 期刊官方页 | 发布时间：公开暂未披露 | 链接：https://sciencedirect.com/science/article/pii/S095183202600606X]

**17. Leak detection in water supply pipeline with small-size leakage using deep learning networks**

- 作者：Pengcheng Guo, Shumin Zheng, Jianguo Yan, Yan Xu, Jiang Li, Jinyang Ma, Shuaihui Sun
- 发布时间：November 2024（ScienceDirect 页面标注）
- 期刊名称/卷/DOI：公开暂未披露（页面仅给出日期与文章号 PII S095758202401276X）；公开核验路径：ScienceDirect 文章页 "Article info" 栏目。
- 摘要原文关键句："inner pipe diameter of 100 mm in the operating range of volume flow Q = 40–80 m3/h, leakage flow Ql = 0.65–1.33 m3/h and pressure p = 130–220 kPa"；"SqueezeNet-GRU demonstrates superior performance in pipeline leak detection and localization applications, achieving 97.40% accuracy for the total dataset."
- 讨论章节原文关键句："especially when the leakage flow is less than 5% of the total volume flow"；"leakage flow accounting for only 1.60–1.90% of the total volume flow"。
- 数据口径：97.40% 为 962 组数据上的总体准确率；1.60–1.90% 为泄漏流量占总流量比例；0.65–1.33 m3/h 为泄漏流量绝对值；该文使用 CNN/LSTM/GRU/SqueezeNet，未使用 Transformer。
- [来源：ScienceDirect 期刊官方页 | 发布时间：November 2024 | 链接：https://sciencedirect.com/science/article/pii/S095758202401276X]

---

## 【数据汇总表】

所有表格字段均填写；无法核实项统一写"公开暂未披露"，不以推测替代。

| 编号 | 论文 | 首次提交/发表 | 核心数字 | 数字口径（原文口径） | 对照基线 |
| --- | --- | --- | --- | --- | --- |
| 1 | Attention Is All You Need (arXiv v7) | 2017-06-12 | 28.4 BLEU / 41.8 BLEU | WMT 2014 En-De / En-Fr；En-Fr 训练 3.5 天、8 GPU | "improving over the existing best results, including ensembles by over 2 BLEU" |
| 1b | Attention Is All You Need (NeurIPS 2017) | 2017 | 27.5 BLEU / 41.1 BLEU / 165M 参数 | En-De / En-Fr；单模型参数量 | En-De "over the existing best ensemble result by over 1 BLEU"；En-Fr "previous single state-of-the-art... by 0.7 BLEU" |
| 2 | Anomaly Transformer | 2021-10-06 | 六个基准；四个下载数据集 | 摘要基准数 / GitHub 提供脚本数，口径不同 | 未标注具体模型的对照数字 |
| 3 | Rigorous Evaluation | 2021-09-11 | 公开暂未披露（摘要未给数字） | 摘要为理论+实验结论性表述 | 随机异常分数、未训练模型 |
| 4 | TranAD | 2022-01-18 | F1 最高 +17%；训练时间最高 −99% | 六个公开数据集上的相对改善幅度，硬件口径公开暂未披露 | baselines（原文未逐一列名于摘要） |
| 5 | Informer | 2020-12-14 | O(L log L) | 时间与内存复杂度，L 为序列长度 | 标准 Transformer 二次复杂度 |
| 6 | PatchTST | 2022-11-27 | 二次下降（quadratically reduced） | 相同回看窗口下注意力图计算与内存；倍数公开暂未披露 | SOTA Transformer 时序模型 |
| 9 | RoFormer | 2021-04-20 | 公开暂未披露（摘要无数字） | 性质描述：长度灵活、依赖随相对距离衰减 | 原文称 "consistently overcomes its alternatives"，未列数字 |
| 10 | ALiBi | 2021-08-27 | 1.3B 参数；1024→2048；训练快 11%；显存少 11% | 参数量；训练长度→外推长度；相同困惑度条件下的训练时间与内存 | 在 2048 长度上训练的正弦位置编码模型 |
| 11 | RMSNorm | 2019-10-16 | 运行时间减少 7%~64% | 不同模型上的相对运行时间减少比例 | LayerNorm |
| 12 | GLU Variants | 2020-02-12 | 公开暂未披露（摘要无数字） | 定性结论：部分变体优于 ReLU/GELU | ReLU / GELU |
| 13 | GQA | 2023-05-22 | 5% | 从 MHA checkpoint uptrain 到 MQA 所用的原始预训练算力比例 | MQA / MHA |
| 14 | FlashAttention | 2022-05-27 | 15%；3×；2.4×；0.7；6.4；61.4%；63.1% | 逐项见下方分行 | 逐项见下方分行 |
| 15 | WDS Leak Benchmark | May 2026 | 47,000+；>95%；75–77% | 仿真样本数；二分类准确率；定位 Top-1 准确率 | ML/DL/GNN/Transformer 四类同条件对比 |
| 16 | DCGN | 公开暂未披露 | 96.32%；96.60%；>86%；>81%；10% | 枝状拓扑准确率；环状拓扑准确率；极端噪声准确率；小样本准确率；训练样本比例 | baselines（ANOVA 验证显著性） |
| 17 | Small-size Leak Detection | November 2024 | 97.40%；1.60–1.90%；962；100 mm | 总体准确率；泄漏流量占比；数据组数；管内径 | CNN/LSTM/GRU/SqueezeNet 互比 |

FlashAttention 数字逐项口径：

| 数字 | 指标类型 | 条件 | 对照基线 |
| --- | --- | --- | --- |
| 15% | 端到端训练加速（wall-clock） | BERT-large，序列长度 512 | MLPerf 1.1 训练速度记录 |
| 3× | 训练加速倍数 | GPT-2，序列长度 1K | 既有基线（原文摘要未列明具体名称） |
| 2.4× | 训练加速倍数 | Long Range Arena，序列长度 1K–4K | 既有基线（原文摘要未列明具体名称） |
| 0.7 | 困惑度改善（perplexity 绝对值） | GPT-2 | 既有基线 |
| 6.4 | 长文档分类准确率提升（points，百分点） | 长文档分类任务 | 既有基线 |
| 61.4% | 准确率 | Path-X，序列长度 16K | "better-than-chance" 阈值 |
| 63.1% | 准确率 | Path-256，序列长度 64K | "better-than-chance" 阈值 |

---

## 【信息可信度说明】

- **一级信源（核心采信）**：arXiv 官方摘要页（14 篇，标题/作者/日期/摘要原文逐条读取）；NeurIPS 官方 proceedings（篇 1 会议版摘要）；ICLR 2022 官方 virtual poster（篇 2 接收事实与摘要）；AAAI 2022 官方 poster 页（篇 3 接收事实与摘要）；ScienceDirect 期刊官方页（篇 15/16/17）。这些页面均给出发布主体与原文，作为事实依据。
- **二级信源（补充参考）**：论文作者官方 GitHub README（篇 2 的 Spotlight 等级调整操作声明）。已单独标注，未作为数字来源。
- **待核验条目**：篇 2 的 Spotlight 等级（OpenReview 页面触发浏览器验证，未完成核验，公开路径已给）；篇 3 的 PA%K 具体定义（未核对原文，公开路径已给）；篇 9/10/12 的会议信息（arXiv 页面未标注，公开路径已给）；篇 15/16/17 的期刊名、卷、DOI（ScienceDirect 页面 "Article info" 栏目，需登录或机构权限完整查看，本轮未突破）。
- **冲突并列登记（不强行统一）**：篇 1 的 arXiv v7 与 NeurIPS 2017 两版摘要 BLEU 数字；篇 2 摘要的六个基准与官方仓库四个脚本；篇 2 使用 adjustment 评估与篇 3 对 point adjustment 的批评；篇 7 与篇 8 对注意力解释性的相反结论。
- **风险/反例维度**：篇 3 指出评测协议可使随机分数达到 SOTA；篇 7 指出注意力权重不构成解释；篇 15 指出漏水二分类已"largely solved"、定位仍是难点且集成 ML 鲁棒性最稳；篇 16 指出统计相关会引入伪连接，需因果约束。以上均为原文表述，非本采集的推论。
- **未采信内容**：此前五份附件中保留"浮点搜索2.0""已阅读 10 篇网页""DeepSeek-R1满血版"等网页残留的 PDF，以及无发布主体的转载文章，均未按论文来源采信。
