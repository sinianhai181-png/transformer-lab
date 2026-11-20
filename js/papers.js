/* =============================================================
 * Transformer 可视化实验室 · 论文清单
 * 数据来自 docs/PAPERS.md 逐条核对的一手来源；只呈事实，不写建议。
 * 字段：标题 / 作者 / 首次提交或发表 / 发表处 / 原文链接 / 一句话结论 / 证据等级
 * ============================================================= */
(function () {
  'use strict';
  var V = window.TMVis;
  var $ = function (id) { return document.getElementById(id); };
  function esc(s) { return V.esc(s); }

  var PAPERS = [
    {
      id: 1, title: 'Attention Is All You Need', authors: 'Vaswani 等 8 人', date: '2017-06-12',
      venue: 'NIPS 2017', url: 'https://arxiv.org/abs/1706.03762',
      claim: '提出仅基于注意力机制的 Transformer 架构，完全摒弃循环与卷积。',
      level: '一级', note: 'arXiv v7 与 NeurIPS 会议版 BLEU 数字不同，并列登记。', topic: 'architecture'
    },
    {
      id: 2, title: 'Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy',
      authors: 'Xu, Wu, Wang, Long', date: '2021-10-06', venue: 'ICLR 2022',
      url: 'https://arxiv.org/abs/2110.02642',
      claim: '以关联差异判异常：异常点难以与全序列建立非平凡关联，其关联集中在相邻时刻。',
      level: '一级', note: '会议等级两源并列：ICLR 官方 poster 页标 2022 Poster，作者仓库自述 Spotlight；官方仓库声明评估使用 adjustment。',
      topic: 'lab:assoc'
    },
    {
      id: 3, title: 'Towards a Rigorous Evaluation of Time-series Anomaly Detection',
      authors: 'Kim 等 5 人', date: '2021-09-11', venue: 'AAAI 2022',
      url: 'https://arxiv.org/abs/2109.05257',
      claim: 'point adjustment 协议可能高估检测性能，随机异常分数也可成为最先进方法。',
      level: '一级', note: 'PA%K 定义已从原文 Section 4.2 核实：异常段内检出比例超 K 才触发 PA；K=0 等价原 PA，K=100 等价普通 F1。', topic: 'lab:pointadjust'
    },
    {
      id: 4, title: 'TranAD: Deep Transformer Networks for Anomaly Detection',
      authors: 'Tuli, Casale, Jennings', date: '2022-01-18', venue: 'VLDB 2022',
      url: 'https://arxiv.org/abs/2201.07284',
      claim: '自条件化与对抗训练结合；摘要称 F1 最高提升 17%、训练时间最高减少 99%。',
      level: '一级', note: '训练时间的硬件口径原文未披露。', topic: ''
    },
    {
      id: 5, title: 'Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting',
      authors: 'Zhou 等 7 人', date: '2020-12-14', venue: 'AAAI 2021',
      url: 'https://arxiv.org/abs/2012.07436',
      claim: 'ProbSparse 自注意力在时间与内存上达到 O(L log L)，生成式解码器一次前向输出长序列。',
      level: '一级', note: '', topic: ''
    },
    {
      id: 6, title: 'A Time Series is Worth 64 Words: Long-term Forecasting with Transformers',
      authors: 'Nie, Nguyen, Sinthong, Kalagnanam', date: '2022-11-27', venue: 'ICLR 2023',
      url: 'https://arxiv.org/abs/2211.14730',
      claim: '把时间序列切成 patch 作为 token，通道独立；相同回看窗口下注意力计算与内存二次下降。',
      level: '一级', note: '具体压缩倍数摘要未给出。', topic: ''
    },
    {
      id: 7, title: 'Attention is not Explanation', authors: 'Jain, Wallace', date: '2019-02-26',
      venue: 'NAACL 2019', url: 'https://arxiv.org/abs/1902.10186',
      claim: '注意力权重常与基于梯度的特征重要性不相关，可构造预测相同而注意力截然不同的分布。',
      level: '一级', note: '', topic: 'explanation'
    },
    {
      id: 8, title: 'Attention is not not Explanation', authors: 'Wiegreffe, Pinter', date: '2019-08-13',
      venue: 'EMNLP 2019', url: 'https://arxiv.org/abs/1908.04626',
      claim: '反驳上一篇：结论取决于解释的定义，并提出四种替代检验。',
      level: '一级', note: '与第 7 篇结论并列登记，未判定谁正确。', topic: 'explanation'
    },
    {
      id: 9, title: 'RoFormer: Enhanced Transformer with Rotary Position Embedding',
      authors: 'Su 等 6 人', date: '2021-04-20', venue: '公开暂未披露',
      url: 'https://arxiv.org/abs/2104.09864',
      claim: '用旋转矩阵编码绝对位置，同时在自注意力中引入显式相对位置依赖。',
      level: '一级', note: '会议信息待核验 arXiv 全文首页。', topic: 'position'
    },
    {
      id: 10, title: 'Train Short, Test Long: Attention with Linear Biases（ALiBi）',
      authors: 'Press, Smith, Lewis', date: '2021-08-27', venue: '公开暂未披露',
      url: 'https://arxiv.org/abs/2108.12409',
      claim: '不加入位置嵌入，改按距离给注意力分数加线性惩罚；1024 训练可外推到 2048。',
      level: '一级', note: '摘要给出同困惑度下训练快 11%、显存少 11%。', topic: 'position'
    },
    {
      id: 11, title: 'Root Mean Square Layer Normalization', authors: 'Zhang, Sennrich', date: '2019-10-16',
      venue: 'NeurIPS 2019', url: 'https://arxiv.org/abs/1910.07467',
      claim: '以均方根归一化替代 LayerNorm，摘要称不同模型上运行时间减少 7%~64%。',
      level: '一级', note: '', topic: 'modern'
    },
    {
      id: 12, title: 'GLU Variants Improve Transformer', authors: 'Shazeer', date: '2020-02-12',
      venue: '公开暂未披露', url: 'https://arxiv.org/abs/2002.05202',
      claim: '在 FFN 子层测试多种 GLU 变体，部分变体优于常用的 ReLU 或 GELU。',
      level: '一级', note: '具体提升数字摘要未给出。', topic: 'modern'
    },
    {
      id: 13, title: 'GQA: Training Generalized Multi-Query Transformer Models',
      authors: 'Ainslie 等 6 人', date: '2023-05-22', venue: 'EMNLP 2023',
      url: 'https://arxiv.org/abs/2305.13245',
      claim: '从多头 checkpoint 上训练为 MQA，只需原预训练算力的 5%。',
      level: '一级', note: '', topic: 'kv'
    },
    {
      id: 14, title: 'FlashAttention: Fast and Memory-Efficient Exact Attention',
      authors: 'Dao, Fu, Ermon, Rudra, Ré', date: '2022-05-27', venue: 'NeurIPS 2022',
      url: 'https://arxiv.org/abs/2205.14135',
      claim: 'IO 感知的精确注意力，用分块减少 HBM 与片上 SRAM 之间的读写。',
      level: '一级', note: '是精确注意力，不消除 O(n²)；摘要各加速数字基线不同。', topic: 'flash'
    },
    {
      id: 15, title: 'Benchmarking Diverse Learning Architectures for Leak Detection and Localization',
      authors: 'Ranganathan, Vinothkumar, Muthuramalingam', date: 'May 2026',
      venue: 'Journal of Water Process Engineering, Vol.87, 110114', url: 'https://sciencedirect.com/science/article/pii/S2214714426006720',
      claim: 'Net3 仿真 47,000+ 样本：二分类漏检所有模型 >95%，定位 Top-1 约 75–77%，集成 ML 最稳。',
      level: '一级', note: '仿真样本，非现场实测；三项数字口径不同，不可互比；DOI 公开暂未披露。', topic: ''
    },
    {
      id: 16, title: 'Dynamic Causal Graph Network for Reliable Pipeline Leak Detection',
      authors: 'Zhang, Lu, Li, Bader', date: 'November 2026',
      venue: 'Reliability Engineering & System Safety, Vol.275 Pt.2, 112795', url: 'https://sciencedirect.com/science/article/pii/S095183202600606X',
      claim: '用 Transformer 因果推断模块聚合长程证据；枝状 96.32%、环状 96.60% 准确率。',
      level: '一级', note: '实验室尺度管道，与第 15 篇管网仿真不是同一数据基础；DOI 公开暂未披露。', topic: ''
    },
    {
      id: 17, title: 'Leak detection in water supply pipeline with small-size leakage',
      authors: 'Guo 等 7 人', date: 'November 2024',
      venue: 'Process Safety and Environmental Protection, Vol.191 Pt.B, pp.2712–2724', url: 'https://sciencedirect.com/science/article/pii/S095758202401276X',
      claim: 'SqueezeNet-GRU 在 962 组数据上总体准确率 97.40%，泄漏流量占比 1.60–1.90%。',
      level: '一级', note: '该文未使用 Transformer；DOI 公开暂未披露。', topic: ''
    }
  ];

  var GROUPS = [
    { label: '一、Transformer 原始架构', ids: [1] },
    { label: '二、时序异常检测方法', ids: [2, 3, 4, 5] },
    { label: '三、时间序列的 token 化', ids: [6] },
    { label: '四、注意力解释性边界', ids: [7, 8] },
    { label: '五、现代架构组件', ids: [9, 10, 11, 12, 13, 14] },
    { label: '六、供水与管道漏水检测', ids: [15, 16, 17] }
  ];

  function jumpButton(topic) {
    if (!topic) return '';
    var lab = topic.indexOf('lab:') === 0;
    var id = lab ? topic.slice(4) : topic;
    var label = lab ? '看实验' : '看演示';
    return '<button type="button" class="paper-jump" data-module="' + (lab ? 'lab' : 'adv') +
      '" data-topic="' + esc(id) + '">' + label + '</button>';
  }

  function render() {
    var host = $('paperList');
    if (!host) return;
    var html = '';
    GROUPS.forEach(function (g) {
      html += '<div class="paper-group"><h4>' + esc(g.label) + '</h4><div class="paper-items">';
      g.ids.forEach(function (id) {
        var p = PAPERS.filter(function (x) { return x.id === id; })[0];
        if (!p) return;
        html += '<div class="paper-item">' +
          '<div class="paper-head"><a href="' + p.url + '" target="_blank" rel="noopener">' + esc(p.title) + '</a>' +
          '<span class="paper-badge">' + esc(p.level) + '</span></div>' +
          '<div class="paper-meta">' + esc(p.authors) + ' · ' + esc(p.date) + ' · ' + esc(p.venue) + '</div>' +
          '<div class="paper-claim">' + esc(p.claim) + '</div>' +
          (p.note ? '<div class="paper-note">' + esc(p.note) + '</div>' : '') +
          '<div class="paper-actions">' + jumpButton(p.topic) +
          '<a href="' + p.url + '" target="_blank" rel="noopener">原文 ↗</a></div></div>';
      });
      html += '</div></div>';
    });
    host.innerHTML = html;
    Array.prototype.forEach.call(host.querySelectorAll('.paper-jump'), function (b) {
      b.addEventListener('click', function () {
        var api = b.dataset.module === 'lab' ? window.TMLab : window.TMAdv;
        var fn = b.dataset.module === 'lab' ? 'selectExperiment' : 'selectTopic';
        if (api && typeof api[fn] === 'function') api[fn](b.dataset.topic);
      });
    });
  }

  window.TMPapers = {
    papers: PAPERS.map(function (p) { return p.id; }),
    count: PAPERS.length,
    render: render
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', render);
  else render();
})();
