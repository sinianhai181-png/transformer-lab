/* =============================================================
 * Transformer 可视化实验室 · 进阶实验室
 * 九个主题：架构族 / Modern Block / 注意力解释边界 / FlashAttention /
 * KV Cache / RoPE 与 ALiBi / QK-OV 电路 / 归纳头 / Logit Lens
 * 数据来自 TMEngine 的当前配置与快照；不做训练，不冒充大模型内部。
 * ============================================================= */
(function () {
  'use strict';
  var E = window.TMEngine, V = window.TMVis;
  var $ = function (id) { return document.getElementById(id); };
  var M = E.matrices;

  var TOPICS = [
    { id: 'architecture', label: '架构族对照' },
    { id: 'modern', label: 'Modern Block' },
    { id: 'explanation', label: '注意力≠解释' },
    { id: 'flash', label: 'FlashAttention' },
    { id: 'kv', label: 'KV Cache' },
    { id: 'position', label: 'RoPE / ALiBi' },
    { id: 'circuits', label: 'QK/OV 电路' },
    { id: 'induction', label: '归纳头' },
    { id: 'logitlens', label: 'Logit Lens' }
  ];

  var state = {
    active: 'architecture', dirty: true,
    flash: 0, kv: 0, induction: 0,
    ropeM: 2, ropeN: 5, alibiAlpha: 4
  };

  /* ---------------- 基础工具 ---------------- */
  function esc(s) { return V.esc(s); }
  function conf() {
    var s = window.TMApp && window.TMApp.state();
    return s && s.config ? s.config : E.DEFAULT_CONFIG;
  }
  function result() {
    var s = window.TMApp && window.TMApp.state();
    return s && s.result ? s.result : null;
  }
  function weights() {
    var s = window.TMApp && window.TMApp.state();
    return s && s.weights ? s.weights : null;
  }
  function fmt(v, n) { return M.fmt(v, n === undefined ? 4 : n); }
  function matmul(A, B) { return M.matmul(A, B); }
  function trans(A) { return M.transpose(A); }
  function panel(title, html, extra) {
    return '<div class="adv-panel' + (extra ? ' ' + extra : '') + '"><h4>' + esc(title) + '</h4>' + html + '</div>';
  }
  function src(label, url) {
    return '<div class="adv-source">' + esc(label) + '：<a href="' + url + '" target="_blank" rel="noopener">' + url + '</a></div>';
  }
  function heat(A, rows, cols) {
    return V.matrixHTML(A, { colorize: 'attention', rowLabels: rows || [], colLabels: cols || [] });
  }
  function diverging(A, rows, cols) {
    return V.matrixHTML(A, { colorize: 'diverging', rowLabels: rows || [], colLabels: cols || [] });
  }
  function maxAbsDiff(A, B) {
    var m = 0;
    A.forEach(function (r, i) {
      r.forEach(function (v, j) { m = Math.max(m, Math.abs(v - (B[i] && B[i][j] !== undefined ? B[i][j] : 0))); });
    });
    return m;
  }
  function interp(A, B, t) {
    return A.map(function (r, i) {
      return r.map(function (v, j) { return v * (1 - t) + B[i][j] * t; });
    });
  }
  function argmax(row) {
    var best = 0;
    for (var i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
    return best;
  }
  function rotate(v, angle) {
    var c = Math.cos(angle), s = Math.sin(angle);
    return [v[0] * c - v[1] * s, v[0] * s + v[1] * c];
  }

  /* ---------------- 主题 1：架构族 ---------------- */
  function renderArchitecture() {
    var c = conf();
    var cards = [
      {
        id: 'encoder', title: 'Encoder-only',
        flow: '<div class="adv-flow"><span>Token</span><b>→</b><span>双向自注意力 ×N</span><b>→</b><span>每个位置的表示</span></div>',
        text: '没有因果掩码，每个位置能看到整句。适合理解、分类、抽取等“看懂整段输入”的任务。BERT 是代表。',
        src: 'Devlin et al. 2018, BERT', url: 'https://arxiv.org/abs/1810.04805'
      },
      {
        id: 'decoder', title: 'Decoder-only',
        flow: '<div class="adv-flow"><span>Token</span><b>→</b><span>掩码自注意力 ×N</span><b>→</b><span>下一 token 概率</span></div>',
        text: '因果掩码让每个位置只看前缀，训练时可并行、推理时逐 token 生成。GPT / Llama / Qwen 等主流 LLM 属于这一族。',
        src: 'Radford et al. 2018, GPT', url: 'https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf'
      },
      {
        id: 'cross', title: 'Encoder–Decoder',
        flow: '<div class="adv-flow"><span>源序列</span><b>→</b><span>Encoder ×N</span><b>→</b><span>交叉注意力</span><b>→</b><span>Decoder ×N</span></div>',
        text: 'Q 来自解码器，K/V 来自编码器输出 Z。原始 Transformer 与 T5 用于翻译等序列到序列任务。',
        src: 'Vaswani et al. 2017', url: 'https://arxiv.org/abs/1706.03762'
      }
    ];
    var html = '<p class="adv-intro">三种架构族共用注意力与 FFN 组件，区别在于“谁能看见谁”与“有没有交叉注意力”。' +
      '当前主实验台模式：<b>' + esc(c.mode) + '</b>。点击卡片可把主实验台切换到对应结构。</p><div class="adv-grid">';
    cards.forEach(function (k) {
      html += panel(k.title, k.flow + '<p class="adv-intro" style="margin-top:8px">' + esc(k.text) + '</p>' +
        '<button type="button" class="btn btn-mini" data-set-mode="' + k.id + '">在主实验台查看</button>' +
        src(k.src, k.url), c.mode === k.id ? 'clickable' : '');
    });
    html += '</div><div class="adv-note">[解读] 本实验室默认展示 cross 是为了把编码器与解码器都摊开；这不代表 mainstream LLM 的主流结构。' +
      '读论文或看模型卡时，先确认它属于哪一族。</div>';
    return html;
  }

  /* ---------------- 主题 2：Modern Block ---------------- */
  function renderModern() {
    var c = conf();
    var old = '<div class="adv-flow"><span>Embedding + 正弦 PE</span><b>→</b><span>MHA</span><b>→</b><span>Add &amp; LN（Post）</span><b>→</b><span>FFN · ReLU · 4d</span><b>→</b><span>Add &amp; LN</span></div>';
    var modern = '<div class="adv-flow"><span>Token Embedding</span><b>→</b><span>RoPE(Q/K)</span><b>→</b><span>GQA</span><b>→</b><span>残差 + RMSNorm（Pre）</span><b>→</b><span>SwiGLU FFN</span><b>→</b><span>残差</span></div>';
    var rows = [
      ['归一化位置', 'Post-LN', 'Pre-LN', 'Xiong et al. 2020'],
      ['归一化函数', 'LayerNorm', 'RMSNorm（去均值、去 bias）', 'Zhang & Sennrich 2019'],
      ['位置信息', '正弦绝对编码（加一次）', 'RoPE（每层旋转 Q/K）', 'Su et al. 2021'],
      ['注意力', 'MHA（每头独立 K/V）', 'GQA（共享少量 KV 头）', 'Ainslie et al. 2023'],
      ['FFN 激活', 'ReLU，d_ff = 4d', 'SwiGLU，d_ff ≈ 8d/3', 'Shazeer 2020'],
      ['线性层 bias', '有', '常见做法：无', '模型实现约定']
    ];
    var table = '<table class="adv-table"><thead><tr><th>组件</th><th>2017 原论文</th><th>现代常见</th><th>一手来源</th></tr></thead><tbody>';
    rows.forEach(function (r) { table += '<tr><td>' + esc(r[0]) + '</td><td>' + esc(r[1]) + '</td><td>' + esc(r[2]) + '</td><td>' + esc(r[3]) + '</td></tr>'; });
    table += '</tbody></table>';
    return '<p class="adv-intro">同一张 Block 图在 2017 论文和今天主流 decoder-only 模型里差异很大。本实验室实现了原论文的 ' +
      esc(c.norm === 'post' ? 'Post-Norm' : 'Pre-Norm') + '、' + esc(c.ffnAct === 'relu' ? 'ReLU' : 'GELU') +
      ' FFN 和正弦/可学习/无位置编码；RMSNorm、SwiGLU、RoPE、GQA 在进阶主题中演示，尚未进入主引擎。</p>' +
      '<div class="adv-grid">' + panel('2017 · Attention Is All You Need', old +
        src('Vaswani et al. 2017', 'https://arxiv.org/abs/1706.03762')) +
        panel('现代常见 · 以 Llama 风格为代表', modern +
          src('Xiong et al. 2020 (Pre-LN)', 'https://arxiv.org/abs/2002.04745') +
          src('Zhang & Sennrich 2019 (RMSNorm)', 'https://arxiv.org/abs/1910.07467') +
          src('Shazeer 2020 (GLU variants)', 'https://arxiv.org/abs/2002.05202') +
          src('Ainslie et al. 2023 (GQA)', 'https://arxiv.org/abs/2305.13245')) + '</div>' +
      '<div style="margin-top:12px">' + table + '</div>' +
      '<div class="adv-note">[论文] RMSNorm 摘要报告在不同模型上运行时间减少 7%~64%；GQA 摘要报告可用原预训练 5% 的算力把 MHA checkpoint uptrain 成 GQA。' +
      '这些是原论文摘要中的实验结论，不是本实验室的实测。</div>';
  }

  /* ---------------- 主题 3：注意力≠解释 ---------------- */
  function renderExplanation() {
    var Vtoy = [[1, 0], [1, 0], [0, 1]];
    var A = [[0.5, 0.5, 0], [0, 0.5, 0.5], [1 / 3, 1 / 3, 1 / 3]];
    var A2 = A.map(function (r) { return [r[1], r[0], r[2]]; });
    var t = state.explT === undefined ? 0.5 : state.explT;
    var At = interp(A, A2, t);
    var O = matmul(A, Vtoy), O2 = matmul(A2, Vtoy), Ot = matmul(At, Vtoy);
    var labels = ['q₀', 'q₁', 'q₂'], cols = ['k₀', 'k₁', 'k₂'];
    var controls = '<div class="adv-controls"><label>λ（在原矩阵与换列矩阵间插值）<input type="range" id="advExplT" min="0" max="1" step="0.1" value="' + t + '"></label>' +
      '<span class="lab-metric">当前 λ = ' + fmt(t, 1) + '</span></div>';
    return '<p class="adv-intro">这里构造一个刻意简化的反例：Value 的前两列完全相同（V₀ = V₁），因此把注意力权重在这两列之间任意交换，' +
      '输出都不变。注意力矩阵可以差别很大，但它不是输出的唯一解释。</p>' + controls +
      '<div class="adv-grid">' +
      panel('原注意力 A', heat(A, labels, cols)) +
      panel('换列后 A′', heat(A2, labels, cols)) +
      panel('插值 λ 后的 Aλ', heat(At, labels, cols)) +
      panel('输出 O = A·V（三列相同）', heat(O, labels, ['v₀', 'v₁']) +
        '<div class="lab-metric">max|AλV − AV| = ' + fmt(maxAbsDiff(Ot, O), 8) + '</div>') +
      '</div>' +
      '<div class="adv-warn">[论文] Jain & Wallace (NAACL 2019) 发现注意力权重常与梯度重要性不相关，且可找到差异很大但预测等价的替代注意力。' +
      '[论文] Wiegreffe & Pinter (EMNLP 2019) 反驳：这取决于“解释”的定义，并提出均匀权重基线、多种子方差校准、冻结权重诊断与端到端对抗训练四种检验。' +
      '本例是数学构造，不是“训练模型里随便换注意力都不影响输出”的断言。</div>' +
      src('Jain & Wallace 2019', 'https://arxiv.org/abs/1902.10186') +
      src('Wiegreffe & Pinter 2019', 'https://arxiv.org/abs/1908.04626');
  }

  /* ---------------- 主题 4：FlashAttention ---------------- */
  function flashGrid(step) {
    var h = '<div class="adv-cells" style="grid-template-columns:repeat(8,1fr);max-width:320px">';
    for (var i = 0; i < 8; i++) for (var j = 0; j < 8; j++) {
      var cls = '';
      if (step >= 1) {
        var bi = i < 4 ? 0 : 2, bj = j < 4 ? 0 : 1, b = bi + bj;
        var shown = step === 1 ? (b === 0) : (step === 2 ? (b <= 1) : true);
        if (shown) cls = 'block' + b;
      }
      if (i === j) cls += ' diag';
      h += '<div class="adv-cell ' + cls + '"></div>';
    }
    return h + '</div>';
  }
  function renderFlash() {
    var step = state.flash;
    var texts = [
      '标准注意力把完整 n×n 分数矩阵写到 HBM，再读回做 softmax。长序列的显存与访存都是 O(n²)。',
      'FlashAttention 把 Q/K/V 切成小块，只在 SRAM 里计算一个 tile（图中高亮左上 4×4 块），避免物化整张 n×n 矩阵。',
      '对每个块做 online softmax：维护运行最大值 m 与归一化分母 ℓ，看到新块就更新，而不是等整行算完再 softmax。',
      '结果在数学上等于 softmax(QKᵀ/√dₖ)V，是精确注意力；省的是 HBM↔SRAM 的读写，FLOPs 仍是 O(n²·d)。'
    ];
    return '<p class="adv-intro">这是最容易讲错的优化：FlashAttention 不是近似注意力，也没有把 O(n²) 变成 O(n)。' +
      '它改变的是数据搬运方式。用「上一步 / 下一步」看分块流程。</p>' +
      '<div class="adv-stepper"><button type="button" class="btn btn-mini" id="advFlashPrev"' + (step === 0 ? ' disabled' : '') + '>上一步</button>' +
      '<span class="lab-metric">第 ' + (step + 1) + ' / 4 步</span>' +
      '<button type="button" class="btn btn-mini" id="advFlashNext"' + (step === 3 ? ' disabled' : '') + '>下一步</button></div>' +
      '<div class="adv-grid">' +
      panel('8×8 注意力矩阵的分块', flashGrid(step) + '<div class="adv-intro">深色 = 当前已载入 SRAM 的 tile；对角线仅作位置参考。</div>') +
      panel('这一步在做什么', '<p class="adv-intro">' + esc(texts[step]) + '</p>' +
        '<div class="adv-note">m_new = max(m_old, max(scores_tile))<br>ℓ = ℓ·exp(m_old − m_new) + Σ exp(scores_tile − m_new)</div>') +
      '</div>' +
      '<div class="adv-warn">[论文] Dao et al. 摘要明确写的是 “IO-aware exact attention algorithm”，用 tiling 减少 HBM 与 SRAM 之间的读写；' +
      '论文里的 block-sparse FlashAttention 才是近似版本。不要把它和线性注意力混为一谈。</div>' +
      src('Dao et al. 2022, FlashAttention', 'https://arxiv.org/abs/2205.14135');
  }

  /* ---------------- 主题 5：KV Cache ---------------- */
  function renderKV() {
    var c = conf();
    var n = c.mode === 'encoder' ? c.tokens.length : c.targetTokens.length;
    state.kv = Math.min(state.kv, Math.max(0, n - 1));
    var t = state.kv + 1;
    var full = n * (n + 1) / 2, cached = n;
    var bytes = 2 * c.layers * c.heads * c.dk * n * 4;
    var labels = [];
    for (var i = 0; i < n; i++) labels.push('位置' + i);
    var Mx = [];
    for (var r = 0; r < n; r++) {
      Mx.push(Array.from({ length: n }, function (_, j) { return j <= r && r < t ? 1 : 0; }));
    }
    var controls = '<div class="adv-controls"><label>生成到第几步<input type="range" id="advKvT" min="1" max="' + Math.max(1, n) + '" step="1" value="' + t + '"></label>' +
      '<span class="lab-metric">t = ' + t + ' / ' + n + '</span></div>';
    var warn = c.mode === 'encoder' ? '<div class="adv-warn">当前是 encoder 模式：没有自回归生成，KV Cache 主要用于 decoder / cross。切换到 decoder 或 cross 后这个演示更贴切。</div>' : '';
    return '<p class="adv-intro">自回归生成时，历史 token 的 K/V 不会变。把它们缓存起来，每步只算新 token 的一行 K/V；' +
      '但新 query 仍要和全部历史位置打分，所以注意力分数计算量没有消失。</p>' + controls + warn +
      '<div class="adv-grid">' +
      panel('第 t 步：缓存的 K/V 与新算的注意力行', heat(Mx, labels, labels)) +
      panel('累计代价对照',
        '<table class="adv-table"><thead><tr><th>项目</th><th>每步重算</th><th>使用 KV Cache</th></tr></thead><tbody>' +
        '<tr><td>K/V 向量计算次数</td><td>' + full + '</td><td>' + cached + '</td></tr>' +
        '<tr><td>注意力分数计算次数</td><td>' + full + '</td><td>' + full + '</td></tr>' +
        '</tbody></table><div class="lab-metric">缓存大小 ≈ 2·' + c.layers + '·' + c.heads + '·' + c.dk + '·' + n + '·4B = ' + bytes + ' B（' + fmt(bytes / 1024, 1) + ' KiB）</div>') +
      '</div>' +
      '<div class="adv-note">[论文] MQA（Shazeer 2019）只用 1 个 KV 头；GQA（Ainslie et al. 2023）用多于 1、少于查询头数的 KV 头。' +
      'GQA 摘要报告质量接近 MHA、速度接近 MQA，并且可用原预训练 5% 的算力从 MHA checkpoint uptrain。</div>' +
      src('Ainslie et al. 2023, GQA', 'https://arxiv.org/abs/2305.13245');
  }

  /* ---------------- 主题 6：RoPE / ALiBi ---------------- */
  function renderPosition() {
    var m = state.ropeM, n2 = state.ropeN;
    var q = rotate([1, 0], m), k = rotate([1, 0], n2);
    var dot = q[0] * k[0] + q[1] * k[1], expect = Math.cos(m - n2);
    var alpha = state.alibiAlpha;
    var len = 6, base = [], biased = [];
    for (var i = 0; i < len; i++) {
      base.push([]); biased.push([]);
      for (var j = 0; j < len; j++) {
        var s = Math.sin(i + j);
        base[i].push(s);
        biased[i].push(s - alpha * Math.abs(i - j));
      }
    }
    var labels = Array.from({ length: len }, function (_, i) { return String(i); });
    var controls = '<div class="adv-controls">' +
      '<label>RoPE 位置 m<input type="number" id="advRopeM" min="0" max="9" step="1" value="' + m + '"></label>' +
      '<label>RoPE 位置 n<input type="number" id="advRopeN" min="0" max="9" step="1" value="' + n2 + '"></label>' +
      '<label>ALiBi 斜率 α<input type="number" id="advAlibiA" min="0" max="12" step="0.5" value="' + alpha + '"></label></div>';
    return '<p class="adv-intro">[说明] 本主题是独立演示，主引擎仍使用正弦 / 可学习 / 无位置编码。' +
      'RoPE 把 Q/K 按位置旋转，点积只依赖相对位移；ALiBi 不给 embedding 加位置向量，而是给注意力分数加距离线性惩罚。</p>' + controls +
      '<div class="adv-grid">' +
      panel('RoPE：旋转后的 q·k',
        '<div class="adv-flow"><span>q=(' + fmt(q[0], 3) + ', ' + fmt(q[1], 3) + ')</span><b>·</b><span>k=(' + fmt(k[0], 3) + ', ' + fmt(k[1], 3) + ')</span><b>=</b><span>' + fmt(dot, 6) + '</span></div>' +
        '<div class="lab-metric">cos(m−n) = ' + fmt(expect, 6) + '</div>' +
        '<div class="adv-intro" style="margin-top:8px">二维演示中频率取 1；完整 RoPE 对每个维度对用不同频率 θ^(2i/d)。</div>') +
      panel('ALiBi：给分数加 −α|i−j|', diverging(biased, labels, labels) +
        '<div class="adv-intro">原始分数 heatmap 可看作 sin(i+j)；加偏置后近处相对更高，且无需任何位置参数。</div>') +
      '</div>' +
      '<div class="adv-warn">[论文] Su et al. 2021 证明 RoPE 的点积只依赖相对位移；Press et al. 2021 的 ALiBi 在 1024 长度训练可外推到 2048，' +
      '摘要报告同等困惑度下训练快 11%、显存少 11%。外推能力来自归纳偏置，不是无条件保证。</div>' +
      src('Su et al. 2021, RoFormer', 'https://arxiv.org/abs/2104.09864') +
      src('Press et al. 2021, ALiBi', 'https://arxiv.org/abs/2108.12409');
  }

  /* ---------------- 主题 7：QK/OV 电路 ---------------- */
  function renderCircuits() {
    var W = weights(), c = conf();
    if (!W || !W.heads || !W.heads[0]) return '<div class="adv-warn">当前没有可用权重。</div>';
    var hw = W.heads[0];
    var qk = matmul(hw.Wq, trans(hw.Wk));                       // d × d（行向量约定）
    var woSlice = W.Wo.slice(0, c.dk);                            // 第 1 个头对应的 d_k 行
    var ov = matmul(hw.Wv, woSlice);                              // d × d
    var labelsD = Array.from({ length: c.dModel }, function (_, i) { return String(i); });
    return '<p class="adv-intro">把一个注意力头拆成两段：QK 回路决定“看向哪里”，OV 回路决定“看中之后写回什么”。' +
      '这里展示第 1 个头的有效矩阵；行向量约定下分别是 W_Q·W_Kᵀ 与 W_V·W_O⁽头⁾，其中 W_O⁽头⁾ 是 W_O 中对应这个头的 d_k 行。</p>' +
      '<div class="adv-grid">' +
      panel('QK 有效矩阵 · W_Q W_Kᵀ（' + qk.length + '×' + qk[0].length + '）', diverging(qk, labelsD, labelsD)) +
      panel('OV 有效矩阵 · W_V W_O⁽头⁾（' + ov.length + '×' + ov[0].length + '）', diverging(ov, labelsD, labelsD)) +
      '</div>' +
      '<div class="adv-flow" style="margin-top:10px"><span>输入残差流</span><b>→</b><span>QK：选出位置</span><b>→</b><span>OV：写入内容</span><b>→</b><span>残差流</span></div>' +
      '<div class="adv-note">[论文] Anthropic《A Mathematical Framework for Transformer Circuits》把头拆成 QK 与 OV 两个近似独立的计算；' +
      '残差流是所有组件共同读写的通道。注意这是分析框架，不是另一种前向实现。</div>' +
      src('Elhage et al. 2021, Transformer Circuits', 'https://transformer-circuits.pub/2021/framework/index.html');
  }

  /* ---------------- 主题 8：归纳头 ---------------- */
  function renderInduction() {
    var H1 = [[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]];
    var H2 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]];
    var step = state.induction;
    var texts = [
      '序列是「A B C A」，最后位置的当前 token 是 A。目标是预测 B。',
      '第 1 层的“前一个 token 头”把前驱信息写进每个位置：位置 1 的前驱是 A。',
      '第 2 层的归纳头在最后位置寻找“前驱等于当前 token A”的位置，找到位置 1。',
      '于是它复制位置 1 的 token B，形成 [A][B] … [A] → [B] 的模式补全。'
    ];
    var labels = ['A', 'B', 'C', 'A'];
    var show = step === 0 ? null : (step === 1 ? H1 : H2);
    return '<p class="adv-intro">归纳头是两层注意力头的组合：前一个 token 头写 Key 来源，归纳头做前缀匹配并复制。' +
      '用「上一步 / 下一步」看它如何完成一次模式补全。</p>' +
      '<div class="adv-stepper"><button type="button" class="btn btn-mini" id="advIndPrev"' + (step === 0 ? ' disabled' : '') + '>上一步</button>' +
      '<span class="lab-metric">第 ' + (step + 1) + ' / 4 步</span>' +
      '<button type="button" class="btn btn-mini" id="advIndNext"' + (step === 3 ? ' disabled' : '') + '>下一步</button></div>' +
      '<div class="adv-grid">' +
      panel('序列与当前步', '<div class="adv-flow">' + labels.map(function (t, i) {
        return '<span' + (i === 3 ? ' style="border-color:#c4b5fd;background:#ede9fe"' : '') + '>' + t + '<small>位置' + i + '</small></span>';
      }).join('') + '</div><p class="adv-intro" style="margin-top:8px">' + esc(texts[step]) + '</p>') +
      panel(step === 1 ? '前一个 token 头（示意）' : '归纳头（示意）', show ? heat(show, labels, labels) : '<div class="adv-intro">点击“下一步”开始。</div>') +
      '</div>' +
      '<div class="adv-warn">[论文] Olsson et al. 2022：在小型 attention-only 模型上有强因果证据；对带 MLP 的较大模型是相关证据。' +
      '归纳头形成时点与上下文学习能力突增、损失曲线上的隆起同步。这里展示的是机制示意，不是训练出来的权重。</div>' +
      src('Olsson et al. 2022, Induction Heads', 'https://arxiv.org/abs/2209.11895');
  }

  /* ---------------- 主题 9：Logit Lens ---------------- */
  function renderLogitLens() {
    var r = result(), W = weights();
    if (!r || !W) return '<div class="adv-warn">请先在主实验台完成一次前向计算。</div>';
    var c = r.config;
    var first = r.steps.filter(function (s) { return s.id === 'decoder-input'; })[0] || r.steps.filter(function (s) { return s.id === 'input'; })[0];
    var layers = [{ label: '输入', H: first ? first.value : r.final }];
    for (var li = 0; li < c.layers; li++) {
      var s = r.steps.filter(function (x) { return x.id === 'addnorm2' + li; })[0];
      if (s) layers.push({ label: '第' + (li + 1) + '层', H: s.value });
    }
    var rows = '';
    var n = layers[0].H.length;
    for (var i = 0; i < n; i++) {
      rows += '<tr><td>位置' + i + '「' + esc((c.mode === 'encoder' ? c.tokens : c.targetTokens)[i] || '') + '」</td>';
      layers.forEach(function (L) {
        var logits = matmul([L.H[i]], trans(W.E))[0];
        var probs = M.softmaxRow(logits);
        var top = argmax(probs);
        var nextTok = c.vocab[top];
        var nextIdx = i + 1;
        var trueTok = (c.mode === 'encoder' ? c.tokens : c.targetTokens)[nextIdx];
        var trueP = trueTok && c.vocab.indexOf(trueTok) >= 0 ? probs[c.vocab.indexOf(trueTok)] : null;
        rows += '<td><b>' + esc(nextTok) + '</b><br><small>' + fmt(probs[top], 3) +
          (trueP !== null ? ' · 下一真实词 ' + fmt(trueP, 3) : '') + '</small></td>';
      });
      rows += '</tr>';
    }
    var head = '<tr><th>位置</th>' + layers.map(function (L) { return '<th>' + esc(L.label) + '</th>'; }).join('') + '</tr>';
    return '<p class="adv-intro">Logit Lens 把每一层的隐藏状态直接用输出 embedding 投影回词表，问一句：“如果现在就输出，模型会猜什么？”' +
      '这能看出预测如何逐层收敛，但不等于模型真的在那一层做了决定。</p>' +
      '<table class="adv-table"><thead>' + head + '</thead><tbody>' + rows + '</tbody></table>' +
      '<div class="adv-note">[解读] 社区方法 Logit Lens；表格显示各层快照的 top-1 概率，以及“下一真实词”的概率（若在下个 token 出现在词表中）。' +
      'Pre-Norm 栈末还有一次额外 LN，所以最后一层快照与最终输出 logits 可能有差异。</div>' +
      src('nostalgebraist 2020, Interpreting GPT: the Logit Lens', 'https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens');
  }

  /* ---------------- 渲染与事件 ---------------- */
  var RENDERERS = {
    architecture: renderArchitecture, modern: renderModern, explanation: renderExplanation,
    flash: renderFlash, kv: renderKV, position: renderPosition,
    circuits: renderCircuits, induction: renderInduction, logitlens: renderLogitLens
  };
  function tabs() {
    $('advTabs').innerHTML = TOPICS.map(function (t) {
      return '<button type="button" class="adv-tab' + (state.active === t.id ? ' active' : '') +
        '" data-id="' + t.id + '">' + esc(t.label) + '</button>';
    }).join('');
    Array.prototype.forEach.call($('advTabs').querySelectorAll('.adv-tab'), function (b) {
      b.addEventListener('click', function () { state.active = b.dataset.id; run(); });
    });
  }
  function run() {
    try {
      $('advBody').innerHTML = (RENDERERS[state.active] || renderArchitecture)();
      state.dirty = false;
      tabs();
      bind();
    } catch (e) {
      $('advBody').innerHTML = '<div class="adv-warn">主题运行失败：' + esc(e.message) + '</div>';
    }
  }
  function bind() {
    var modeBtns = $('advBody').querySelectorAll('[data-set-mode]');
    Array.prototype.forEach.call(modeBtns, function (b) {
      b.addEventListener('click', function () {
        var sel = document.getElementById('mode');
        if (!sel) return;
        sel.value = b.dataset.setMode;
        sel.dispatchEvent(new Event('change', { bubbles: true }));
      });
    });
    var prev = $('advFlashPrev'), next = $('advFlashNext');
    if (prev) prev.addEventListener('click', function () { state.flash = Math.max(0, state.flash - 1); run(); });
    if (next) next.addEventListener('click', function () { state.flash = Math.min(3, state.flash + 1); run(); });
    var ip = $('advIndPrev'), inext = $('advIndNext');
    if (ip) ip.addEventListener('click', function () { state.induction = Math.max(0, state.induction - 1); run(); });
    if (inext) inext.addEventListener('click', function () { state.induction = Math.min(3, state.induction + 1); run(); });
    var kv = $('advKvT');
    if (kv) kv.addEventListener('change', function () { state.kv = Math.max(0, Number(kv.value) - 1); run(); });
    var t = $('advExplT');
    if (t) t.addEventListener('change', function () { state.explT = Number(t.value); run(); });
    var m = $('advRopeM'), n = $('advRopeN'), a = $('advAlibiA');
    if (m) m.addEventListener('change', function () { state.ropeM = Number(m.value); run(); });
    if (n) n.addEventListener('change', function () { state.ropeN = Number(n.value); run(); });
    if (a) a.addEventListener('change', function () { state.alibiAlpha = Number(a.value); run(); });
  }
  function boot() {
    tabs();
    var btn = $('btnRunAdvanced');
    if (btn) btn.addEventListener('click', run);
    if (window.TMApp && window.TMApp.state().config) run();
  }

  window.TMAdv = {
    invalidate: function () { state.dirty = true; },
    run: run,
    topics: TOPICS.map(function (t) { return t.id; })
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
