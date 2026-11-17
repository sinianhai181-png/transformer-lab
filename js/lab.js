/* =============================================================
 * Transformer 可视化实验室 · 拓展实验台
 * 五个对照实验：位置编码消融 / 因果掩码 / 多头分工 / 缩放与温度 / 注意力复杂度
 * 原则：固定同一组权重，只改变被研究的那个变量。
 * ============================================================= */
(function () {
  'use strict';
  var E = window.TMEngine, V = window.TMVis;
  var $ = function (id) { return document.getElementById(id); };
  var M = E.matrices;

  var EXPERIMENTS = [
    { id: 'pe', label: '位置编码消融' },
    { id: 'mask', label: '因果掩码对比' },
    { id: 'heads', label: '多头分工' },
    { id: 'scale', label: '缩放与温度' },
    { id: 'complexity', label: '注意力复杂度' }
  ];
  var state = { active: 'pe', dirty: true, report: null };

  function cfg() { return window.TMApp.state().config; }
  function model() { return window.TMApp.state().weights; }
  function esc(s) { return V.esc(s); }
  function smallHeat(A, rowLabels, colLabels) {
    return V.matrixHTML(A, { colorize: 'attention', rowLabels: rowLabels || [], colLabels: colLabels || [] });
  }
  function frob(A, B) {
    var s = 0;
    for (var i = 0; i < A.length; i++) for (var j = 0; j < A[i].length; j++) {
      var d = A[i][j] - (B[i] ? B[i][j] : 0); s += d * d;
    }
    return Math.sqrt(s);
  }
  function upperSum(A) {
    var s = 0;
    for (var i = 0; i < A.length; i++) for (var j = i + 1; j < A[i].length; j++) s += A[i][j];
    return s;
  }
  function stepOf(result, id) {
    return result.steps.find(function (s) { return s.id === id; });
  }
  function attentionOf(result, id) {
    var s = stepOf(result, id);
    return s ? s.value : null;
  }

  /* 从当前权重直接算某个头的注意力，引擎快照只保留头 0，这里补齐其他头 */
  function headAttention(headIndex, tokens, conf, W) {
    var d = conf.dModel, dk = conf.dk, n = tokens.length;
    var vi = {}; conf.vocab.forEach(function (t, i) { vi[t] = i; });
    var X = tokens.map(function (t) {
      return W.E[vi[t] === undefined ? 0 : vi[t]].map(function (v) { return v * Math.sqrt(d); });
    });
    var PE = conf.pe === 'none' ? M.zeros(n, d) :
      (conf.pe === 'learned' ? W.PE.slice(0, n).map(function (r) { return r.slice(0, d); }) : E.sinusoidalPE(n, d));
    var x0 = M.add(X, PE);
    var src = conf.norm === 'pre' ? M.layerNorm(x0) : x0;
    var hw = W.heads[headIndex];
    var Q = M.matmul(src, hw.Wq), K = M.matmul(src, hw.Wk), Vv = M.matmul(src, hw.Wv);
    return E.attention(Q, K, Vv, { dk: dk, scale: conf.scale, temperature: conf.temperature,
      mask: conf.mode === 'decoder' ? E.causalMatrix(n) : null });
  }

  /* ---------------- 实验 1：位置编码消融 ---------------- */
  function runPE() {
    var conf = cfg(), W = model();
    var variants = [
      { pe: 'sinusoidal', label: '正弦绝对编码（论文）' },
      { pe: 'learned', label: '可学习位置编码' },
      { pe: 'none', label: '无位置编码' }
    ];
    var results = variants.map(function (v) {
      return { label: v.label, result: E.forward(Object.assign({}, conf, { pe: v.pe, weights: W })) };
    });
    var base = results[0].result.final;
    var html = '<p class="lab-intro">三组实验共用同一组 Q/K/V 权重，只改位置编码。观察注意力矩阵是否变化，以及最终表示 Z 与论文方案的距离。' +
      '把源序列 token 顺序打乱再运行，可以看到「无位置编码」组的输出作为集合几乎不变——这正是自注意力置换不变性的直接后果。</p>' +
      '<div class="lab-grid">';
    results.forEach(function (r) {
      var A = attentionOf(r.result, 'self-attn0');
      html += '<div class="lab-panel"><h4>' + esc(r.label) + '</h4>' +
        smallHeat(A, conf.tokens.map(function (t, i) { return '位置' + i + '「' + t + '」'; }),
          conf.tokens.map(function (t, i) { return '位置' + i + '「' + t + '」'; })) +
        '<div class="lab-metric">‖ΔZ‖_F = ' + M.fmt(frob(r.result.final, base), 4) + '</div></div>';
    });
    html += '</div><div class="lab-warn">[解读] 论文只给出「embedding 乘 √d_model」这个操作，没有论证原因；' +
      '「为了和正弦编码量级匹配」是常见解释而非论文结论。正弦编码理论上不受查表长度限制，但实践中超过训练长度后效果明显下降。</div>';
    return html;
  }

  /* ---------------- 实验 2：因果掩码 ---------------- */
  function runMask() {
    var conf = cfg(), W = model();
    var tgt = conf.mode === 'encoder' ? conf.tokens : conf.targetTokens;
    var base = Object.assign({}, conf, { weights: W, tokens: tgt, targetTokens: tgt });
    var open = E.forward(Object.assign({}, base, { mode: 'encoder', pe: conf.pe }));
    var causal = E.forward(Object.assign({}, base, { mode: 'decoder', pe: conf.pe }));
    var A1 = attentionOf(open, 'self-attn0'), A2 = attentionOf(causal, 'self-attn0');
    var labels = tgt.map(function (t, i) { return '位置' + i + '「' + t + '」'; });
    return '<p class="lab-intro">同一组权重、同一串 token，只切换是否施加因果掩码。上三角代表「偷看未来」的权重；' +
      '掩码在 softmax 之前把未来位置置为 −1e9，softmax 后近似为 0。</p>' +
      '<div class="lab-grid">' +
      '<div class="lab-panel"><h4>双向自注意力（Encoder）</h4>' + smallHeat(A1, labels, labels) +
      '<div class="lab-metric">上三角权重和 = ' + M.fmt(upperSum(A1), 4) + '</div></div>' +
      '<div class="lab-panel"><h4>因果掩码（Decoder）</h4>' + smallHeat(A2, labels, labels) +
      '<div class="lab-metric">上三角权重和 = ' + M.fmt(upperSum(A2), 8) + '</div></div>' +
      '</div><div class="lab-warn">训练时的并行不是「同时生成所有 token」：右移输入 + 因果掩码 + teacher forcing 保证每个位置只基于已知前缀计算损失；' +
      '推理仍然逐 token 串行。论文写 −∞，哈佛实现用 −1e9 作数值近似。</div>';
  }

  /* ---------------- 实验 3：多头分工 ---------------- */
  function runHeads() {
    var conf = cfg(), W = model();
    if (conf.heads < 2) {
      return '<p class="lab-intro">当前只有 1 个头。请到左侧「注意力头数」切换到 2 或 4，再运行本实验。</p>';
    }
    var tokens = conf.mode === 'encoder' ? conf.tokens : conf.targetTokens;
    var heads = [];
    for (var i = 0; i < conf.heads; i++) heads.push(headAttention(i, tokens, conf, W));
    var labels = tokens.map(function (t, k) { return '位置' + k + '「' + t + '」'; });
    var html = '<p class="lab-intro">各头共享同一输入，只是 Q/K/V 投影矩阵不同。实践观察到不同头会分化出不同功能（局部依存、指代、长程关系），' +
      '但这是观察性结论，不是架构保证。各头结果必须经 Wᴼ 融合。</p><div class="lab-grid">';
    heads.forEach(function (h, i) {
      var maxA = 0;
      h.A.forEach(function (r) { r.forEach(function (v) { maxA = Math.max(maxA, v); }); });
      html += '<div class="lab-panel"><h4>头 ' + (i + 1) + ' · d_k=' + conf.dk + '</h4>' +
        smallHeat(h.A, labels, labels) +
        '<div class="lab-metric">最大权重 = ' + M.fmt(maxA, 4) + '</div></div>';
    });
    html += '</div>';
    if (heads.length >= 2) {
      var diff = frob(heads[0].A, heads[1].A);
      html += '<div class="lab-metric">‖A₁ − A₂‖_F = ' + M.fmt(diff, 4) + '</div>';
    }
    html += '<div class="lab-warn">头数 h 翻倍时 d_k 减半（d_k = d_model / h），所以总计算量基本不变——多头几乎是免费的「多视角」。</div>';
    return html;
  }

  /* ---------------- 实验 4：缩放与温度 ---------------- */
  function runScale() {
    var conf = cfg(), result = window.TMApp.state().result;
    var rawStep = stepOf(result, 'self-scores0');
    if (!rawStep) return '<p class="lab-intro">未找到注意力分数步骤。</p>';
    var rawRow = rawStep.value[0];
    var dk = conf.dk;
    function dist(divisor) { return M.softmaxRow(rawRow.map(function (v) { return v / divisor; })); }
    var series = [
      { label: '不缩放', values: dist(1), color: '#f97316' },
      { label: '除以 √d_k', values: dist(Math.sqrt(dk)), color: '#4f46e5' },
      { label: '√d_k · T=0.5（更尖锐）', values: dist(Math.sqrt(dk) * 0.5), color: '#7c3aed' },
      { label: '√d_k · T=2（更平滑）', values: dist(Math.sqrt(dk) * 2), color: '#10b981' }
    ];
    var labels = (conf.mode === 'encoder' ? conf.tokens : conf.targetTokens).map(function (t, i) { return '「' + t + '」'; });
    var chart = V.lineChart(series, { xLabels: labels, xAxis: 'Key 位置', ymin: 0, ymax: 1, width: 640, height: 250 });
    // 模拟：随机 q、k，观察 d_k 增大时是否缩放对最大 softmax 权重的影响
    var rng = E.mulberry32(1234), dks = [4, 16, 64, 256];
    var maxPlain = [], maxScaled = [];
    dks.forEach(function (dim) {
      var q = [], k = [];
      for (var i = 0; i < dim; i++) { q.push(rng() - 0.5); k.push(rng() - 0.5); }
      var score = M.dot(q, k);
      maxPlain.push(M.softmaxRow([score, 0, 0, 0])[0]);
      maxScaled.push(M.softmaxRow([score / Math.sqrt(dim), 0, 0, 0])[0]);
    });
    var chart2 = V.lineChart([
      { label: '不缩放', values: maxPlain, color: '#f97316' },
      { label: '除以 √d_k', values: maxScaled, color: '#4f46e5' }
    ], { xLabels: dks.map(String), xAxis: 'd_k', ymin: 0, ymax: 1, width: 640, height: 220 });

    return '<p class="lab-intro">上半部分用当前网络位置 0 的真实分数，比较不同缩放 / 温度下的注意力分布。' +
      '下半部分用独立同分布的 q、k 模拟：d_k 越大，未缩放的内积方差越大，softmax 越接近 one-hot，梯度越小。</p>' +
      '<div class="lab-grid"><div class="lab-panel"><h4>当前网络的注意力分布（位置 0）</h4>' + chart + '</div>' +
      '<div class="lab-panel"><h4>d_k 增大时的饱和趋势（模拟）</h4>' + chart2 + '</div></div>' +
      '<div class="lab-warn">[论文] 脚注给出的理由是：q、k 各分量均值 0、方差 1 时，内积方差为 d_k，缩放是为了避免 softmax 进入梯度极小的饱和区。' +
      '准确说法是「控制方差、防止梯度消失」，而不是笼统的「防止数值溢出」。论文原文还指出：d_k 大时，不缩放的加性注意力反而优于点积注意力。</div>';
  }

  /* ---------------- 实验 5：注意力复杂度 ---------------- */
  function runComplexity() {
    var conf = cfg();
    var qLen = conf.mode === 'cross' ? conf.targetTokens.length : conf.tokens.length;
    var kLen = conf.tokens.length;
    var cells = qLen * kLen;
    var lens = [], quad = [];
    for (var L = 2; L <= 16; L++) { lens.push(L); quad.push(L * L); }
    var chart = V.lineChart([
      { label: '自注意力分数格数 n²', values: quad, color: '#4f46e5' }
    ], { xLabels: lens.map(String), xAxis: '序列长度 n', ymin: 0, ymax: 256, width: 640, height: 240 });
    var samples = [2, 8, 16, 64, 512, 1024];
    var rows = samples.map(function (n) {
      return '<tr><td>' + n + '</td><td>' + (n * n).toLocaleString() + '</td><td>' +
        M.fmt((n * n) / (cells || 1), 1) + '×</td></tr>';
    }).join('');
    return '<p class="lab-intro">自注意力要为每对 Query/Key 算一个分数：单头 n×n，h 头合计 h·n²。' +
      '下面以当前配置（Query 长度 ' + qLen + '，Key 长度 ' + kLen + '，注意力矩阵 ' + qLen + '×' + kLen +
      ' = ' + cells + ' 格）作参照，观察分数格数怎样随序列长度平方增长。</p>' +
      '<div class="lab-grid"><div class="lab-panel"><h4>n 从 2 到 16：分数格数 n²</h4>' + chart + '</div>' +
      '<div class="lab-panel"><h4>更长序列的格数</h4>' +
      '<table class="matrix-table"><thead><tr><th>n</th><th>n² 格</th><th>相对当前</th></tr></thead><tbody>' +
      rows + '</tbody></table><div class="lab-metric">当前配置：' + qLen + ' × ' + kLen + ' = ' + cells + ' 格</div></div></div>' +
      '<div class="lab-warn">[论文] Table 1：自注意力层复杂度 O(n²·d)，循环层 O(n·d²)，逐位置 FFN O(n·d²)。' +
      '当 n &lt; d 时自注意力更省；n 很大时 n² 成为主要瓶颈，稀疏注意力与线性注意力都为了绕开这一项。' +
      '本页玩具规模 n ≤ 12，平方项尚不明显；把 n 换成 2048 就能看出量级。</div>';
  }

  /* ---------------- 渲染 ---------------- */
  function tabs() {
    $('labTabs').innerHTML = EXPERIMENTS.map(function (e) {
      return '<button type="button" class="lab-tab' + (state.active === e.id ? ' active' : '') +
        '" data-id="' + e.id + '">' + e.label + '</button>';
    }).join('');
    Array.prototype.forEach.call($('labTabs').querySelectorAll('.lab-tab'), function (b) {
      b.addEventListener('click', function () {
      state.active = b.dataset.id;
      run();
      });
    });
  }
  function run() {
    try {
      var fn = {
        pe: runPE, mask: runMask, heads: runHeads, scale: runScale, complexity: runComplexity
      }[state.active];
      $('labBody').innerHTML = fn();
      state.dirty = false;
      tabs();
    } catch (e) {
      $('labBody').innerHTML = '<div class="lab-warn">实验运行失败：' + esc(e.message) + '</div>';
    }
  }

  window.TMLab = {
    invalidate: function () { state.dirty = true; },
    run: run
  };

  function boot() {
    tabs();
    var btn = document.getElementById('btnRunLab');
    if (btn) btn.addEventListener('click', run);
    // app.js 的 init 稍后才完成第一次 forward，这里先只画标签；
    // app.js 会在 recompute 之后主动调用 run()。
    if (window.TMApp && window.TMApp.state().config) run();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
