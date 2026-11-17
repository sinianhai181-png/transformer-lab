/* =============================================================
 * Transformer 可视化实验室 · 展示层
 * window.TMVis：全景结构图 / 数值矩阵 / 注意力热力图 / 折线图
 * 只做渲染，不做业务计算；所有数据来自 TMEngine 的 forward 快照。
 * ============================================================= */
(function (root) {
  'use strict';
  var E = root.TMEngine;
  var fmt = E.matrices.fmt;

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  /* ---------------- 悬浮提示（全局事件委托，只绑定一次） ---------------- */
  function bindTooltip() {
    var tip = document.getElementById('tooltip');
    if (!tip || tip.dataset.bound) return;
    tip.dataset.bound = '1';
    document.addEventListener('mousemove', function (ev) {
      var el = ev.target.closest ? ev.target.closest('[data-tip]') : null;
      if (!el) { tip.hidden = true; return; }
      tip.textContent = el.getAttribute('data-tip');
      tip.hidden = false;
      var x = Math.min(ev.clientX + 14, window.innerWidth - tip.offsetWidth - 12);
      var y = Math.min(ev.clientY + 16, window.innerHeight - tip.offsetHeight - 12);
      tip.style.left = Math.max(6, x) + 'px';
      tip.style.top = Math.max(6, y) + 'px';
    });
    document.addEventListener('mouseleave', function () { tip.hidden = true; });
  }
  if (typeof document !== 'undefined') bindTooltip();

  /* ---------------- 颜色映射 ---------------- */
  function cellStyle(v, mode, maxAbs) {
    var a, bg, fg = '#1e293b';
    if (mode === 'attention') {
      a = clamp(v, 0, 1);
      bg = 'rgba(79,70,229,' + (0.05 + 0.72 * a).toFixed(3) + ')';
      if (a > 0.45) fg = '#fff';
    } else if (mode === 'diverging') {
      var m = maxAbs || 1;
      a = clamp(Math.abs(v) / m, 0, 1);
      bg = v >= 0 ? 'rgba(59,130,246,' + (0.05 + 0.55 * a).toFixed(3) + ')'
                  : 'rgba(249,115,22,' + (0.05 + 0.55 * a).toFixed(3) + ')';
      if (a > 0.62) fg = '#fff';
    } else {
      bg = '#f8fafc';
    }
    return 'background:' + bg + ';color:' + fg;
  }
  function maxAbsOf(M) {
    var m = 0;
    M.forEach(function (r) { r.forEach(function (v) { m = Math.max(m, Math.abs(v)); }); });
    return m || 1;
  }

  /* ---------------- 矩阵表 ---------------- */
  function matrixHTML(M, opts) {
    opts = opts || {};
    if (!Array.isArray(M)) return '<div class="text-note">该步骤不是矩阵数据。</div>';
    var is1D = !Array.isArray(M[0]);
    var rows = is1D ? [M] : M;
    var cols = is1D ? M.length : (M[0] ? M[0].length : 0);
    var maxAbs = opts.colorize === 'diverging' ? maxAbsOf(rows) : 1;
    var rowLabels = opts.rowLabels || [];
    var colLabels = opts.colLabels || [];
    var h = '<div class="matrix-wrap">';
    if (opts.title) h += '<div class="matrix-caption">' + esc(opts.title) +
      (opts.tag ? '<span class="tag">' + esc(opts.tag) + '</span>' : '') + '</div>';
    h += '<table class="matrix-table"><thead><tr><th></th>';
    for (var j = 0; j < cols; j++) h += '<th>' + esc(colLabels[j] !== undefined ? colLabels[j] : 'j=' + j) + '</th>';
    h += '</tr></thead><tbody>';
    for (var i = 0; i < rows.length; i++) {
      h += '<tr><th>' + esc(rowLabels[i] !== undefined ? rowLabels[i] : 'i=' + i) + '</th>';
      for (var k = 0; k < cols; k++) {
        var v = rows[i][k];
        var tip = '第 ' + (i + 1) + ' 行 · 第 ' + (k + 1) + ' 列 = ' + Number(v).toFixed(6) +
          (opts.colorize === 'attention' ? '\n该位置被分配到的注意力权重' : '');
        h += '<td class="heat-cell' + (i === k ? ' diag' : '') + '" style="' +
          cellStyle(v, opts.colorize, maxAbs) + '" data-tip="' + esc(tip) + '">' + fmt(v, 3) + '</td>';
      }
      h += '</tr>';
    }
    h += '</tbody></table></div>';
    if (opts.colorize === 'attention') {
      h += '<div class="legend"><span>0</span><span class="legend-bar"></span><span>1</span><span>颜色越深 = 权重越大</span></div>';
    } else if (opts.colorize === 'diverging') {
      h += '<div class="legend"><span>负（橙）</span><span style="width:60px;height:8px;border-radius:4px;background:linear-gradient(90deg,#fb923c,#f8fafc,#60a5fa)"></span><span>正（蓝）</span></div>';
    }
    return h;
  }

  /* ---------------- 三种 / 分组矩阵 ---------------- */
  function tripleHTML(obj, opts) {
    opts = opts || {};
    var keys = Object.keys(obj || {});
    if (!keys.length) return '<div class="text-note">无数据</div>';
    var tips = opts.tips || {};
    var colors = opts.colors || {};
    var lab = opts.labels || {};
    var h = '<div class="triple-grid">';
    keys.forEach(function (k) {
      h += '<div class="triple-cell">' + matrixHTML(obj[k], {
        title: lab[k] || k,
        tag: tips[k] || '',
        colorize: colors[k] || 'none',
        rowLabels: (opts.rowLabels || []),
        colLabels: (opts.colLabels || [])
      }) + '</div>';
    });
    return h + '</div>';
  }

  function tokensHTML(list) {
    if (!Array.isArray(list)) return '<div class="text-note">无数据</div>';
    return '<div class="token-list">' + list.map(function (item) {
      var token = typeof item === 'string' ? item : item.token;
      var index = typeof item === 'string' ? '' : item.index;
      return '<div class="token-card">' + esc(token) + (index !== '' ? '<small>位置 ' + index + '</small>' : '') + '</div>';
    }).join('') + '</div>';
  }

  /* ---------------- 步骤数值面板 ---------------- */
  function renderStep(container, step, ctx) {
    if (!container || !step) return;
    var v = step.value, meta = step.meta || {}, html = '';
    if (step.kind === 'tokens') {
      html = tokensHTML(v) + '<div class="text-note">' + esc(step.detail || '') + '</div>';
    } else if (step.kind === 'triple') {
      var isWeight = Object.keys(v || {}).some(function (k) { return k.indexOf('W_') === 0; });
      html = tripleHTML(v, {
        labels: isWeight ? { W_Q: 'Query 投影 W_Q', W_K: 'Key 投影 W_K', W_V: 'Value 投影 W_V' } :
          { Q: 'Query 矩阵 Q', K: 'Key 矩阵 K', V: 'Value 矩阵 V' },
        tips: isWeight ? { W_Q: 'd_model × d_k', W_K: 'd_model × d_k', W_V: 'd_model × d_k' } :
          { Q: 'n × d_k', K: 'm × d_k', V: 'm × d_k' },
        rowLabels: meta.rowLabels || [], colLabels: meta.colLabels || []
      });
    } else if (step.kind === 'heat') {
      html = matrixHTML(v, {
        colorize: step.id.indexOf('probs') >= 0 ? 'attention' : 'attention',
        rowLabels: meta.rowLabels || [], colLabels: meta.colLabels || []
      });
    } else if (step.kind === 'text') {
      html = '<div class="text-note">' + esc(step.detail || '该步骤无矩阵输出。') + '</div>';
    } else {
      html = matrixHTML(v, {
        colorize: step.id.indexOf('scores') >= 0 || step.id.indexOf('scale') >= 0 || step.id.indexOf('mask') >= 0 ? 'diverging' : 'none',
        title: '数值矩阵',
        rowLabels: meta.rowLabels || [], colLabels: meta.colLabels || []
      });
    }
    container.innerHTML = html;
  }

  /* ---------------- 全景结构图 ---------------- */
  function node(focus, x, y, w, h, label, sub, fill, stroke) {
    return '<g class="pano-node" data-focus="' + focus + '">' +
      '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="9" fill="' + fill + '" stroke="' + stroke + '"/>' +
      '<text x="' + (x + w / 2) + '" y="' + (y + (sub ? h / 2 - 3 : h / 2 + 4)) + '" text-anchor="middle" class="pano-label">' + esc(label) + '</text>' +
      (sub ? '<text x="' + (x + w / 2) + '" y="' + (y + h / 2 + 13) + '" text-anchor="middle" class="pano-small">' + esc(sub) + '</text>' : '') +
      '</g>';
  }
  function edge(d) { return '<path class="pano-edge" d="' + d + '"/>'; }

  function panoramaSVG(cfg) {
    var mode = cfg.mode, heads = cfg.heads, d = cfg.dModel;
    var s = '<svg viewBox="0 0 1000 330" role="img" aria-label="Transformer 结构全景图">' +
      '<defs><marker id="tmArrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">' +
      '<path d="M0,0 L8,4 L0,8 z" fill="#cbd5e1"/></marker></defs>';

    // 输入区
    s += '<rect class="pano-group" x="16" y="18" width="176" height="284" rx="14"/>';
    s += '<text x="24" y="38" class="pano-title">输入</text>';
    s += node('input', 32, 52, 144, 50, 'Token Embedding', 'E(token) × √d', '#eef2ff', '#c7d2fe');
    s += node('pe', 32, 114, 144, 46, '位置编码', cfg.pe === 'none' ? '已关闭' : (cfg.pe === 'sinusoidal' ? '正弦 / 余弦' : '可学习查表'), '#f5f3ff', '#ddd6fe');
    s += '<text x="104" y="186" text-anchor="middle" class="pano-small">x₀ = Embedding + PE</text>';

    // Encoder 栈
    s += '<rect class="pano-group" x="224" y="18" width="308" height="' + (mode === 'encoder' ? 284 : 146) + '" rx="14"/>';
    s += '<text x="236" y="38" class="pano-title">Encoder × ' + cfg.layers + '</text>';
    s += node('attn', 242, 52, 152, 48, '多头自注意力', 'Q=K=V · 双向', '#dbeafe', '#93c5fd');
    s += node('norm', 404, 52, 112, 48, 'Add & Norm', '残差 + LN', '#f1f5f9', '#cbd5e1');
    s += node('ffn', 242, 114, 152, 48, '逐位置 FFN', 'd_ff = ' + cfg.dff, '#dbeafe', '#93c5fd');
    s += node('norm', 404, 114, 112, 48, 'Add & Norm', '残差 + LN', '#f1f5f9', '#cbd5e1');
    s += edge('M194,77 L238,77');
    s += edge('M176,77 Q200,77 194,77');
    s += edge('M394,76 L400,76');
    s += edge('M320,100 L320,110');
    s += edge('M394,138 L400,138');
    if (mode === 'encoder') {
      s += '<text x="378" y="196" text-anchor="middle" class="pano-small">Z ∈ ℝ^(n × ' + d + ')</text>';
      s += edge('M378,162 L378,190');
      s += node('output', 270, 206, 212, 62, '共享 embedding → Softmax', 'logits = H · Eᵀ，输出下一 token 概率', '#d1fae5', '#6ee7b7');
      s += edge('M378,190 L378,202');
    }

    // Decoder 栈（decoder / cross 模式显示）
    if (mode !== 'encoder') {
      s += '<rect class="pano-group" x="560" y="18" width="290" height="' + (mode === 'cross' ? 284 : 190) + '" rx="14"/>';
      s += '<text x="572" y="38" class="pano-title">Decoder × ' + cfg.layers + '</text>';
      s += node('attn', 578, 52, 150, 46, '掩码自注意力', 'causal mask · j ≤ i', '#ede9fe', '#c4b5fd');
      s += node('norm', 736, 52, 100, 46, 'Add & Norm', '残差 + LN', '#f1f5f9', '#cbd5e1');
      if (mode === 'cross') {
        s += node('cross', 578, 116, 150, 46, '交叉注意力', 'Q=dec, K/V=Z', '#ede9fe', '#c4b5fd');
        s += node('norm', 736, 116, 100, 46, 'Add & Norm', '残差 + LN', '#f1f5f9', '#cbd5e1');
      }
      s += node('ffn', 578, (mode === 'cross' ? 186 : 116), 150, 46, '逐位置 FFN', 'ReLU / GELU', '#ede9fe', '#c4b5fd');
      s += node('norm', 736, (mode === 'cross' ? 186 : 116), 100, 46, 'Add & Norm', '残差 + LN', '#f1f5f9', '#cbd5e1');
      s += edge('M728,75 L732,75');
      if (mode === 'cross') {
        s += edge('M728,139 L732,139');
        // Encoder 输出 Z → 交叉注意力的 K/V
        s += '<path class="pano-edge" d="M532,190 Q552,190 552,139 L574,139"/>';
        s += '<text x="544" y="182" text-anchor="middle" class="pano-small">K/V = Z</text>';
      }
      s += node('output', 578, (mode === 'cross' ? 252 : 196), 258, 46, '共享 embedding → Softmax', '输出下一 token 概率', '#d1fae5', '#6ee7b7');
    }

    s += '<text x="20" y="322" class="pano-small">h = ' + heads + ' 头 · d_k = ' + (d / heads) + ' · 点击任意模块跳转到对应步骤</text>';
    return s + '</svg>';
  }

  function renderPanorama(container, cfg, activeFocus, onNodeClick) {
    if (!container) return;
    container.innerHTML = panoramaSVG(cfg);
    var nodes = container.querySelectorAll('.pano-node');
    nodes.forEach(function (n) {
      var focus = n.getAttribute('data-focus');
      if (activeFocus && activeFocus.indexOf(focus) >= 0) n.classList.add('active');
      n.addEventListener('click', function () { if (onNodeClick) onNodeClick(focus); });
    });
  }

  /* ---------------- 折线图（实验台用） ---------------- */
  function lineChart(series, opts) {
    opts = opts || {};
    var w = opts.width || 560, hgt = opts.height || 220;
    var pad = { l: 42, r: 14, t: 16, b: 30 };
    var n = series.length ? series[0].values.length : 0;
    var ymin = opts.ymin !== undefined ? opts.ymin : 0;
    var ymax = opts.ymax !== undefined ? opts.ymax : 1;
    if (opts.autoMax) {
      ymax = 0;
      series.forEach(function (s) { s.values.forEach(function (v) { ymax = Math.max(ymax, v); }); });
      ymax = ymax * 1.15 || 1;
    }
    var X = function (i) { return pad.l + (n <= 1 ? (w - pad.l - pad.r) / 2 : i * (w - pad.l - pad.r) / (n - 1)); };
    var Y = function (v) { return pad.t + (1 - (v - ymin) / (ymax - ymin || 1)) * (hgt - pad.t - pad.b); };
    var s = '<svg class="lab-chart" viewBox="0 0 ' + w + ' ' + hgt + '">';
    [0, 0.25, 0.5, 0.75, 1].forEach(function (t) {
      var y = pad.t + t * (hgt - pad.t - pad.b);
      var val = ymax - t * (ymax - ymin);
      s += '<line x1="' + pad.l + '" y1="' + y + '" x2="' + (w - pad.r) + '" y2="' + y + '" stroke="#eef2f7"/>' +
        '<text x="' + (pad.l - 6) + '" y="' + (y + 4) + '" text-anchor="end" class="pano-small">' + val.toFixed(2) + '</text>';
    });
    (opts.xLabels || []).forEach(function (lab, i) {
      if (n > 12 && i % 2) return;
      s += '<text x="' + X(i) + '" y="' + (hgt - 9) + '" text-anchor="middle" class="pano-small">' + esc(lab) + '</text>';
    });
    series.forEach(function (se) {
      var pts = se.values.map(function (v, i) { return X(i) + ',' + Y(v); }).join(' ');
      s += '<polyline points="' + pts + '" fill="none" stroke="' + se.color + '" stroke-width="2.4" stroke-linejoin="round"/>';
      se.values.forEach(function (v, i) {
        s += '<circle cx="' + X(i) + '" cy="' + Y(v) + '" r="3.2" fill="' + se.color + '"><title>' +
          esc(se.label + ' · ' + ((opts.xLabels || [])[i] || i) + ' = ' + Number(v).toFixed(4)) + '</title></circle>';
      });
    });
    if (opts.xAxis) s += '<text x="' + (w - pad.r) + '" y="' + (hgt - 9) + '" text-anchor="end" class="pano-small">' + esc(opts.xAxis) + '</text>';
    return s + '</svg>';
  }

  root.TMVis = {
    esc: esc,
    matrixHTML: matrixHTML,
    tripleHTML: tripleHTML,
    tokensHTML: tokensHTML,
    renderStep: renderStep,
    renderPanorama: renderPanorama,
    panoramaSVG: panoramaSVG,
    lineChart: lineChart,
    cellStyle: cellStyle
  };
})(typeof globalThis !== 'undefined' ? globalThis : this);
