/* =============================================================
 * Transformer 可视化实验室 · 交互控制层
 * 负责：读取控件 → 创建模型 → 播放步骤 → 权重编辑 → 导入导出
 * 数值全部来自 TMEngine，界面全部交给 TMVis，本文件不写公式。
 * ============================================================= */
(function () {
  'use strict';
  var E = window.TMEngine, V = window.TMVis;
  var $ = function (id) { return document.getElementById(id); };

  var state = {
    config: null, model: null, result: null,
    stepIndex: 0, timer: null, head: 0, signature: '', playing: false
  };

  function tokenize(s) {
    return String(s || '').trim().split(/\s+/).filter(Boolean).slice(0, 12);
  }
  function clone(o) { return JSON.parse(JSON.stringify(o)); }
  function signature(cfg) {
    return [cfg.dModel, cfg.heads, cfg.dff, cfg.layers, cfg.pe, cfg.norm, cfg.seed, cfg.vocab.join('|')].join('#');
  }

  /* ---------------- 读取配置（可选继承当前权重） ---------------- */
  function readConfig(preserveWeights) {
    var raw = {
      tokens: tokenize($('srcTokens').value),
      targetTokens: tokenize($('tgtTokens').value),
      mode: $('mode').value,
      dModel: Number($('dModel').value),
      heads: Number($('heads').value),
      dff: Number($('dff').value),
      layers: Number($('layers').value),
      pe: $('pe').value,
      norm: $('norm').value,
      ffnAct: $('ffnAct').value,
      scale: $('scale').checked,
      temperature: Number($('temperature').value),
      seed: Number($('seed').value)
    };
    var model = E.create(raw);
    var sig = signature(model.config);
    if (preserveWeights && state.model && state.signature === sig) {
      var src = state.model.weights, dst = model.weights;
      ['Wo', 'W1', 'b1', 'W2', 'b2', 'E', 'PE'].forEach(function (k) {
        if (Array.isArray(src[k]) && Array.isArray(dst[k])) dst[k] = clone(src[k]);
      });
      if (Array.isArray(src.heads) && src.heads.length === dst.heads.length) dst.heads = clone(src.heads);
    }
    state.signature = sig;
    return model;
  }

  function recompute(preserveWeights) {
    pause();
    state.model = readConfig(!!preserveWeights);
    state.config = state.model.config;
    state.result = state.model.forward();
    state.stepIndex = 0;
    state.head = Math.min(state.head, state.config.heads - 1);
    renderHeadTabs();
    renderWeightGrid();
    renderAll();
  }

  /* ---------------- 渲染 ---------------- */
  function renderAll() {
    var cfg = state.config, r = state.result;
    var summary = '源序列 n=' + cfg.tokens.length + ' · d_model=' + cfg.dModel + ' · h=' + cfg.heads +
      ' · d_k=' + cfg.dk + ' · d_ff=' + cfg.dff + ' · 注意力矩阵 ' +
      (cfg.mode === 'cross' ? cfg.targetTokens.length + '×' + cfg.tokens.length : cfg.tokens.length + '×' + cfg.tokens.length);
    $('shapeSummary').textContent = summary;
    $('panoramaStat').textContent = cfg.layers + ' 层 · ' + r.steps.length + ' 个计算步骤';

    var hint = '当前：' + (cfg.norm === 'post' ? 'Post-Norm（论文写法，先残差相加再 LN）' : 'Pre-Norm（哈佛实现 / 现代大模型，先 LN 再进子层）');
    if (cfg.pe === 'none') hint += '；位置编码已关闭，注意力对 token 顺序不敏感。';
    if (!cfg.scale) hint += '；未缩放，d_k 较大时 softmax 容易饱和。';
    $('mechanismHint').textContent = hint;

    $('statusBadge').textContent = '已计算 · ' + r.steps.length + ' 步';
    V.renderPanorama($('panorama'), cfg, r.steps[state.stepIndex].focus, function (focus) {
      var idx = r.steps.findIndex(function (s) { return (s.focus || []).indexOf(focus) >= 0; });
      if (idx >= 0) { pause(); setStep(idx); }
    });
    renderChips();
    setStep(state.stepIndex);
    if (window.TMLab) window.TMLab.invalidate();
  }

  function renderChips() {
    var box = $('stepChips');
    box.innerHTML = state.result.steps.map(function (s, i) {
      return '<button type="button" class="step-chip" data-i="' + i + '" title="' +
        V.esc(s.title + ' · ' + s.formula) + '">' + (i + 1) + '. ' + V.esc(s.title) + '</button>';
    }).join('');
    Array.prototype.forEach.call(box.querySelectorAll('.step-chip'), function (chip) {
      chip.addEventListener('click', function () { pause(); setStep(Number(chip.dataset.i)); });
    });
  }

  /* 只在容器内部横向滚动，绝不触发整页 jump —— 与 BP 项目同一处修复 */
  function scrollChipIntoView(container, el) {
    var cr = container.getBoundingClientRect(), er = el.getBoundingClientRect();
    if (er.left < cr.left) container.scrollLeft -= (cr.left - er.left) + 14;
    else if (er.right > cr.right) container.scrollLeft += (er.right - cr.right) + 14;
  }

  function highlightPanorama(focus) {
    var nodes = $('panorama').querySelectorAll('.pano-node');
    Array.prototype.forEach.call(nodes, function (n) {
      n.classList.toggle('active', !!focus && focus.indexOf(n.getAttribute('data-focus')) >= 0);
    });
  }

  function setStep(i) {
    var steps = state.result.steps;
    i = Math.max(0, Math.min(steps.length - 1, i));
    state.stepIndex = i;
    var s = steps[i];
    $('stepCounter').textContent = '第 ' + (i + 1) + ' / ' + steps.length + ' 步';
    $('stepTitle').textContent = s.title;
    $('stepFormula').textContent = s.formula;
    $('stepDetail').textContent = s.detail || '';
    $('stepShape').textContent = s.shape;
    $('stepSource').textContent = s.source || '—';
    V.renderStep($('stepBody'), s, { config: state.config });
    highlightPanorama(s.focus);

    var chips = $('stepChips').querySelectorAll('.step-chip');
    Array.prototype.forEach.call(chips, function (chip, k) {
      chip.classList.toggle('active', k === i);
      chip.classList.toggle('done', k < i);
    });
    if (chips[i]) scrollChipIntoView($('stepChips'), chips[i]);

    $('btnPrev').disabled = i === 0;
    $('btnNext').disabled = i === steps.length - 1;
  }

  /* ---------------- 播放器 ---------------- */
  function pause() {
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    state.playing = false;
    $('btnPlay').textContent = '▶ 自动播放';
  }
  function play() {
    if (state.playing) { pause(); return; }
    if (state.stepIndex >= state.result.steps.length - 1) setStep(0);
    state.playing = true;
    $('btnPlay').textContent = '⏸ 暂停';
    state.timer = setInterval(function () {
      if (state.stepIndex >= state.result.steps.length - 1) { pause(); return; }
      setStep(state.stepIndex + 1);
    }, 850);
  }

  /* ---------------- 权重编辑器 ---------------- */
  function renderHeadTabs() {
    var n = state.config.heads, h = '<button type="button" class="head-tab" data-h="0">头 1</button>';
    for (var i = 1; i < n; i++) h += '<button type="button" class="head-tab" data-h="' + i + '">头 ' + (i + 1) + '</button>';
    $('headTabs').innerHTML = h;
    Array.prototype.forEach.call($('headTabs').querySelectorAll('.head-tab'), function (tab) {
      tab.classList.toggle('active', Number(tab.dataset.h) === state.head);
      tab.addEventListener('click', function () {
        state.head = Number(tab.dataset.h);
        renderHeadTabs(); renderWeightGrid();
      });
    });
  }

  function matrixInputs(M, key, head) {
    var rows = M.length, cols = M[0] ? M[0].length : 0;
    var h = '<div class="weight-matrix" style="grid-template-columns:repeat(' + cols + ',minmax(0,1fr))">';
    for (var i = 0; i < rows; i++) for (var j = 0; j < cols; j++) {
      h += '<input type="number" step="0.05" data-key="' + key + '" data-head="' + head +
        '" data-i="' + i + '" data-j="' + j + '" value="' + E.matrices.fmt(M[i][j], 4) + '" aria-label="' +
        key + '[' + i + '][' + j + ']">';
    }
    return h + '</div>';
  }

  function renderWeightGrid() {
    var head = state.head, hw = state.model.weights.heads[head];
    if (!hw) return;
    $('weightGrid').innerHTML =
      '<div class="weight-block"><h4>W_Q <span>Query 投影 · ' + hw.Wq.length + '×' + hw.Wq[0].length + '</span></h4>' +
        matrixInputs(hw.Wq, 'Wq', head) + '</div>' +
      '<div class="weight-block"><h4>W_K <span>Key 投影 · ' + hw.Wk.length + '×' + hw.Wk[0].length + '</span></h4>' +
        matrixInputs(hw.Wk, 'Wk', head) + '</div>' +
      '<div class="weight-block"><h4>W_V <span>Value 投影 · ' + hw.Wv.length + '×' + hw.Wv[0].length + '</span></h4>' +
        matrixInputs(hw.Wv, 'Wv', head) + '</div>';
  }

  function applyWeightEdits() {
    var inputs = $('weightGrid').querySelectorAll('input[data-key]');
    Array.prototype.forEach.call(inputs, function (inp) {
      var head = Number(inp.dataset.head), key = inp.dataset.key;
      var i = Number(inp.dataset.i), j = Number(inp.dataset.j);
      var v = Number(inp.value);
      if (!Number.isFinite(v)) return;
      var M = state.model.weights.heads[head][key];
      if (M && M[i] && j < M[i].length) M[i][j] = v;
    });
    recompute(true);
  }

  function fillPreset() {
    var name = $('presetSelect').value;
    var cfg = state.config, d = cfg.dModel, dk = cfg.dk, head = state.head;
    var I = E.matrices.zeros(d, dk);
    for (var i = 0; i < Math.min(d, dk); i++) I[i][i] = 1;
    var K = name === 'random' ? null : E.presetMatrix(name, d, dk);
    var hw = state.model.weights.heads[head];
    hw.Wq = I.map(function (r) { return r.slice(); });
    hw.Wk = K ? K.map(function (r) { return r.slice(); }) : hw.Wk.map(function (r) { return r.slice(); });
    if (name !== 'random') hw.Wv = I.map(function (r) { return r.slice(); });
    renderWeightGrid();
    $('presetHint').textContent = '已填入「' + E.WEIGHT_PRESETS[name].label + '」：' + E.WEIGHT_PRESETS[name].desc +
      '。点击「应用权重并重算」后，注意力热力图会按新权重更新。';
  }

  /* ---------------- 导入导出 ---------------- */
  function exportConfig() {
    var payload = { version: E.version, config: state.config, weights: state.model.weights };
    var blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'transformer-lab-config.json';
    a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
  }
  function importConfig(file) {
    var reader = new FileReader();
    reader.onload = function () {
      try {
        var data = JSON.parse(reader.result);
        var cfg = Object.assign({}, E.DEFAULT_CONFIG, data.config || {});
        cfg.weights = data.weights || null;
        var m = E.create(cfg);
        state.model = m; state.config = m.config; state.signature = signature(m.config);
        Object.keys(cfg).forEach(function (k) {
          var el = $(k === 'tokens' ? 'srcTokens' : k === 'targetTokens' ? 'tgtTokens' : k);
          if (!el) return;
          if (el.type === 'checkbox') el.checked = !!cfg[k];
          else if (el.tagName === 'SELECT' || el.type === 'number') el.value = cfg[k];
          else if (el.type === 'text') el.value = (cfg[k] || []).join(' ');
        });
        recompute(false);
        $('statusBadge').textContent = '已导入配置';
      } catch (e) {
        $('statusBadge').textContent = '导入失败：' + e.message;
        $('statusBadge').className = 'badge badge-warn';
      }
    };
    reader.readAsText(file);
  }

  /* ---------------- 初始化 ---------------- */
  function init() {
    var preset = $('presetSelect');
    Object.keys(E.WEIGHT_PRESETS).forEach(function (k) {
      var o = document.createElement('option');
      o.value = k; o.textContent = E.WEIGHT_PRESETS[k].label;
      preset.appendChild(o);
    });
    preset.addEventListener('change', function () {
      $('presetHint').textContent = E.WEIGHT_PRESETS[preset.value].desc;
    });

    ['srcTokens', 'tgtTokens', 'mode', 'dModel', 'heads', 'dff', 'layers', 'pe', 'norm', 'ffnAct', 'temperature', 'seed']
      .forEach(function (id) { $(id).addEventListener('change', function () { recompute(true); }); });
    $('scale').addEventListener('change', function () { recompute(true); });

    $('btnFirst').addEventListener('click', function () { pause(); setStep(0); });
    $('btnPrev').addEventListener('click', function () { pause(); setStep(state.stepIndex - 1); });
    $('btnNext').addEventListener('click', function () { pause(); setStep(state.stepIndex + 1); });
    $('btnPlay').addEventListener('click', play);
    $('btnApplyWeights').addEventListener('click', applyWeightEdits);
    $('btnFillPreset').addEventListener('click', fillPreset);
    $('btnResetWeights').addEventListener('click', function () {
      state.model.resetWeights(); recompute(false);
      $('presetHint').textContent = '已恢复为标准 Xavier 初始化。';
    });
    $('btnExport').addEventListener('click', exportConfig);
    $('btnImport').addEventListener('click', function () { $('importFile').click(); });
    $('importFile').addEventListener('change', function (e) { if (e.target.files[0]) importConfig(e.target.files[0]); });

    recompute(false);
    $('presetHint').textContent = E.WEIGHT_PRESETS[preset.value].desc;
    if (window.TMLab) window.TMLab.run();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();

  /* 供实验台与自动化测试使用 */
  window.TMApp = {
    state: function () {
      return {
        config: state.config, weights: state.model ? state.model.weights : null,
        result: state.result, stepIndex: state.stepIndex,
        steps: state.result ? state.result.steps : []
      };
    },
    setStep: setStep,
    recompute: recompute,
    pause: pause
  };
})();
