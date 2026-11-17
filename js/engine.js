/* =============================================================
 * Transformer 可视化实验室 · 计算核心 (无第三方依赖)
 * 浏览器: window.TMEngine   /   Node: require('./engine.js')
 *
 * 设计约定（与 docs/TRANSFORMER.md 逐节对应）：
 *  1. 张量用二维数组表示：M[i][j] = 第 i 行第 j 列；一维向量用普通数组。
 *  2. 注意力分数 S = QKᵀ / (√d_k · T)，掩码在 softmax 之前置 -1e9（哈佛实现）。
 *  3. 多头拆分采用与 view + transpose 等价的写法：每头独立投影到 d_k = d_model/h。
 *  4. 归一化提供 Post-Norm（论文）与 Pre-Norm（哈佛实现 / 现代大模型）两种。
 *  5. 输出层与 embedding 共享权重（论文要求）：logits = H · Eᵀ。
 *  6. 每次 forward 返回完整步骤快照，供逐环节演示、矩阵面板与热力图使用。
 *
 * 阅读顺序：矩阵工具 -> sinusoidalPE -> create -> forward -> attentionBlock
 * 公式来源标注 [论文] / [实现] / [解读] 与 docs/TRANSFORMER.md 一致。
 * ============================================================= */
(function (root, factory) {
  var api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.TMEngine = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  var EPS = 1e-9;
  var MASK_VALUE = -1e9;   // [实现] 哈佛注解用 -1e9 近似论文的 -∞

  /* 可复现随机数（与 BP 项目同一套，保证同样 seed 得到同样权重） */
  function mulberry32(seed) {
    var a = (seed >>> 0) || 1;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gaussian(rnd) {
    var u = Math.max(rnd(), EPS), v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /* ---------------- 矩阵工具 ---------------- */
  function zeros(rows, cols) {
    var m = [];
    for (var i = 0; i < rows; i++) { m.push(new Array(cols).fill(0)); }
    return m;
  }
  function clone(M) {
    return M.map(function (r) { return Array.isArray(r) ? r.slice() : r; });
  }
  function shape(M) {
    if (!Array.isArray(M)) return '—';
    if (!Array.isArray(M[0])) return M.length + ' 维向量';
    return M.length + ' × ' + M[0].length;
  }
  function transpose(M) {
    var n = M.length, m = M[0] ? M[0].length : 0, T = zeros(m, n);
    for (var i = 0; i < n; i++) for (var j = 0; j < m; j++) T[j][i] = M[i][j];
    return T;
  }
  function matmul(A, B) {
    var n = A.length, k = B.length, m = B[0] ? B[0].length : 0;
    var C = zeros(n, m);
    for (var i = 0; i < n; i++) {
      for (var p = 0; p < k; p++) {
        var aip = A[i][p];
        if (aip === 0) continue;
        for (var j = 0; j < m; j++) C[i][j] += aip * B[p][j];
      }
    }
    return C;
  }
  function add(A, B) { return A.map(function (r, i) { return r.map(function (v, j) { return v + B[i][j]; }); }); }
  function sub(A, B) { return A.map(function (r, i) { return r.map(function (v, j) { return v - B[i][j]; }); }); }
  function scale(A, s) { return A.map(function (r) { return r.map(function (v) { return v * s; }); }); }
  function rowMap(A, fn) { return A.map(function (r) { return r.map(fn); }); }
  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

  /* 稳定 softmax：沿最后一维（key 维）归一化，每行和为 1 */
  function softmaxRow(v) {
    var mx = -Infinity;
    for (var i = 0; i < v.length; i++) if (v[i] > mx) mx = v[i];
    if (!Number.isFinite(mx)) return v.map(function () { return 1 / v.length; }); // 整行被掩码的兜底
    var ex = v.map(function (x) { return Math.exp(x - mx); });
    var sum = ex.reduce(function (a, b) { return a + b; }, 0);
    return ex.map(function (x) { return x / sum; });
  }
  function softmaxRows(M) { return M.map(softmaxRow); }

  /* LayerNorm：在单个位置沿特征维做标准化，γ=1 β=0，eps=1e-6 */
  function layerNormRow(v, eps) {
    eps = eps || 1e-6;
    var mean = v.reduce(function (a, b) { return a + b; }, 0) / v.length;
    var variance = v.reduce(function (a, b) { return a + (b - mean) * (b - mean); }, 0) / v.length;
    var sd = Math.sqrt(variance + eps);
    return v.map(function (x) { return (x - mean) / sd; });
  }
  function layerNorm(M) { return M.map(function (r) { return layerNormRow(r); }); }

  /* FFN 激活：ReLU（论文）与 GELU 近似（现代实现常见） */
  function relu(z) { return Math.max(0, z); }
  function gelu(z) { return 0.5 * z * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (z + 0.044715 * z * z * z))); }
  function activate(z, name) { return name === 'gelu' ? gelu(z) : relu(z); }

  /* ---------------- 位置编码 ---------------- */
  /* [论文] PE(pos,2i) = sin(pos / 10000^(2i/d)), PE(pos,2i+1) = cos(...) */
  function sinusoidalPE(n, d) {
    var P = zeros(n, d);
    for (var pos = 0; pos < n; pos++) {
      for (var i = 0; i < d; i++) {
        var pair = Math.floor(i / 2);
        var angle = pos / Math.pow(10000, 2 * pair / d);
        P[pos][i] = i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
      }
    }
    return P;
  }
  /* [实现] 因果掩码：下三角（含对角线）为 true，上三角为 false */
  function causalMatrix(n) {
    var M = [];
    for (var i = 0; i < n; i++) {
      var row = [];
      for (var j = 0; j < n; j++) row.push(j <= i);
      M.push(row);
    }
    return M;
  }
  function allAllowed(n, m) {
    var M = zeros(n, m);
    return M.map(function (r, i) { return r.map(function (_, j) { return { i: i, j: j, ok: true }; }); });
  }

  /* ---------------- 格式化（界面显示用） ---------------- */
  function fmt(v, digits) {
    if (!Number.isFinite(v)) return String(v);
    digits = digits || 3;
    if (Math.abs(v) < 1e-12) return '0';
    var a = Math.abs(v);
    if (a >= 1e5 || a < 1e-3) return v.toExponential(1);
    return Number(v.toFixed(digits)).toString();
  }
  function fmtMatrix(M, digits) {
    return M.map(function (r) { return '[' + r.map(function (v) { return fmt(v, digits); }).join(', ') + ']'; }).join('\n');
  }

  /* ---------------- 默认配置 ---------------- */
  var DEFAULT_CONFIG = {
    tokens: ['I', 'love', 'transformer'],
    targetTokens: ['<s>', 'I', 'love'],
    dModel: 8,
    heads: 2,
    dff: 16,
    layers: 1,
    pe: 'sinusoidal',      // sinusoidal | learned | none
    norm: 'post',          // post（论文） | pre（哈佛实现 / 现代）
    ffnAct: 'relu',        // relu（论文） | gelu（现代）
    mode: 'encoder',       // encoder | decoder | cross
    scale: true,           // 是否除以 √d_k
    temperature: 1,        // 额外温度系数 T：S = QKᵀ / (√d_k · T)
    seed: 7,
    weights: null          // 可编辑权重覆盖
  };

  function intClamp(v, lo, hi, dflt) {
    v = Number(v);
    if (!Number.isFinite(v)) return dflt;
    return Math.max(lo, Math.min(hi, Math.round(v)));
  }
  function normalize(raw) {
    var c = Object.assign({}, DEFAULT_CONFIG, raw || {});
    c.dModel = intClamp(c.dModel, 2, 32, 8);
    c.dff = intClamp(c.dff, 2, 64, c.dModel * 2);
    c.layers = intClamp(c.layers, 1, 3, 1);
    c.heads = intClamp(c.heads, 1, 8, 2);
    // d_k = d_model / h 必须为整数（论文 base: 512/8 = 64）
    while (c.heads > 1 && c.dModel % c.heads !== 0) c.heads--;
    if (['sinusoidal', 'learned', 'none'].indexOf(c.pe) < 0) c.pe = 'sinusoidal';
    if (['post', 'pre'].indexOf(c.norm) < 0) c.norm = 'post';
    if (['relu', 'gelu'].indexOf(c.ffnAct) < 0) c.ffnAct = 'relu';
    if (['encoder', 'decoder', 'cross'].indexOf(c.mode) < 0) c.mode = 'encoder';
    c.scale = !!c.scale;
    c.temperature = Math.max(0.1, Math.min(5, Number(c.temperature) || 1));
    c.seed = intClamp(c.seed, 1, 999999, 7);
    var clean = function (arr, fallback) {
      var out = (Array.isArray(arr) ? arr : []).map(function (t) { return String(t).trim() || '·'; }).slice(0, 12);
      return out.length ? out : fallback.slice();
    };
    c.tokens = clean(c.tokens, DEFAULT_CONFIG.tokens);
    c.targetTokens = clean(c.targetTokens, DEFAULT_CONFIG.targetTokens);
    c.vocab = [];
    c.tokens.concat(c.targetTokens).forEach(function (t) { if (c.vocab.indexOf(t) < 0) c.vocab.push(t); });
    c.dk = c.dModel / c.heads;
    return c;
  }

  /* ---------------- 权重初始化 ---------------- */
  /* Xavier / Glorot uniform：limit = √(6 / (fanIn + fanOut))，[实现] 哈佛注解的做法 */
  function xavier(rows, cols, rnd) {
    var limit = Math.sqrt(6 / (rows + cols));
    return zeros(rows, cols).map(function (r) { return r.map(function () { return (rnd() * 2 - 1) * limit; }); });
  }
  function bias(rows, rnd) { return zeros(rows, 1).map(function (r) { return r.map(function () { return (rnd() - 0.5) * 0.2; }); }).map(function (r) { return r[0]; }); }

  function initWeights(cfg, override) {
    var rnd = mulberry32(cfg.seed);
    var V = cfg.vocab.length, d = cfg.dModel, dk = cfg.dk, h = cfg.heads, dff = cfg.dff;
    var w = {
      heads: [],
      Wo: xavier(d, d, rnd),
      W1: xavier(d, dff, rnd),
      b1: bias(dff, rnd),
      W2: xavier(dff, d, rnd),
      b2: bias(d, rnd),
      E: xavier(V, d, rnd),
      PE: null
    };
    for (var i = 0; i < h; i++) {
      w.heads.push({ Wq: xavier(d, dk, rnd), Wk: xavier(d, dk, rnd), Wv: xavier(d, dk, rnd) });
    }
    if (cfg.pe === 'learned') {
      w.PE = zeros(12, d).map(function (r) { return r.map(function () { return gaussian(rnd) * 0.1; }); });
    }
    if (override && typeof override === 'object') {
      ['Wo', 'W1', 'b1', 'W2', 'b2', 'E', 'PE'].forEach(function (k) {
        if (Array.isArray(override[k])) w[k] = clone(override[k]);
      });
      if (Array.isArray(override.heads)) {
        override.heads.slice(0, h).forEach(function (hh, i) {
          ['Wq', 'Wk', 'Wv'].forEach(function (k) { if (hh && Array.isArray(hh[k])) w.heads[i][k] = clone(hh[k]); });
        });
      }
    }
    return w;
  }

  /* ---------------- 单步注意力 ---------------- */
  /* 返回未缩放 / 缩放 / 掩码 / 注意力 / 上下文，供界面逐步展示 */
  function attention(Q, K, V, opts) {
    opts = opts || {};
    var raw = matmul(Q, transpose(K));                    // S = QKᵀ
    var dk = opts.dk || Q[0].length;
    var divisor = Math.sqrt(dk) * (opts.temperature || 1);
    var scaled = opts.scale === false ? clone(raw) : scale(raw, 1 / divisor);
    var mask = opts.mask;                                  // 布尔矩阵：true = 允许
    var masked = mask ? scaled.map(function (r, i) {
      return r.map(function (v, j) { return mask[i] && mask[i][j] ? v : MASK_VALUE; });
    }) : clone(scaled);
    var A = softmaxRows(masked);
    var O = matmul(A, V);                                  // 每个位置是所有 value 的凸组合
    return { raw: raw, scaled: scaled, masked: masked, A: A, O: O, divisor: divisor };
  }

  /* ---------------- 记录步骤 ---------------- */
  function rec(steps, id, title, formula, value, opts) {
    // 标题里的 ①…⑳ 只是编写时的顺序提示；运行时统一用自动序号，避免多层 / cross 模式重复。
    var no = steps.length + 1;
    title = String(title).replace(/^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]+[a-z]?\s*/, '') || '计算步骤';
    var step = {
      no: no, id: id, title: title, formula: formula, value: value,
      shape: shape(value), kind: 'matrix', source: '', detail: '', focus: [], meta: null
    };
    steps.push(Object.assign(step, opts || {}));
  }

  /* ---------------- 模型 ---------------- */
  function create(rawConfig) {
    var cfg = normalize(rawConfig);
    var weights = initWeights(cfg, cfg.weights);
    cfg.weights = null; // 已并入 weights，避免快照里重复携带大对象

    var model = {
      config: cfg,
      weights: weights,
      get: function () { return JSON.parse(JSON.stringify({ config: cfg, weights: model.weights })); },
      setWeight: function (head, name, matrix) {
        if (name === 'heads') {
          if (model.weights.heads[head] && Array.isArray(matrix)) model.weights.heads[head] = clone(matrix);
        } else if (Array.isArray(model.weights[name])) {
          model.weights[name] = clone(matrix);
        }
      },
      setHeadWeight: function (head, key, matrix) {
        if (model.weights.heads[head] && ['Wq', 'Wk', 'Wv'].indexOf(key) >= 0) model.weights.heads[head][key] = clone(matrix);
      },
      resetWeights: function () { model.weights = initWeights(cfg, null); },
      forward: function () { return forward(cfg, model.weights); }
    };
    return model;
  }

  /* ---------------- 完整前向 + 快照 ---------------- */
  function forward(cfg, W) {
    var steps = [];
    var d = cfg.dModel, dk = cfg.dk, h = cfg.heads, dff = cfg.dff;
    var mode = cfg.mode;
    var srcTokens = cfg.tokens;
    var tgtTokens = mode === 'encoder' ? srcTokens : cfg.targetTokens;
    var vocabIndex = {};
    cfg.vocab.forEach(function (t, i) { vocabIndex[t] = i; });

    /* embedding 查表：E(token) × √d_model [论文] */
    function embed(tokens) {
      return tokens.map(function (t) {
        return W.E[vocabIndex[t] === undefined ? 0 : vocabIndex[t]].map(function (v) { return v * Math.sqrt(d); });
      });
    }
    function peFor(n) {
      if (cfg.pe === 'none') return zeros(n, d);
      if (cfg.pe === 'learned') return W.PE.slice(0, n).map(function (r) { return r.slice(0, d); });
      return sinusoidalPE(n, d);
    }

    rec(steps, 'tokens', '① 输入序列', 'tokenize → 位置索引',
      srcTokens.map(function (t, i) { return { index: i, token: t }; }),
      { kind: 'tokens', shape: srcTokens.length + ' 个 token', focus: ['input'],
        detail: mode === 'encoder' ? '源序列：每个 token 查一次词嵌入表。' :
          (mode === 'decoder' ? 'Decoder-only：输入为目标序列（右移一位，teacher forcing）。' :
            'Encoder–Decoder：源序列先编码，目标序列右移一位后送入解码器。'),
        source: '[论文]' });

    var X = embed(srcTokens);
    rec(steps, 'embedding', '② Token Embedding', 'E(token) × √d_model',
      X, { focus: ['input'], source: '[论文]',
        detail: '论文明确要求嵌入权重乘 √d_model。[解读] 常见解释是与正弦位置编码的量级匹配；论文只给出操作，未论证原因。' });

    var PE = peFor(srcTokens.length);
    rec(steps, 'pe', '③ 位置编码 PE', cfg.pe === 'sinusoidal' ?
      'PE(pos,2i)=sin(pos/10000^(2i/d))，PE(pos,2i+1)=cos(…) ' : (cfg.pe === 'learned' ? '可学习的位置查表（固定不更新）' : '无位置编码'),
      PE, { kind: cfg.pe === 'none' ? 'text' : 'matrix', focus: ['pe'], source: '[论文]',
        detail: cfg.pe === 'none' ? '关闭位置编码后，自注意力是置换不变的：打乱 token 顺序，输出作为集合不变。可到「位置编码消融」实验观察差异。' :
          (cfg.pe === 'learned' ? '[解读] 可学习位置编码不能外推到训练长度之外。' : '多频率指纹 + 相对位置可由线性变换表示 + 理论上可外推。') });

    var x0 = add(X, PE);
    rec(steps, 'input', '④  embeddings + PE', 'x₀ = Embedding × √d + PE',
      x0, { focus: ['input', 'pe'], source: '[实现]',
        detail: '加法之后再 dropout（本演示省略 dropout 以便观察确定性数值）。' });

    /* 编码器：为 cross 模式先算一份 Z（不展开每一步，避免重复刷屏） */
    var encoderZ = null;
    if (mode === 'cross') encoderZ = encodeForCross(srcTokens, embed, PE, W, cfg, x0);

    var input = mode === 'cross' ? embed(tgtTokens).map(function (r, i) {
      var peT = peFor(tgtTokens.length);
      return r.map(function (v, j) { return v + peT[i][j]; });
    }) : x0;
    var rows = input.length;
    if (mode !== 'encoder') {
      rec(steps, 'decoder-input', '④b Decoder 输入', 'tgt[:, :-1] = [<s>, y₁, y₂, …]',
        tgtTokens.map(function (t, i) { return { index: i, token: t }; }),
        { kind: 'tokens', shape: tgtTokens.length + ' 个 token', focus: ['input'],
          source: '[实现]', detail: '标签是 tgt[:, 1:] = [y₁, y₂, …, </s>]。右移 + 因果掩码共同保证自回归性：训练时可并行，推理时不能偷看未来。' });
    }

    /* 逐层 */
    var x = input;
    for (var layer = 0; layer < cfg.layers; layer++) {
      var suffix = cfg.layers > 1 ? '·第' + (layer + 1) + '层' : '';
      x = runLayer(x, layer, suffix);
    }

    /* 输出层：与 embedding 共享权重 [论文] */
    var logits = matmul(x, transpose(W.E));
    rec(steps, 'logits', '⑱ 输出 Logits', 'H · Eᵀ（与 embedding 共享权重）',
      logits, { focus: ['output'], source: '[论文]',
        detail: '论文要求两个 embedding 层与 pre-softmax 线性层共享权重。[实现差异] 哈佛注解默认不共享（代码写在 if False 块里）。' });
    var probs = softmaxRows(logits);
    rec(steps, 'probs', '⑲ 下一 token 概率', 'softmax(logits)，逐行归一化',
      probs, { kind: 'heat', focus: ['output'], source: '[论文]',
        detail: '行 i 表示「看到位置 i 及其之前的内容后，下一个 token 是词表中各词的概率」。概率最高的格子即模型的预测。',
        meta: { rowLabels: tgtTokens.map(function (t, i) { return '位置' + i + '「' + t + '」'; }), colLabels: cfg.vocab.slice() } });

    return {
      config: cfg, steps: steps, final: x, logits: logits, probs: probs,
      encoderZ: encoderZ, vocab: cfg.vocab, tokens: srcTokens, targetTokens: tgtTokens,
      d: d, dk: dk, heads: h, dff: dff
    };

    /* ---- 单层：多头注意力 + FFN + 残差 Norman ---- */
    function runLayer(Xin, li, suffix) {
      var maskSelf = mode === 'encoder' ? null : causalMatrix(Xin.length);
      var att = attentionBlock(Xin, Xin, maskSelf, 'self', 'self-attn', '自注意力', suffix, li);
      var mha = att.mha;
      var x1;
      if (cfg.norm === 'post') {
        x1 = layerNorm(add(Xin, mha));
        rec(steps, 'addnorm1' + li, '⑮ Add & Norm ①' + suffix, 'LN(x + MHA(x)) · Post-Norm',
          x1, { focus: ['norm'], source: '[论文]',
            detail: '残差要求 x 与子层输出维度相同 —— 这就是 d_model 贯穿全模型的原因。Post-Norm 是原论文写法。' });
      } else {
        var xNorm = layerNorm(Xin);
        x1 = add(Xin, mha); // mha 是在 xNorm 上算的，见下方 attentionBlock 的输入处理
        rec(steps, 'addnorm1' + li, '⑮ Add & Norm ①' + suffix, 'x + MHA(LN(x)) · Pre-Norm',
          x1, { focus: ['norm'], source: '[实现]',
            detail: '哈佛注解的 SublayerConnection 实际是 Pre-Norm（代码注释承认「for code simplicity the norm is first」）。现代大模型普遍采用。' });
      }

      /* cross 模式：解码器的第二子层 —— Q 来自 decoder，K/V 来自 encoder 输出 Z */
      if (mode === 'cross') {
        var cr = attentionBlock(x1, encoderZ, null, 'cross', 'cross-attn', '交叉注意力', suffix, li);
        x1 = cfg.norm === 'post' ? layerNorm(add(x1, cr.mha)) : add(x1, cr.mha);
        rec(steps, 'addnorm-cross' + li, 'Add & Norm（交叉注意力）' + suffix,
          cfg.norm === 'post' ? 'LN(x + CrossAttn(x, Z))' : 'x + CrossAttn(LN(x), Z)',
          x1, { focus: ['norm', 'cross'], source: '[论文]',
            detail: '交叉注意力是源序列信息进入解码器的唯一通道。最容易记反的一点：Q 来自 decoder，K/V 来自 encoder 输出 Z；注意力矩阵形状是 目标长 × 源长。' });
      }

      var h1 = x1.map(function (r) {
        return Array.from({ length: dff }, function (_, j) {
          var s = W.b1[j];
          for (var k = 0; k < d; k++) s += r[k] * W.W1[k][j];
          return activate(s, cfg.ffnAct);
        });
      });
      rec(steps, 'ffn1' + li, '⑯ FFN 中间层' + suffix,
        'H₁ = ' + (cfg.ffnAct === 'gelu' ? 'GELU' : 'ReLU') + '(xW₁ + b₁)',
        h1, { focus: ['ffn'], source: '[论文]',
          detail: 'FFN 逐位置作用，位置之间不交换信息；序列混合完全由注意力完成。[解读] 主流观点把它看作 key–value 记忆。' });

      var ffnOut = h1.map(function (r) {
        return Array.from({ length: d }, function (_, j) {
          var s = W.b2[j];
          for (var k = 0; k < dff; k++) s += r[k] * W.W2[k][j];
          return s;
        });
      });
      rec(steps, 'ffn2' + li, '⑰ FFN 输出' + suffix, 'FFN = H₁W₂ + b₂',
        ffnOut, { focus: ['ffn'], source: '[论文]',
          detail: '论文 d_ff = 4 × d_model；[解读] 这是经验配置，不是定理。' });

      var x2;
      if (cfg.norm === 'post') {
        x2 = layerNorm(add(x1, ffnOut));
      } else {
        x2 = add(x1, ffnOut);
      }
      rec(steps, 'addnorm2' + li, '⑱ Add & Norm ②' + suffix,
        cfg.norm === 'post' ? 'LN(x₁ + FFN(x₁))' : 'x₁ + FFN(LN(x₁))，栈末再补一次 LN',
        x2, { focus: ['norm'], source: cfg.norm === 'post' ? '[论文]' : '[实现]' });
      return cfg.norm === 'pre' && li === cfg.layers - 1 ? layerNorm(x2) : x2;
    }

    /* ---- 注意力块：多头拆分 → 每头打分 → 拼接 → Wᴼ ---- */
    function attentionBlock(X, KV, mask, tag, label, cn, suffix, li) {
      var src = cfg.norm === 'pre' ? layerNorm(X) : X;
      var kvSrc = (cfg.norm === 'pre' && KV === X) ? src : KV;
      var n = src.length, m = kvSrc.length;
      var headsData = [];

      rec(steps, tag + '-weights' + li, '⑥ ' + cn + '投影矩阵' + suffix,
        'Wᴽ, Wᴷ, Wⱽ ∈ ℝ^(d_model × d_k)',
        { 'W_Q': W.heads[0].Wq, 'W_K': W.heads[0].Wk, 'W_V': W.heads[0].Wv },
        { kind: 'triple', focus: ['attn'], source: '[论文]',
          detail: '当前展示第 1 个头（共 ' + h + ' 个）。在左侧「权重编辑器」里可直接改这些数字，观察注意力热力图如何变化。',
          meta: { head: 0 } });

      for (var head = 0; head < h; head++) {
        var Wq = W.heads[head].Wq, Wk = W.heads[head].Wk, Wv = W.heads[head].Wv;
        var Q = matmul(src, Wq), K = matmul(kvSrc, Wk), V = matmul(kvSrc, Wv);
        if (head === 0) {
          rec(steps, tag + '-qkv' + li, '⑦ Q / K / V' + suffix,
            'Q=XWᴽ, K=XWᴷ, V=XWⱽ → 每头 n × d_k',
            { 'Q': Q, 'K': K, 'V': V },
            { kind: 'triple', focus: ['attn'], source: '[论文]',
              detail: 'Query「我在找什么」；Key「我提供什么索引」；Value「我实际提供的内容」。软匹配：所有位置按权重贡献，不是硬选取。[解读]' });
        }
        var res = attention(Q, K, V, { dk: dk, scale: cfg.scale, temperature: cfg.temperature, mask: mask });
        headsData.push(res);
        if (head === 0) {
          var headName = '注意力分数 S = QKᵀ' + suffix;
          rec(steps, tag + '-scores' + li, '⑧ ' + headName, 'S[i][j] = qᵢ · kⱼ',
            res.raw, { kind: 'heat', focus: ['attn'], source: '[论文]',
              detail: '第 i 行第 j 列 = 位置 i 对位置 j 的原始匹配强度。',
              meta: { rowLabels: rowLabel(n, tag), colLabels: colLabel(m, tag), head: 0 } });

          rec(steps, tag + '-scale' + li, '⑨ 缩放 ' + suffix,
            cfg.scale ? 'S/√(d_k)' + (cfg.temperature !== 1 ? ' /T' : '') + '，d_k=' + dk : '跳过缩放（演示对照）',
            res.scaled, { kind: 'heat', focus: ['attn'], source: '[论文]',
              detail: cfg.scale ?
                '论文脚注：q、k 各分量均值 0 方差 1 时，内积方差为 d_k。不缩放会把 softmax 推入梯度极小的饱和区。[论文] 原文还指出：d_k 大时，不缩放的加性注意力反而优于点积注意力。' :
                '关闭缩放后，点积幅值随 d_k 增大，softmax 趋于 one-hot，梯度变小。可到「缩放与温度」实验观察。',
              meta: { rowLabels: rowLabel(n, tag), colLabels: colLabel(m, tag), head: 0 } });

          rec(steps, tag + '-mask' + li, '⑩ 掩码 ' + suffix,
            mode === 'encoder' || tag !== 'self' ? '无掩码（全部位置可见）' : 'causal mask：j ≤ i 才允许',
            res.masked, { kind: 'heat', focus: ['attn'], source: '[实现]',
              detail: mode === 'encoder' ? '编码器自注意力是双向的：每个位置能看到整个源序列，这是 BERT 类模型的结构基础。' :
                (tag === 'cross' ? '交叉注意力只用 padding mask，不加因果掩码 —— 源序列已经全部可知。' :
                  '上三角被置为 −1e9（论文的 −∞ 的数值近似），softmax 后权重近似为 0。注意掩码在 softmax 之前生效。'),
              meta: { rowLabels: rowLabel(n, tag), colLabels: colLabel(m, tag), head: 0 } });

          rec(steps, tag + '-attn' + li, '⑪ 注意力权重 A' + suffix,
            'A = softmax(S_masked)，每行和为 1',
            res.A, { kind: 'heat', focus: ['attn'], source: '[论文]',
              detail: '注意力权重是一次「软路由」，不等于因果重要性，不能直接当解释性证据。[解读] 悬停热力图可看具体数值。',
              meta: { rowLabels: rowLabel(n, tag), colLabels: colLabel(m, tag), head: 0 } });

          rec(steps, tag + '-context' + li, '⑫ 加权求和 AV' + suffix,
            'Context = A · V → n × d_v',
            res.O, { focus: ['attn'], source: '[论文]',
              detail: '每个位置的输出是所有位置 Value 的凸组合 —— 注意力的本质是内容相关的动态信息聚合。' });
        }
      }

      /* 拼接：view + transpose 的等价写法 */
      var concat = zeros(n, h * dk);
      for (var i = 0; i < n; i++) for (var head2 = 0; head2 < h; head2++) for (var k = 0; k < dk; k++) {
        concat[i][head2 * dk + k] = headsData[head2].O[i][k];
      }
      if (h > 1) {
        rec(steps, tag + '-concat' + li, '⑬ 多头拼接' + suffix,
          'Concat(head₁,…,head_h) → n × (h·d_k) = n × d_model',
          concat, { focus: ['attn'], source: '[实现]',
            detail: '张量形状链：(B,L,512) → (B,L,8,64) → (B,8,L,64) → 每头打分 → (B,L,8,64) → (B,L,512)。各头共享输入，只是投影不同；必须经 Wᴼ 融合，不是 h 个小模型。' });
      }
      var mha = matmul(concat, W.Wo);
      rec(steps, tag + '-wo' + li, '⑭ 输出投影 Wᴼ' + suffix,
        'MultiHead = Concat · Wᴼ → n × d_model',
        mha, { focus: ['attn'], source: '[论文]',
          detail: 'Wᴼ 不是多余的：它把 h 个子空间的结果融合回统一表示。去掉会破坏表达能力。' });
      return { mha: mha, heads: headsData };

      function rowLabel(len, tg) {
        if (tg === 'cross') return tgtTokens.slice(0, len).map(function (t, i) { return '目标' + i + '「' + t + '」'; });
        return (mode === 'encoder' ? srcTokens : tgtTokens).slice(0, len).map(function (t, i) { return '位置' + i + '「' + t + '」'; });
      }
      function colLabel(len, tg) {
        if (tg === 'cross') return srcTokens.slice(0, len).map(function (t, i) { return '源' + i + '「' + t + '」'; });
        return (mode === 'encoder' ? srcTokens : tgtTokens).slice(0, len).map(function (t, i) { return '位置' + i + '「' + t + '」'; });
      }
    }

    /* cross 模式的编码器：只算 Z，不重复记录所有步骤 */
    function encodeForCross(tokens, embedFn, peFn, Wt, cf, xIn) {
      var src = xIn;
      var n = src.length;
      var concat = zeros(n, h * dk);
      for (var head = 0; head < h; head++) {
        var Q = matmul(src, Wt.heads[head].Wq);
        var K = matmul(src, Wt.heads[head].Wk);
        var V = matmul(src, Wt.heads[head].Wv);
        var r = attention(Q, K, V, { dk: dk, scale: cf.scale, temperature: cf.temperature });
        for (var i = 0; i < n; i++) for (var k = 0; k < dk; k++) concat[i][head * dk + k] = r.O[i][k];
      }
      var mha = matmul(concat, Wt.Wo);
      var x1 = cf.norm === 'post' ? layerNorm(add(src, mha)) : add(src, mha);
      var h1 = x1.map(function (r) {
        return Array.from({ length: dff }, function (_, j) {
          var s = Wt.b1[j];
          for (var k = 0; k < d; k++) s += r[k] * Wt.W1[k][j];
          return activate(s, cf.ffnAct);
        });
      });
      var ffn = h1.map(function (r) {
        return Array.from({ length: d }, function (_, j) {
          var s = Wt.b2[j];
          for (var k = 0; k < dff; k++) s += r[k] * Wt.W2[k][j];
          return s;
        });
      });
      return cf.norm === 'post' ? layerNorm(add(x1, ffn)) : layerNorm(add(x1, ffn));
    }
  }

  /* ---------------- 预设权重（教学用，可直接点选） ---------------- */
  /* 单头 / d=dk 时最直观：把 Wq 设为单位阵，Wk 做成不同匹配规则 */
  var WEIGHT_PRESETS = {
    random: { label: 'Xavier 随机（默认）', desc: '标准初始化，数值稳定' },
    self: { label: '关注自己', desc: 'Q=I，K=I：对角线注意力最强' },
    previous: { label: '关注前一个 token', desc: 'K 向下移位：第 i 行主要看 i−1' },
    first: { label: '关注首个 token', desc: 'K 第一列全 1：所有位置都看第 0 个' },
    uniform: { label: '均匀关注', desc: 'K 全 1：每个位置平均看待全序列' }
  };
  function presetMatrix(name, d, dk) {
    var M = zeros(d, dk);
    for (var i = 0; i < d; i++) for (var j = 0; j < dk; j++) {
      if (name === 'self' && i === j) M[i][j] = 1;
      if (name === 'previous' && i === (j + 1) % d) M[i][j] = 1;
      if (name === 'first' && j === 0) M[i][j] = 1;
      if (name === 'uniform') M[i][j] = 1 / Math.sqrt(dk);
    }
    return M;
  }

  /* ---------------- 对外 API ---------------- */
  return {
    version: '1.0.0',
    DEFAULT_CONFIG: DEFAULT_CONFIG,
    WEIGHT_PRESETS: WEIGHT_PRESETS,
    create: create,
    forward: function (cfg) { return create(cfg).forward(); },
    attention: attention,
    presetMatrix: presetMatrix,
    matrices: {
      zeros: zeros, clone: clone, matmul: matmul, transpose: transpose, add: add, sub: sub,
      scale: scale, dot: dot, softmaxRow: softmaxRow, softmaxRows: softmaxRows,
      layerNorm: layerNorm, layerNormRow: layerNormRow, relu: relu, gelu: gelu,
      shape: shape, fmt: fmt, fmtMatrix: fmtMatrix
    },
    sinusoidalPE: sinusoidalPE,
    causalMatrix: causalMatrix,
    mulberry32: mulberry32
  };
});
