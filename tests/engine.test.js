/* Transformer 实验室 · 数值引擎测试（纯 Node，无浏览器依赖）
 * 覆盖：softmax / 位置编码 / 掩码 / 形状不变量 / 可编辑权重 / 序列化
 * 运行：node tests/run-all.js
 */
const E = require('../js/engine.js');
const M = E.matrices;

exports.run = function (t) {
  t.group('引擎 · 矩阵工具');

  const S = M.softmaxRows([[1, 2, 3], [-1000, 0, 1000], [5, 5, 5]]);
  S.forEach((row, i) => t.near(row.reduce((a, b) => a + b, 0), 1, 1e-12, 'softmax 第' + i + '行和为 1'));
  t.ok(S[1][0] < 1e-12 && S[1][2] > 0.99, 'softmax 在极端值下稳定且单调');
  t.ok(Math.abs(S[2][0] - 1 / 3) < 1e-12, 'softmax 对等值输入均匀分布');

  t.eq(M.shape(M.matmul([[1, 2], [3, 4]], [[1, 0], [0, 1]])), '2 × 2', '矩阵乘法维度正确');
  t.near(M.matmul([[1, 2], [3, 4]], [[1, 0], [0, 1]])[1][0], 3, 1e-12, '矩阵乘法数值正确');

  t.group('引擎 · 位置编码与掩码');

  const PE = E.sinusoidalPE(5, 8);
  t.eq(M.shape(PE), '5 × 8', '正弦位置编码形状 n × d_model');
  t.near(PE[0][0], 0, 1e-12, 'PE(0,0) = sin(0) = 0');
  t.near(PE[0][1], 1, 1e-12, 'PE(0,1) = cos(0) = 1');
  t.ok(PE[1][0] !== PE[0][0], '不同位置编码不同');
  const mask = E.causalMatrix(3);
  t.ok(mask[0][0] && !mask[0][1] && !mask[0][2], '因果掩码第 0 行只含自己');
  t.ok(mask[2][0] && mask[2][1] && mask[2][2], '因果掩码最后一行可见全部前缀');

  const q = [[1, 0], [1, 0]], k = [[1, 0], [0, 1]], v = [[1, 2], [3, 4]];
  const att = E.attention(q, k, v, { dk: 2, scale: true, mask: E.causalMatrix(2) });
  t.near(att.A[0][1], 0, 1e-8, '掩码位置注意力权重近似为 0');
  t.near(att.A[0][0], 1, 1e-8, '因果第 0 行只关注自己');
  t.near(att.A[1][0] + att.A[1][1], 1, 1e-12, '注意力每行和为 1');
  t.near(att.divisor, Math.sqrt(2), 1e-12, '缩放因子为 √d_k');

  t.group('引擎 · 前向步骤与形状不变量');

  const enc = E.forward({ seed: 7 });
  t.eq(enc.steps.length, 19, 'encoder 默认 19 个计算步骤');
  t.eq(enc.steps[enc.steps.length - 1].id, 'probs', '最后一步是输出概率');
  t.eq(M.shape(enc.final), '3 × 8', '编码器输出形状 n × d_model（残差约束）');
  t.eq(M.shape(enc.steps.find(s => s.id === 'self-scores0').value), '3 × 3', '自注意力分数 n × n');
  t.eq(M.shape(enc.steps.find(s => s.id === 'ffn10').value), '3 × 16', 'FFN 中间层 n × d_ff');
  enc.steps.forEach(s => t.ok(s.no >= 1 && s.title && s.formula, '步骤 ' + s.no + ' 含序号/标题/公式'));

  const dec = E.forward({ mode: 'decoder', seed: 7 });
  t.eq(dec.steps.length, 20, 'decoder 比 encoder 多一步右移说明');
  t.ok(dec.steps.find(s => s.id === 'decoder-input'), 'decoder 记录右移输入步骤');
  const maskedRow = dec.steps.find(s => s.id === 'self-mask0').value[0];
  t.ok(maskedRow.slice(1).every(v => v < -1e8), 'decoder 掩码后未来位置为 -1e9');

  const cross = E.forward({ mode: 'cross', tokens: ['a', 'b', 'c', 'd'], targetTokens: ['<s>', 'x'], seed: 7 });
  t.eq(M.shape(cross.steps.find(s => s.id === 'cross-scores0').value), '2 × 4', '交叉注意力形状 目标长 × 源长');
  t.eq(cross.steps.filter(s => s.id.indexOf('cross-') === 0).length > 0, true, 'cross 模式记录交叉注意力步骤');

  t.group('引擎 · 开关与可编辑权重');

  const plain = E.forward({ scale: false, seed: 7 });
  const raw = plain.steps.find(s => s.id === 'self-scores0').value;
  const scaled = plain.steps.find(s => s.id === 'self-scale0').value;
  t.ok(raw.every((r, i) => r.every((v, j) => Math.abs(v - scaled[i][j]) < 1e-12)), '关闭缩放后 S 保持不变');

  const hot = E.forward({ scale: true, temperature: 0.5, seed: 7 }).steps.find(s => s.id === 'self-scale0').value;
  const base = enc.steps.find(s => s.id === 'self-scale0').value;
  t.ok(Math.abs(hot[0][0]) > Math.abs(base[0][0]) || Math.abs(hot[0][2]) > Math.abs(base[0][2]),
    '降低温度 T 放大分数幅值');

  const m = E.create({ dModel: 4, heads: 1, seed: 3 });
  m.setHeadWeight(0, 'Wq', M.zeros(4, 4));
  const zeroed = m.forward().steps.find(s => s.id === 'self-attn0').value;
  t.ok(zeroed.every(row => row.every(v => Math.abs(v - 1 / 3) < 1e-12)), 'Q 全 0 时注意力退化为均匀分布');

  const first = E.presetMatrix('first', 4, 4);
  t.ok(first.every(row => row[0] === 1 && row.slice(1).every(v => v === 0)), '预设「关注首个 token」构造正确');
  const selfMat = E.presetMatrix('self', 4, 4);
  t.ok(selfMat.every((row, i) => row.every((v, j) => v === (i === j ? 1 : 0))), '预设「关注自己」是单位阵');

  const pre = E.forward({ norm: 'pre', seed: 7 });
  t.eq(M.shape(pre.final), '3 × 8', 'Pre-Norm 同样满足维度不变量');
  const two = E.forward({ layers: 2, seed: 7 });
  t.ok(two.steps.length > enc.steps.length, '堆叠 2 层后步骤数增加');
  const multi = E.forward({ dModel: 6, heads: 4, seed: 7 });
  t.eq(multi.heads, 3, 'd_model=6 且请求 4 头时自动降为能整除的最大头数 3');

  const snapshot = m.get();
  const clone = E.create(snapshot.config);
  clone.weights = JSON.parse(JSON.stringify(snapshot.weights));
  t.ok(JSON.stringify(clone.forward().final) === JSON.stringify(m.forward().final), '配置与权重可序列化并复现相同结果');
};
