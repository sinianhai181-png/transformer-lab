/* Transformer 实验室 · 界面交互测试（jsdom）
 * 覆盖：初始化 / 参数联动 / 播放器 / 权重编辑 / 全景跳转 / 四个实验
 * 运行：node tests/run-all.js
 */
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');
const ROOT = path.join(__dirname, '..');

function boot() {
  const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
  const dom = new JSDOM(html, { runScripts: 'dangerously', url: 'file:///transformer-lab/index.html' });
  const w = dom.window;
  ['js/engine.js', 'js/visualizer.js', 'js/lab.js', 'js/app.js'].forEach(function (f) {
    w.eval(fs.readFileSync(path.join(ROOT, f), 'utf8'));
  });
  w.document.dispatchEvent(new w.Event('DOMContentLoaded'));
  return w;
}
function fire(w, el, type) { el.dispatchEvent(new w.Event(type, { bubbles: true })); }

exports.run = function (t) {
  t.group('界面 · 初始化');

  const w = boot();
  const doc = w.document;
  t.ok(!!w.TMApp && !!w.TMLab && !!w.TMVis, '三个全局模块均已加载');
  const s0 = w.TMApp.state();
  t.ok(s0.config && s0.result && s0.steps.length >= 19, '首次加载完成前向计算');
  t.ok(doc.getElementById('stepBody').innerHTML.length > 0, '第一步数值面板已渲染');
  t.ok(doc.querySelectorAll('#stepChips .step-chip').length === s0.steps.length, '步骤导航数量与快照一致');
  t.ok(doc.querySelectorAll('#panorama .pano-node').length >= 5, '全景结构图节点已渲染');

  t.group('界面 · 参数联动');

  const src = doc.getElementById('srcTokens');
  src.value = 'a b c d';
  fire(w, src, 'change');
  const s1 = w.TMApp.state();
  t.eq(s1.config.tokens.length, 4, '修改 token 序列后配置生效');
  t.eq(s1.result.steps.find(x => x.id === 'tokens').value.length, 4, '快照同步更新为 4 个 token');
  t.ok(doc.getElementById('shapeSummary').textContent.indexOf('n=4') >= 0, '结构摘要同步显示 n=4');

  const dModel = doc.getElementById('dModel');
  dModel.value = '16'; fire(w, dModel, 'change');
  const heads = doc.getElementById('heads');
  heads.value = '4'; fire(w, heads, 'change');
  t.eq(w.TMApp.state().config.dk, 4, 'd_model=16, h=4 时 d_k=4');
  t.eq(doc.querySelectorAll('#headTabs .head-tab').length, 4, '权重编辑器的头标签同步为 4 个');

  const srcBack = doc.getElementById('srcTokens');
  srcBack.value = 'I love transformer'; fire(w, srcBack, 'change');

  t.group('界面 · 播放器与全景跳转');

  w.TMApp.setStep(3);
  t.ok(doc.getElementById('stepCounter').textContent.indexOf('第 4 /') >= 0, 'setStep 正确定位到第 4 步');
  const next = doc.getElementById('btnNext');
  fire(w, next, 'click');
  t.eq(w.TMApp.state().stepIndex, 4, '「下一步」按钮推进步骤');
  const play = doc.getElementById('btnPlay');
  fire(w, play, 'click');
  t.ok(play.textContent.indexOf('暂停') >= 0, '自动播放按钮切换到暂停态');
  fire(w, play, 'click');
  t.ok(play.textContent.indexOf('播放') >= 0, '再次点击恢复播放态');

  const node = doc.querySelector('#panorama .pano-node[data-focus="ffn"]');
  fire(w, node, 'click');
  const jumped = w.TMApp.state().stepIndex;
  t.ok(w.TMApp.state().steps[jumped].focus.indexOf('ffn') >= 0, '点击全景 FFN 节点跳转到相应步骤');

  t.group('界面 · 权重编辑');

  const preset = doc.getElementById('presetSelect');
  preset.value = 'first'; fire(w, preset, 'change');
  fire(w, doc.getElementById('btnFillPreset'), 'click');
  fire(w, doc.getElementById('btnApplyWeights'), 'click');
  const Wk = w.TMApp.state().weights.heads[0].Wk;
  t.ok(Wk.every(row => row[0] === 1), '预设权重填入并应用：W_K 第一列全 1');
  t.ok(doc.getElementById('stepBody').innerHTML.length > 0, '重算后数值面板仍正常渲染');

  t.group('界面 · 四个实验');

  const tabs = doc.querySelectorAll('#labTabs .lab-tab');
  t.eq(tabs.length, 4, '实验台有四个实验标签');
  ['pe', 'mask', 'heads', 'scale'].forEach(function (id) {
    const b = doc.querySelector('#labTabs .lab-tab[data-id="' + id + '"]');
    fire(w, b, 'click');
    const html = doc.getElementById('labBody').innerHTML;
    t.ok(html.length > 100 && html.indexOf('实验运行失败') < 0, '实验 ' + id + ' 正常渲染');
  });

  w.TMApp.pause();
  w.close();
};
