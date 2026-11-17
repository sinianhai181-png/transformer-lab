/* 统一测试入口：node tests/run-all.js */
const suites = [require('./engine.test.js'), require('./dom.test.js')];

let pass = 0, fail = 0, group = '';
const t = {
  group: function (g) { group = g; console.log('\n  ' + g); },
  ok: function (cond, msg) { record(!!cond, msg); },
  eq: function (actual, expected, msg) {
    record(actual === expected, msg + '  [实际=' + JSON.stringify(actual) + '，期望=' + JSON.stringify(expected) + ']');
  },
  near: function (actual, expected, eps, msg) {
    const ok = Number.isFinite(actual) && Math.abs(actual - expected) <= eps;
    record(ok, msg + '  [实际=' + actual + '，期望=' + expected + '±' + eps + ']');
  }
};
function record(ok, msg) {
  if (ok) { pass++; console.log('    PASS  ' + msg); }
  else { fail++; console.log('    FAIL  ' + group + ' → ' + msg); }
}

console.log('Transformer 可视化实验室 · 测试');
console.log('================================');
suites.forEach(s => s.run(t));
console.log('\n================================');
console.log('共 ' + (pass + fail) + ' 项：' + pass + ' 通过，' + fail + ' 失败');
process.exit(fail ? 1 : 0);
