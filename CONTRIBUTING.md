# 贡献指南

## 运行环境

页面本身零依赖：直接用浏览器打开 `index.html` 即可。开发与测试需要 Node.js 18+。

```bash
npm ci
npm test
```

测试分两部分：`engine.test.js` 校验数值、形状与可复现性；`dom.test.js` 用 jsdom 校验界面联动、播放器和四个实验。新增功能时请同步补测试。

## 修改引擎的约定

`js/engine.js` 每一步计算都会写入 `steps` 快照。新增或修改步骤时请遵守：

1. 每个步骤包含 `no / title / formula / value / shape / source / detail / focus`，其中 `source` 取 `[论文]`、`[实现]`、`[解读]` 三者之一。
2. 凡是论文与常见实现不一致的地方（Post/Pre-Norm、权重共享、缩放理由、掩码数值），不要把二手博客的结论写成事实；依据见 `../docs/TRANSFORMER.md` 的来源分级。
3. 所有张量用二维数组，矩阵乘法维度必须与 `docs/MATH.md` 的表格一致；改动后先跑形状不变量测试。
4. 权重对象通过 `model.weights` 访问，不要新增闭包内私有引用——外部替换权重后 `forward()` 必须立即用上新对象。

## 添加实验

`js/lab.js` 中每个实验是一个返回 HTML 字符串的函数。原则：**固定同一组权重，只改变被研究的变量**。在 `EXPERIMENTS` 数组注册后，在 `run()` 的分发表中加入对应函数，并更新 `docs/EXPERIMENTS.md`。

## 提交

- 一次提交只做一件事；提交信息用祈使句，例如 `Add causal mask experiment`。
- 不要提交 `node_modules/`、临时截图脚本或个人配置。
- 界面的表述保持正式、克制：用真实数字，少用营销词，主动写出限制。
