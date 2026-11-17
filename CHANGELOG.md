# 更新日志

## 1.0.0（2026-11-17）

首个公开版本。

- 零依赖前端实现 Transformer 完整前向：embedding×√d、三种位置编码、缩放点积注意力、因果掩码、多头注意力、FFN、Post/Pre-Norm、共享 embedding 输出层。
- 可编辑输入 token、结构超参、机制开关与每个头的 Q/K/V 权重；支持五种权重规律预设与 JSON 配置导入导出。
- 19–30 步完整计算快照，逐步展示矩阵、形状、公式、来源标注与解释；全景结构图可点击跳转。
- 四个对照实验：位置编码消融、因果掩码、多头分工、缩放与温度。
- Node 测试套件 77 项，CI 覆盖 Node 18/20/22。
- 文档：`docs/MATH.md`、`docs/EXPERIMENTS.md`，以及真实浏览器截图 6 张。

### 已知限制

- 教学玩具规模，不做训练、无 batch 维、无 dropout、无 KV Cache。
- LayerNorm 的 γ/β 固定为 1/0；`Wᴼ`、FFN 权重与 embedding 由随机种子决定，不可逐格编辑。
