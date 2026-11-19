# 独立补充提示

本文件记录工作过程批注、口径规则说明、采集局限与操作指引，不属于事实素材，也不属于写作建议。事实见 [PAPERS.md](PAPERS.md)，观点见 [WRITING-MEMO.md](WRITING-MEMO.md)。

## 一、采集口径规则（本次执行）

1. 先确认存在性与发布主体：论文标题、作者、日期一律从 arXiv 官方摘要页或期刊官方页读取，不从搜索结果摘要抄数字。
2. arXiv 编号先搜索再核对，不凭记忆填写。本次过程中发现两处记忆编号错误并已纠正：TranAD 曾误记为 arXiv:2202.02906（实际为另一篇关于 parametric coupling flows 的论文），正确编号 arXiv:2201.07284；PatchTST 曾误记为 arXiv:2303.11156（实际为 AI 生成文本检测论文），正确编号 arXiv:2211.14730。两个错误编号均未进入事实文档。
3. 会议信息若 arXiv 页面未标注，改查会议官方页面；仍查不到写"公开暂未披露"，并附公开核验路径。
4. 同一事实多源不一致时并列登记，不取舍、不平均、不改写。本次并列了四处：原论文两版 BLEU、Anomaly Transformer 基准数量、评估协议争议、注意力解释性两派。
5. 百分比、倍数、点数逐项写明口径：区分准确率、Top-1 准确率、运行时间减少比例、算力比例、训练加速倍数、困惑度绝对值、准确率百分点。同一篇论文里不同指标不合并描述。
6. 仿真数据与实测数据分区：篇 15 是 Net3 水力模型仿真样本，篇 16 是实验室尺度管道传感器数据，篇 17 是物理实验台 962 组数据，三者准确率不可横向比较。已在数据汇总表逐条注明。
7. 风险维度必采：评测协议缺陷、注意力解释争议、漏水检测领域"检测已近解决、定位仍难"的反例，均已纳入。

## 二、采集局限（如实登记）

- OpenReview 的 Anomaly Transformer 页面访问时触发浏览器验证，未能读取，因此 Spotlight 等级只能依据作者官方 GitHub 自述并标为二级信源。公开核验路径：https://openreview.net/forum?id=LzQQ89U1qm_
- ScienceDirect 三篇论文页面未展示期刊名、卷、文章号与 DOI 的完整字段，需登录或机构权限查看 "Article info"。本轮未尝试绕过任何访问限制，字段统一填"公开暂未披露"。公开核验路径已写入事实文档对应条目。
- 篇 3 的 PA%K 具体定义来自第三方论文解读页，未采信为事实；只登记了 arXiv 摘要明确写出的"new baseline and an evaluation protocol"。核验路径：arXiv 全文第 4 节。
- 供水/管道漏水方向的中文综述（网易转载、百度百科等）均为三级或无明确发布主体，未进入事实文档。
- 本轮只做资料采集与项目文档补充，没有改动计算引擎，也没有新增实验台功能。页面上"注意力≠解释"等既有主题的新论文依据需要后续另行接入。

## 三、用户操作指引（供自行复核）

- 核对 arXiv 论文：打开 https://arxiv.org/abs/<编号>，查看 Title / Authors / Submission history / Abstract 四个字段。
- 核对会议接收：ICLR 用 https://iclr.cc/virtual/<年份>/poster/<编号>；NeurIPS 用 https://papers.nips.cc/paper_files/paper/<年份>；AAAI 用会议 virtualchair poster 页。
- 核对期刊论文： ScienceDirect 文章页 → Article info 栏目查看期刊、卷期、日期、DOI；摘要与 Highlights 区域可免费读取。
- 复核原论文 BLEU 差异：分别打开 arXiv v7 摘要页与 NeurIPS 2017 proceedings 摘要页，对照 Abstract 段落。

## 四、文件分区说明

- `docs/PAPERS.md`：只放可追溯事实、原文表述、来源主体、发布时间、链接、数据口径。无建议、无过程批注。
- `docs/WRITING-MEMO.md`：写作建议与分析推论，开头已标注"非事实素材，不可作为数据引用"。
- `docs/RESEARCH-NOTES.md`（本文件）：过程批注、口径规则、局限与操作指引。
- 三份文件互不混写；此前五份附件中的网页残留文字与 AI 问答拼接内容未被当作论文来源。
