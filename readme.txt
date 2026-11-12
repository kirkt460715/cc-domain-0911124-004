# 语料门

接收授权来源的文档，保存版本、标准化文本与审查决定，支持转载识别、去重簇和受权发布。文档撤回与来源撤权可以使发布失效。

除重复簇之外，还可以在既有去重结果上做主题探索实验：把每个重复簇当作一个计权单位，冻结词表与 TF–IDF 变换，用余弦距离的球面聚类比较若干候选主题数，并借助轮廓系数和多次固定种子重采样的成员稳定性来选择，而不是只挑训练目标最小的方案。每个主题输出区分词、代表文档、来源构成和边界文档；短文本、空词表、唯一有效簇和未见词都有明确的去向。编辑可以保存“必须同主题”或“必须分开”的约束，约束矛盾不会被处理顺序吞掉；某个候选主题数无法满足约束时会明确标为不可行。再次分组可对照旧主题的拆分、合并与成员迁移，主题编号变化不会被误报为内容变化。新文档使用冻结变换归入主题或进入待归类区，不会重新拟合旧实验。授权撤回后仍执行既有导出限制，并可由 `topic.list`、`system.source_impact` 指出失效的实验。

演练包含重复文本、审查、发布校验，以及转载去权前后的主题差别和约束引起的真实主题调整。相似关系的连通分量不代表语义等价；联系信息检测仅供复核。

主要代码位于 corpus_gate，按 ingest、review、text、dedup、topics、releases 划分职责；tests 保存领域行为、边界及存储回归用例。命令接收严格 JSON，详细操作名与字段可通过 `operations` 及各帮助入口查询。主题探索相关操作包括 `topic.run`、`topic.get`、`topic.list`、`topic.participants`、`topic.compare`、`topic.classify`，以及 `topic_constraint.put`、`topic_constraint.remove`、`topic_constraint.list`、`topic_constraint.history`。示例中的机构、客户、案件和测量均为本地合成记录，不涉及真实个人资料。

使用 Python 3.11.2 与标准库，无第三方包。运行无需访问外部服务或提供账号。依赖清单 requirements.txt 为空，.python-version 固定解释器版本。

在项目根目录运行：

```sh
python3 -m unittest discover -s tests -v
python3 -m compileall -q corpus_gate
python3 -m corpus_gate --db :memory: demo --fixture fixtures/demo.json
```

也可执行下面的统一检查，按顺序进行语法编译、领域回归和完整本地演练，任一环节失败即退出。运行写入均在临时目录或内存数据库，重复执行不需清理已有业务账。正常数据规模的检查可在 90 秒内完成。

```sh
sh scripts/check.sh
```
