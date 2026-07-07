# CropGenome-FM 训练进展与评估

更新时间: 2026-07-07 17:45 CST

本文件是 GitHub 上的训练进展主入口。结论优先，只保留当前判断所需的训练曲线、2080Ti 下游评估和下一步计划；旧的逐 checkpoint（模型存档点）细节放在 `docs/training_progress/` 下作为可复核源数据。

## 1. 当前一句话结论

`CropGenome-FM-v2-Stable-8K`（作物基因组基础模型第二版稳健 8192 碱基版）已训练到 step17000 并触发 early stopping（早停）。训练阶段应先冻结，不继续盲目加训或扩大 checkpoint 扫描。

2080Ti 上完成的 fixed formal-lite benchmark（固定轻量正式化测试，Stage_B proxy 标签）显示：step17000 的模型向量平均 F1 = 0.7439，高于 step14000 的 0.6961，阶段主候选选 step17000；但 step14000 在 TES_polyA 和 promoter_TSS 上更好，因此正式 GFF-derived benchmark（由 GFF 精确构建的正式基准）前仍保留 step14000 作为 sensitivity/backup（敏感性/备选）候选。

## 2. 当前状态表

| 项目 | 当前值 | 解释 |
|---|---:|---|
| 主模型 | `CropGenome-FM-v2-Stable-8K` | 8K context（8192 碱基上下文）作物基因组预训练模型。 |
| 训练状态 | step17000 early stop | 训练已到冻结点；当前不建议继续加训。 |
| 最新 train loss（训练损失） | 1.0611 | 来自 step17000。 |
| 最新 val loss（验证总损失） | 1.1410 | 来自 step17000。 |
| 最新 val selection loss（验证选择损失） | 1.0729 | `MLM loss + 0.02 × RC loss`，TSV 当前最低。 |
| `checkpoint_best.pt` 冲突 | 指向 step14000 | 训练日志 best 文件名与 step17000 的 validation/probe 信号不完全一致，所以用下游评估解冲突。 |
| full-region diagnostic probe（完整区域诊断探针） | step17000 最强 | embedding Macro-F1（向量类别平均 F1）0.3030；region head Macro-F1（区域头类别平均 F1）0.3213。 |
| formal-lite test（轻量正式化测试） | step17000 平均更强 | mean model-embedding F1: step14000=0.6961，step17000=0.7439。 |
| GitHub 上传策略 | 只传轻量产物 | TSV/JSON/PNG；不上传 checkpoint、训练日志、逐样本表、二进制输入缓存、PDF/SVG 或原始大数据。 |

## 3. 训练曲线

### 3.1 Total loss（总损失）

![v2 total loss](docs/training_progress/figures/v2_stable_stageB_loss.png)

- 图: [docs/training_progress/figures/v2_stable_stageB_loss.png](docs/training_progress/figures/v2_stable_stageB_loss.png)
- 源数据: [docs/training_progress/source_data/v2_stable_stageB_metrics.tsv](docs/training_progress/source_data/v2_stable_stageB_metrics.tsv)

### 3.2 Selection loss（选择损失）

![v2 selection loss](docs/training_progress/figures/v2_stable_stageB_selection_loss.png)

- 图: [docs/training_progress/figures/v2_stable_stageB_selection_loss.png](docs/training_progress/figures/v2_stable_stageB_selection_loss.png)
- 曲线摘要: [docs/training_progress/source_data/v2_stable_stageB_curve_summary.tsv](docs/training_progress/source_data/v2_stable_stageB_curve_summary.tsv)

解释: selection loss（选择损失）是主模型选择指标，定义为 `MLM loss + 0.02 × RC loss`（遮盖碱基预测损失 + 小权重反向互补一致性损失）。region loss/acc（区域辅助损失/准确率）只作辅助健康检查。

## 4. 关键证据链

| 证据 | 结果 | 怎么看 |
|---|---:|---|
| validation selection loss（验证选择损失） | step10000: 1.0897 → step14000: 1.0743 → step17000: 1.0729 | 预训练验证指标持续改善，支持训练有效并可冻结。 |
| full-region diagnostic embedding F1（完整区域诊断向量 F1） | step17000: 0.3030 | 诊断 probe 中 step17000 当前最高，但不是论文正式 benchmark。 |
| full-region diagnostic region-head F1（完整区域诊断区域头 F1） | step17000: 0.3213 | 区域辅助头也在 step17000 达到当前最高。 |
| medium validation-only promoter_TSS | step14000: 0.7592，step17000: 0.7259 | promoter（启动子）任务支持保留 step14000。 |
| medium validation-only splice_acceptor | step14000: 0.5197，step17000: 0.6825 | splice acceptor（剪接受体）任务明显支持 step17000。 |
| formal-lite test mean F1 | step14000: 0.6961，step17000: 0.7439 | 固定 train/test proxy 测试下，step17000 是阶段主候选。 |

## 5. 2080Ti formal-lite benchmark: step14000 vs step17000

这是本次为解决“只靠 2080Ti 能否先形成阶段性正式结果”而做的固定评测。它使用 Stage_B 的独立 `train/test` split（训练/测试划分）和 proxy labels（代理标签），比之前 validation-only 更接近正式流程；但它仍不是最终 GFF-derived paper benchmark（由 GFF 精确标签构建的论文正式基准）。

![formal-lite comparison](docs/training_progress/cropgenome_bench_v1_formal_lite_test/step14000_vs_step17000_embedding.png)

| 任务 | step14000 F1 | step17000 F1 | 赢家 | 解释 |
|---|---:|---:|---|---|
| TES_polyA（转录终止/多聚腺苷酸化） | 0.8436 | 0.8385 | step14000 | 两者很接近，step14000 略高。 |
| promoter_TSS（启动子/转录起始位点） | 0.7525 | 0.7266 | step14000 | promoter proxy 仍偏向 step14000。 |
| splice_acceptor（剪接受体） | 0.4921 | 0.6667 | step17000 | step17000 大幅修复 splice_acceptor，是平均分提升的主要来源。 |
| mean F1（3 任务平均） | 0.6961 | 0.7439 | step17000 | 阶段主候选选 step17000。 |

文件入口:

- formal-lite 说明: [docs/training_progress/cropgenome_bench_v1_formal_lite_test/README.md](docs/training_progress/cropgenome_bench_v1_formal_lite_test/README.md)
- 汇总结论 JSON: [docs/training_progress/cropgenome_bench_v1_formal_lite_test/formal_lite_summary.json](docs/training_progress/cropgenome_bench_v1_formal_lite_test/formal_lite_summary.json)
- 对比表 TSV: [docs/training_progress/cropgenome_bench_v1_formal_lite_test/step14000_vs_step17000_embedding.tsv](docs/training_progress/cropgenome_bench_v1_formal_lite_test/step14000_vs_step17000_embedding.tsv)
- 全部指标长表: [docs/training_progress/cropgenome_bench_v1_formal_lite_test/summary_metrics_long.tsv](docs/training_progress/cropgenome_bench_v1_formal_lite_test/summary_metrics_long.tsv)
- 每个方法平均表: [docs/training_progress/cropgenome_bench_v1_formal_lite_test/summary_method_means.tsv](docs/training_progress/cropgenome_bench_v1_formal_lite_test/summary_method_means.tsv)

## 6. 历史评估入口

| 目录 | 用途 | 结论边界 |
|---|---|---|
| [docs/training_progress/downstream_evaluations_2080/](docs/training_progress/downstream_evaluations_2080/) | step1000–17000 full-region diagnostic probe（完整区域诊断探针） | 诊断 checkpoint 趋势，不进论文主表。 |
| [docs/training_progress/cropgenome_bench_v1_medium_validation/](docs/training_progress/cropgenome_bench_v1_medium_validation/) | medium validation-only benchmark（中等规模验证集基准） | 只用于 checkpoint 选择，不当正式 test。 |
| [docs/training_progress/cropgenome_bench_v1_pilot_smoke/](docs/training_progress/cropgenome_bench_v1_pilot_smoke/) | pilot smoke benchmark（小规模冒烟测试） | 只证明流程跑通。 |

## 7. 当前决策

1. 训练停止/冻结在 step17000，不继续盲目加训。
2. 阶段主 checkpoint（模型存档点）采用 step17000。
3. step14000 作为 sensitivity/backup（敏感性/备选）候选保留，因为 promoter_TSS 和 TES_polyA 更强。
4. 不再给所有 checkpoint 跑 formal test（正式测试），避免 test-set leakage（测试集泄漏）。
5. 后续论文主结论必须来自 GFF-derived hard negatives（由 GFF 精确构建的硬负样本）、固定 split（固定划分）、多 seed（多随机种子）和外部模型同口径 baseline（基线模型）。

## 8. 下一步计划

| 优先级 | 任务 | 目的 | 完成标准 |
|---|---|---|---|
| P0 | 构建 GFF-derived CropGenome-Bench v1 | 把 proxy benchmark 升级为正式论文 benchmark | splice/promoter/TES 等任务有固定 train/valid/test、hard negatives 和脱敏 manifest。 |
| P0 | 只评 step14000 与 step17000 | 避免扫描所有 checkpoint 造成测试集泄漏 | 两个候选在同一 split、同一头、同一 metrics 下比较。 |
| P0 | 外部 baseline（基线模型） | 证明作物模型相对通用 DNA 模型的优势 | 至少 1-mer/CNN/可运行公开模型 embedding 同口径结果。 |
| P0 | 多 seed 稳定性 | 区分单 seed 最高点和稳定提升 | 5 seeds mean ± std（平均值±标准差）或先给 3 seeds 阶段版。 |
| P1 | 跨物种/少样本 | 支撑“作物基因组预训练模型”的核心定位 | species/genus holdout（物种/属留出）或 few-shot（少样本）显示收益。 |

当前最重要的停止条件: 在 GFF-derived benchmark 没构建好之前，不再把 proxy 结果写成论文正式胜利；但可以把本文件当前结果作为“2080Ti 已完成阶段性 checkpoint 决策”的 GitHub 进展记录。
