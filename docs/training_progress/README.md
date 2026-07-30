# Training progress artifacts

主入口请看仓库根目录的 [TRAINING_PROGRESS.md](../../TRAINING_PROGRESS.md)。本目录只保存可复核的轻量源数据和图片，不保存 checkpoint（模型存档点）、训练日志、逐样本表、二进制输入缓存、PDF/SVG 或原始大数据。

## 当前应优先查看

| 目录/文件 | 内容 | 怎么看 |
|---|---|---|
| [cropgenome_bench_v1_formal_a100/](cropgenome_bench_v1_formal_a100/) | GFF-derived 正式 benchmark 与 Stage C1 64K gate | 当前论文评测主结果和四项 64K gate 证据。 |
| [STAGE_C1_64K_TRAINING_LOGIC_AUDIT_20260710.md](STAGE_C1_64K_TRAINING_LOGIC_AUDIT_20260710.md) | Stage C1 脚本/训练逻辑审计与修复闭环 | 看原运行为何作废、如何修复、怎样验证和重启。 |
| [cropgenome_bench_v1_formal_lite_test/](cropgenome_bench_v1_formal_lite_test/) | 2080Ti fixed formal-lite benchmark（固定轻量正式化测试） | 历史阶段决策证据；当前唯一 8K 最终版已统一为 early-stop step14000。 |
| [figures/v2_stable_stageB_loss.png](figures/v2_stable_stageB_loss.png) | v2 Stable total loss（总损失）曲线 | 看训练是否稳定下降。 |
| [figures/v2_stable_stageB_selection_loss.png](figures/v2_stable_stageB_selection_loss.png) | selection loss（选择损失）曲线 | 看 checkpoint 选择指标。 |
| [source_data/v2_stable_stageB_metrics.tsv](source_data/v2_stable_stageB_metrics.tsv) | 训练/验证指标原始轻量表 | 训练曲线源数据。 |
| [figures/v2_stageB_continuation_no_replacement_loss.png](figures/v2_stageB_continuation_no_replacement_loss.png) | 三GPU无放回续训total loss | 灰色原始点、蓝色滚动中位数、红色validation和紫色最新点。 |
| [figures/v2_stageB_continuation_no_replacement_selection_loss.png](figures/v2_stageB_continuation_no_replacement_selection_loss.png) | 三GPU无放回续训selection loss | 当前续训checkpoint选择指标趋势。 |
| [source_data/v2_stageB_continuation_no_replacement_metrics.tsv](source_data/v2_stageB_continuation_no_replacement_metrics.tsv) | 当前续训训练/验证指标 | 新训练代际独立源数据，不覆盖历史曲线。 |

## 历史/辅助结果

| 目录 | 内容 | 边界 |
|---|---|---|
| [downstream_evaluations_2080/](downstream_evaluations_2080/) | step1000–17000 full-region diagnostic probe（完整区域诊断探针） | 只用于观察 checkpoint 趋势。 |
| [cropgenome_bench_v1_medium_validation/](cropgenome_bench_v1_medium_validation/) | medium validation-only benchmark（中等规模验证集基准） | 只用于候选选择，不是正式 test。 |
| [cropgenome_bench_v1_pilot_smoke/](cropgenome_bench_v1_pilot_smoke/) | pilot smoke benchmark（小规模冒烟测试） | 只证明流程跑通。 |

## 当前阶段结论

2026年7月21日冻结的publication-v2正式证据继续绑定step14000。新的Stage B三GPU全局无放回续训已从step14000启动并继续向step50000运行，每500步永久保存checkpoint。A类10项和B类7项全部保留，待17项协议冻结后手动评估候选checkpoint。历史Stage C1只完成64K计算gate并停在step569，不是当前运行。
