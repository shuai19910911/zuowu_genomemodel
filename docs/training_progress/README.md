# Training progress artifacts

主入口请看仓库根目录的 [TRAINING_PROGRESS.md](../../TRAINING_PROGRESS.md)。本目录只保存可复核的轻量源数据和图片，不保存 checkpoint（模型存档点）、训练日志、逐样本表、二进制输入缓存、PDF/SVG 或原始大数据。

## 当前应优先查看

| 目录/文件 | 内容 | 怎么看 |
|---|---|---|
| [cropgenome_bench_v1_formal_lite_test/](cropgenome_bench_v1_formal_lite_test/) | 2080Ti fixed formal-lite benchmark（固定轻量正式化测试） | 用于 step14000 vs step17000 阶段决策；step17000 是主候选。 |
| [figures/v2_stable_stageB_loss.png](figures/v2_stable_stageB_loss.png) | v2 Stable total loss（总损失）曲线 | 看训练是否稳定下降。 |
| [figures/v2_stable_stageB_selection_loss.png](figures/v2_stable_stageB_selection_loss.png) | selection loss（选择损失）曲线 | 看 checkpoint 选择指标。 |
| [source_data/v2_stable_stageB_metrics.tsv](source_data/v2_stable_stageB_metrics.tsv) | 训练/验证指标原始轻量表 | 训练曲线源数据。 |

## 历史/辅助结果

| 目录 | 内容 | 边界 |
|---|---|---|
| [downstream_evaluations_2080/](downstream_evaluations_2080/) | step1000–17000 full-region diagnostic probe（完整区域诊断探针） | 只用于观察 checkpoint 趋势。 |
| [cropgenome_bench_v1_medium_validation/](cropgenome_bench_v1_medium_validation/) | medium validation-only benchmark（中等规模验证集基准） | 只用于候选选择，不是正式 test。 |
| [cropgenome_bench_v1_pilot_smoke/](cropgenome_bench_v1_pilot_smoke/) | pilot smoke benchmark（小规模冒烟测试） | 只证明流程跑通。 |

## 当前阶段结论

训练冻结到 step17000；formal-lite test 的 3 任务平均模型向量 F1 为 step14000=0.6961、step17000=0.7439。step17000 是阶段主候选，但 step14000 因 TES_polyA/promoter_TSS 更好而保留为备选。最终论文结论仍需 GFF-derived CropGenome-Bench v1。
