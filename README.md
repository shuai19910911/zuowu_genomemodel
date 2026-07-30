# CropGenome-FM

面向作物基因组序列的基础模型。仓库只展示当前状态、最新训练曲线、核心正式结果和最小复现代码；旧试验和逐checkpoint诊断已从main移除，仍可在Git历史中恢复。

## 当前状态

- Stage B 8K续训：从step14000精确恢复，3×A100全局无放回训练，目标step50000。
- 当前validation-best：step24000；选择指标为`MLM loss + 0.02 × RC loss`。
- step19000非EDTA下游：A1–A10、B13–B17正在RTX 2080 Ti集群执行。
- B11/B12：等待EDTA正式标签，不使用替代标签补数。

详细实时快照、曲线和结果边界见：[TRAINING_PROGRESS.md](TRAINING_PROGRESS.md)

## 快速入口

- [训练与下游进展](TRAINING_PROGRESS.md)
- [模型结构](MODEL_ARCHITECTURE.md)
- [当前训练配置](training_server_transfer/configs/train_stage_B_continuation_3gpu_no_replacement.json)
- [当前模型配置](training_server_transfer/configs/model_large.json)
- [训练曲线源数据](docs/training_progress/source_data/stage_b_continuation_metrics.tsv)
- [正式基准主指标](docs/results/formal_full_data_metrics.tsv)

## 仓库边界

保留：Markdown、聚合TSV、PNG、当前配置及最小训练代码。

不上传：原始FASTA/GFF、checkpoint、embedding cache、逐样本预测、完整日志、GPU运行目录或凭据。
