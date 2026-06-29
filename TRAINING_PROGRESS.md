# CropGenome-FM 训练进展与评估

更新时间: 2026-06-29 08:20 CST

本文件是 GitHub 唯一进展入口。用户只需要看本文件；下游 probe（探针评测）明细、confusion matrix（混淆矩阵）、per-class metrics（逐类别指标）、run manifest（运行清单）和训练日志只保留在本地，不上传 GitHub。

## 0. 当前一句话结论

`CropGenome-FM-v2-Stable-8K`（作物基因组基础模型第二版稳健 8192 碱基版）已从 step1000 checkpoint（模型存档点）恢复并继续训练到 step2530；step2000 validation（验证）明显优于 step1000，`checkpoint_best.pt`（最佳模型存档点）已更新到 step2000。

当前最重要结论: 预训练 validation selection loss（验证选择损失）从 step1000 的 1.2375897 降到 step2000 的 1.1890236，训练方向正常。step1000 的轻量下游 probe（探针评测）已完成，但只是早期弱阳性，不是正式 splice/promoter/TES benchmark（剪接/启动子/转录终止基准评测）。

## 1. 当前训练状态

| 项目 | 当前值 | 解释 |
|---|---:|---|
| 训练版本 | `v2_stable_from_scratch` | v2 Stable（第二版稳健版）正式 from scratch（从头训练）；恢复只用于同一 run（训练轮次）中断续跑。 |
| 当前 step（训练步） | 2530 | 已超过 step2000，继续向 step3000 前进。 |
| 最新 train loss（训练损失） | 1.2172345 | 训练损失继续下降。 |
| 最新 train MLM loss（遮盖碱基预测损失） | 1.1400132 | 主预训练目标继续改善。 |
| 最新 train selection loss（选择损失） | 1.1417660 | 训练选择损失继续下降。 |
| 最新 validation（验证） | step2000 | 下一次验证/保存是 step3000。 |
| step2000 val loss（验证总损失） | 1.2675664 | 比 step1000 更好。 |
| step2000 val selection loss（验证选择损失） | 1.1890236 | 当前 best checkpoint（最佳模型存档点）依据。 |
| 当前 checkpoint（模型存档点） | `checkpoint_best.pt`, `step_00001000.pt`, `step_00002000.pt` | checkpoint 文件本地保留，不上传 GitHub。 |
| A100 GPU2 | 约 32.3GB / 40GB, 100% | 主训练正常运行。 |
| 下一 checkpoint | step3000 | 预计 2026-06-29 14:41 CST 左右。 |

## 2. 核心训练曲线

GitHub 只保留两张最有判断价值的曲线，避免文件树再次变乱。其他 RC loss（反向互补损失）、region loss/acc（区域辅助损失/准确率）图本地保留，必要时再汇总进本文，不单独作为入口。

### 2.1 Total loss（总损失）

![v2 total loss](docs/training_progress/figures/v2_stable_stageB_loss.png)

- 图: [docs/training_progress/figures/v2_stable_stageB_loss.png](docs/training_progress/figures/v2_stable_stageB_loss.png)
- 源数据: [docs/training_progress/source_data/v2_stable_stageB_metrics.tsv](docs/training_progress/source_data/v2_stable_stageB_metrics.tsv)

解释: total loss（总损失）综合主 MLM loss（遮盖碱基预测损失）、小权重 RC consistency（反向互补一致性）和区域辅助项。曲线只用于监控训练是否稳定，不等同于下游 benchmark（基准评测）成功。

### 2.2 Selection loss（选择损失）

![v2 selection loss](docs/training_progress/figures/v2_stable_stageB_selection_loss.png)

- 图: [docs/training_progress/figures/v2_stable_stageB_selection_loss.png](docs/training_progress/figures/v2_stable_stageB_selection_loss.png)
- 曲线摘要: [docs/training_progress/source_data/v2_stable_stageB_curve_summary.tsv](docs/training_progress/source_data/v2_stable_stageB_curve_summary.tsv)

解释: selection loss（选择损失）定义为 `MLM loss + 0.02 × RC loss`。best checkpoint（最佳模型存档点）和 early stopping（早停）按 validation selection loss（验证选择损失）判断，不把 region loss（区域辅助损失）放进主选择指标。

## 3. Validation（验证）趋势

| checkpoint（模型存档点） | val loss（验证总损失） | val MLM loss（验证遮盖损失） | val RC loss（验证反向互补损失） | val selection loss（验证选择损失） | 结论 |
|---|---:|---:|---:|---:|---|
| step1000 | 1.3238202 | 1.2372975 | 0.0146100 | 1.2375897 | 第一个可用 checkpoint；恢复训练从这里继续。 |
| step2000 | 1.2675664 | 1.1879575 | 0.0533046 | 1.1890236 | 明显优于 step1000，当前 best checkpoint。 |

评估: step2000 的 validation selection loss 下降约 0.0486，这是实质改善。当前应继续训练观察 step3000/4000/5000，而不是在 step2000 就停止。

## 4. 下游 probe（探针评测）摘要

GitHub 不上传 `docs/training_progress/downstream_evaluations/step_*/...` 明细目录；该目录只在本地保留，供自动 watcher（检查器）判断某个 checkpoint 是否已经评估过。对外只在本节保留最小摘要。

### 4.1 step1000 full region annotation probe（完整区域注释探针）

状态: 已完成，2026-06-28 13:31 CST，RTX 2080 Ti 上运行。

| 方法 | Accuracy（准确率） | Macro-F1（类别平均 F1） | Balanced accuracy（类别均衡准确率） | 解释 |
|---|---:|---:|---:|---|
| 1-mer nearest centroid（单碱基组成最近中心基线） | 0.1875 | 0.1592 | 0.1875 | 最低限度序列组成 baseline（基线）。 |
| model embedding nearest centroid（模型向量最近中心） | 0.1964 | 0.1764 | 0.1964 | 略高于 1-mer，说明 step1000 表示有弱信号。 |
| model region head argmax（区域预测头直接分类） | 0.1518 | 0.0702 | 0.1518 | 较弱；region head 只能作健康检查。 |

评估: step1000 下游结果是弱阳性，不是正式成功。它只能说明 embedding（向量表示）比简单单碱基组成略好；正式结论必须等 splice/promoter/TES 等独立 benchmark（基准评测）和后续 checkpoint 趋势。

### 4.2 step2000 下游状态

step2000 checkpoint 已产生，但 step2000 下游 probe 尚未完成。原因是原计划使用 gpu05 的 2080Ti 自动评估，而 gpu05 曾出现 `No route to host`（无法路由到主机）。A100 GPU4 目前被其他用户任务占用 13.4GB 显存且 GPU 利用率 100%，不建议抢占或共用。

## 5. 文件整理规则

为了避免 GitHub 再次变成一堆目录，后续固定执行下面规则:

1. GitHub 只看 `README.md`、`PROJECT_PLAN.md`、`MODEL_ARCHITECTURE.md`、`TRAINING_PROGRESS.md`。
2. `docs/training_progress/` 只跟踪少量核心 PNG（位图）曲线和必要 TSV（表格源数据）。
3. `docs/training_progress/downstream_evaluations/` 只本地保留，不上传 GitHub。
4. 旧 `docs/downstream/*`、`docs/training_curves/*`、临时 handoff（交接）文档、旧训练指标散表都不再作为 GitHub 入口。
5. checkpoint（模型存档点）、训练日志、run manifest（运行清单）、per-class/confusion 明细和逐样本预测一律本地保留。

## 6. 下一步

| 事件 | 要做什么 | 判断标准 |
|---|---|---|
| step3000 validation（验证） | 更新本文件和核心曲线 | val selection loss 是否继续低于 1.1890。 |
| step2000 下游 probe（探针评测） | 等 2080 网络恢复或找到真正空闲 GPU 后运行 | 不抢占别人 GPU；结果只写摘要进本文件。 |
| step5000 | 进入 early stopping（早停）观察期 | 若 validation 连续无有效改善，再考虑早停或调参。 |
| 第一轮正式 benchmark（基准评测） | 构建 splice/promoter/TES 等独立任务 | 必须包含 1-mer/CNN/公开模型或可解释基线。 |
