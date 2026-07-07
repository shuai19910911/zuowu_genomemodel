# CropGenome-FM 训练进展与评估

更新时间: 2026-07-07 16:05 CST

本文件是 GitHub 唯一进展入口。用户只需要看本文件；本次 2080 GPU 下游 probe（探针评测）只上传轻量 TSV（表格）、JSON（运行清单）和 PNG（位图），不上传 checkpoint（模型存档点）、训练日志、PDF/SVG（矢量图）或原始大数据。

## 0. 当前一句话结论

`CropGenome-FM-v2-Stable-8K`（作物基因组基础模型第二版稳健 8192 碱基版）已训练到 step17000 并触发 early stopping（早停）。训练本身应先冻结；当前需要把 step14000 和 step17000 同时保留为正式 benchmark（基准评测）前的候选点，而不是只凭 `checkpoint_best.pt` 或单一 validation loss（验证损失）下结论。

当前最重要结论: validation selection loss（验证选择损失）从 step10000 的 1.0897 降到 step17000 的 1.0729；step17000 在 full-region diagnostic probe（完整区域诊断探针）上也达到当前最高，embedding Macro-F1（向量类别平均 F1）= 0.3030、region head Macro-F1（区域预测头类别平均 F1）= 0.3213。但 medium validation-only benchmark（中等规模验证集基准，只用于选 checkpoint，不是正式 test）显示任务间不一致：step14000 最强 promoter_TSS，step17000 最强 splice_acceptor，TES_polyA 仍由更早 checkpoint 略强。因此阶段结论是“训练已可冻结，正式测试前候选集保留 step14000 + step17000”。

## 1. 当前训练状态

| 项目 | 当前值 | 解释 |
|---|---:|---|
| 训练版本 | `v2_stable_from_scratch` | v2 Stable（第二版稳健版）正式 from scratch（从头训练）；恢复只用于同一 run（训练轮次）中断续跑。 |
| 当前 step（训练步） | 17000 | 训练日志在 step17000 触发 early stopping（早停），当前没有我们的 A100 训练进程继续运行。 |
| 最新 train loss（训练损失） | 1.0611 | 训练单点有噪声；最终 checkpoint 判断以 validation/downstream validation-only 为主。 |
| 最新 train selection loss（选择损失） | 0.9886 | selection loss = MLM loss + 0.02 × RC loss（遮盖碱基预测损失加小权重反向互补一致性损失）。 |
| 最新 validation（验证） | step17000 | 最新验证点也是 TSV 中最低 selection loss。 |
| step17000 val loss（验证总损失） | 1.1410 | 比 step14000 的 validation total loss 更低。 |
| step17000 val selection loss（验证选择损失） | 1.0729 | TSV 口径当前最低；训练日志里 `checkpoint_best.pt` 仍指向 step14000，需用下游验证解冲突。 |
| step17000 full-region probe | embedding F1 0.3030; region head F1 0.3213 | 诊断性 probe 当前最高，但不是正式论文 benchmark。 |
| medium validation-only 结果 | step14000: promoter_TSS 最强；step17000: splice_acceptor 最强 | 说明不能只选一个“全任务最优” checkpoint；正式 test 前候选集保留 step14000 + step17000。 |
| 当前 checkpoint（模型存档点） | `step_00014000.pt`, `step_00017000.pt`, `checkpoint_best.pt` | checkpoint 文件本地保留，不上传 GitHub；`checkpoint_best.pt` 与 TSV/probe 信号存在冲突。 |
| 2080Ti 评估机 | step11000–17000 probe + step14000/17000 medium validation 已完成 | GPU 已完成本轮任务；公开只提交 TSV/JSON/PNG 轻量产物。 |

## 2. 核心训练曲线

GitHub 只保留两张最有判断价值的曲线，避免文件树再次变乱。两张核心曲线已改为 Python matplotlib（Python 绘图库）正式坐标图，包含标题、横坐标、纵坐标、刻度、图例、validation checkpoints（验证检查点）和 latest train point（最新训练点）。其他 RC loss（反向互补损失）、region loss/acc（区域辅助损失/准确率）图本地保留，必要时再汇总进本文，不单独作为入口。

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

| checkpoint（模型存档点） | val loss（验证总损失） | val selection loss（验证选择损失） | region acc（区域准确率） | 结论 |
|---|---:|---:|---:|---|
| step1000 | 1.3238 | 1.2376 | 0.3438 | 早期 validation checkpoint。 |
| step2000 | 1.2676 | 1.1890 | 0.4062 | 早期 validation checkpoint。 |
| step3000 | 1.2348 | 1.1591 | 0.4688 | 早期 validation checkpoint。 |
| step4000 | 1.2234 | 1.1468 | 0.3750 | 早期 validation checkpoint。 |
| step5000 | 1.2107 | 1.1354 | 0.4062 | 早期 validation checkpoint。 |
| step6000 | 1.1995 | 1.1253 | 0.4062 | 早期 validation checkpoint。 |
| step7000 | 1.1886 | 1.1149 | 0.4375 | 早期 validation checkpoint。 |
| step8000 | 1.1774 | 1.1039 | 0.4531 | 早期 validation checkpoint。 |
| step9000 | 1.1705 | 1.0960 | 0.4688 | 早期 validation checkpoint。 |
| step10000 | 1.1647 | 1.0897 | 0.4531 | 后期候选 checkpoint；用于趋势判断。 |
| step11000 | 1.1645 | 1.0864 | 0.4219 | 后期候选 checkpoint；用于趋势判断。 |
| step12000 | 1.1573 | 1.0820 | 0.4844 | 后期候选 checkpoint；用于趋势判断。 |
| step13000 | 1.1531 | 1.0764 | 0.4688 | 后期候选 checkpoint；用于趋势判断。 |
| step14000 | 1.1470 | 1.0743 | 0.5312 | 训练日志 `checkpoint_best.pt` 指向该步；medium promoter_TSS 当前最强。 |
| step15000 | 1.1509 | 1.0753 | 0.4375 | 后期候选 checkpoint；用于趋势判断。 |
| step16000 | 1.1454 | 1.0766 | 0.4844 | 后期候选 checkpoint；用于趋势判断。 |
| step17000 | 1.1410 | 1.0729 | 0.5312 | TSV 当前最低；early stop 发生点；full-region probe 当前最强。 |

评估: validation selection loss（验证选择损失）总体从 step1000 的 1.2376 降到 step17000 的 1.0729，说明预训练在验证集上持续改善并已到 early stop（早停）阶段。需要特别说明：训练日志中的 `checkpoint_best.pt` 仍指向 step14000，但 TSV 最新 validation 和 full-region probe 更支持 step17000；因此不能只看一个文件名决定最终模型，必须结合 medium validation-only benchmark（中等规模验证集基准）和后续正式 benchmark（正式基准评测）。

## 4. 公正评估与外部指标尺度参考

### 4.1 当前训练是否有效

公正结论: 当前预训练有效，且训练阶段应先冻结到 step17000；但下游仍不能写成正式论文胜利，因为 full-region probe 和 medium validation-only 都是 diagnostic/validation-only（诊断/验证集选择）证据，不是 formal test（正式测试集）证据。

| 证据 | 当前数字 | 怎么看 | 结论 |
|---|---:|---|---|
| validation selection loss（验证选择损失） | step1000: 1.2376 → step10000: 1.0897 → step17000: 1.0729 | 验证选择损失持续下降，step17000 是 TSV 当前最低。 | 训练有效，已到可冻结阶段。 |
| full-region embedding probe（完整区域向量探针） | step9000: 0.2636 → step17000: 0.3030 | step17000 明显高于此前 step9000 峰值。 | 诊断信号支持 step17000。 |
| full-region region head（区域预测头） | step9000: 0.2537 → step17000: 0.3213 | 区域头在 step17000 达到当前最高。 | 辅助头也支持 step17000。 |
| medium TES_polyA | best≈step5000/10000; step17000: 0.8106 | TES proxy 没有随训练继续增强。 | 不能声称 step17000 全任务最优。 |
| medium promoter_TSS | step14000: 0.7592; step17000: 0.7259 | promoter proxy 支持 step14000。 | step14000 需保留为候选。 |
| medium splice_acceptor | step14000: 0.5197; step17000: 0.6825 | splice proxy 在 step17000 大幅增强并超过 1-mer。 | step17000 是主候选，但仍需正式 benchmark。 |

保守判断: Stage_B 训练可停止/冻结；正式 test 前保留 step14000 与 step17000 两个候选。论文主表仍必须来自 GFF-derived hard negatives（GFF 精确构建硬负样本）、固定 split（固定划分）、多 seed（多随机种子）和外部模型同口径比较。

### 4.2 与公开基因组模型指标的尺度参考

重要限制: 下面不是直接胜负比较。我们的 full region annotation probe（完整区域注释探针）是自定义 7 类小样本任务；公开模型通常报告 GenomicBenchmarks（基因组分类基准）、GUE（Genome Understanding Evaluation，基因组理解评测）、NT tasks（Nucleotide Transformer 任务集）等标准任务。预训练 loss（损失）也不能跨模型直接比较，因为 tokenizer（分词方式）、masking objective（遮盖目标）、物种/序列长度、训练数据都不同。

| 来源/模型 | 公开任务与指标 | 公开数字 | 对我们的含义 |
|---|---|---:|---|
| HyenaDNA（Nguyen et al., 2023） | GenomicBenchmarks Top-1 Accuracy（8 个短序列分类任务平均准确率） | HyenaDNA 约 88.46%；CNN baseline 约 79.24%；DNABERT 约 83.05%。 | 公开标准分类任务的成熟模型通常在 70%–97% accuracy 区间；我们的 step1000 probe Accuracy 0.1964 不能按同一尺度解读。 |
| Caduceus（Schiff et al., 2024） | GenomicBenchmarks Top-1 Accuracy（5-fold CV，5 折交叉验证） | Caduceus-PS 约 0.869；Caduceus-Ph 约 0.866；HyenaDNA 约 0.853；CNN 约 0.803。 | 这些是标准二分类/近二分类任务，且样本和 protocol（流程）成熟；我们当前 probe 只是内部早期诊断。 |
| Caduceus / NT tasks（Dalla-Torre et al. 任务） | Histone/enhancer 用 MCC（马修斯相关系数），promoter/splice 用 F1/accuracy（F1/准确率） | 例: promoter all F1: NT-v2 0.976、DNABERT-2 0.971、HyenaDNA 0.960、Caduceus-Ph 0.970；splice all accuracy: NT-v2 0.983、HyenaDNA 0.956、Caduceus-Ph 0.940。 | promoter/splice 这类公开任务接近饱和；我们必须后续做同口径 splice/promoter/TES benchmark，不能拿当前小 probe 代替。 |
| DNABERT-2（Zhou et al., ICLR 2024） | GUE 28 个数据集平均 evaluation score（平均评测分数，混合 F1/MCC） | DNABERT-2 66.80；DNABERT-2 继续在 GUE 训练集预训练后 67.77；NT-2500M-multi 66.93；NT-2500M-1000g 61.41。 | 这是跨任务平均分，不等同 accuracy；可作为“成熟公开模型在标准 benchmark 上约 60–70 分”的尺度。 |
| HyenaDNA ultra-long species classification（超长物种分类） | 5-way species Top-1 Accuracy（5 类物种准确率） | 1k: 61.1%；32k: 93.4%；250k: 97.9%；450k: 99.4%；1M: 99.5%。 | 长上下文任务可很高，但任务类型完全不同；不能用来证明我们当前 8K 作物模型好坏。 |

当前最公平的对比方式: 下一步不要再只看 step1000 内部 probe，应在固定 benchmark 上对比 1-mer/CNN/公开模型或公开模型 embedding（向量表示）。只有同一 split（划分）、同一 metric（指标）、同一任务的结果，才可写成论文主表。

## 5. 下游 probe（探针评测）摘要

本节记录已经真实在 gpu05 的 RTX 2080 Ti 上复跑的 lightweight downstream probe（轻量下游探针）。GitHub 只上传轻量 TSV（表格）、JSON（运行清单）和 PNG（位图）；不上传 PDF/SVG（矢量图）、checkpoint（模型存档点）、训练日志或原始大数据。

运行环境:

- 机器: `gpu05`，NVIDIA GeForce RTX 2080 Ti。
- GPU: `CUDA_VISIBLE_DEVICES=0`。
- 脚本: `scripts/evaluate_v2_checkpoint_region_probe.py`。
- model config（模型配置）: `training_server_transfer/configs/model_large.json`。第一次误用 `model_large_v2_no_region.json` 会因 checkpoint 含 `region_head` 权重而失败，已改正后复跑。
- 采样: 每类 train/eval 最多 32 个窗口；共 224 train + 224 eval 样本。
- 任务边界: 这是 `full_region_annotation_probe`（完整区域注释探针），只用于诊断 embedding（向量表示）和 region head（区域预测头）是否有早期信号，不进入论文主表。

### 5.1 2080Ti step1000 / step2000 / step3000 对比

| checkpoint（模型存档点） | 方法 | Accuracy（准确率） | Macro-F1（类别平均 F1） | Balanced accuracy（类别均衡准确率） | 解释 |
|---|---|---:|---:|---:|---|
| step1000 | 1-mer nearest centroid（单碱基组成最近中心基线） | 0.1875 | 0.1542 | 0.1875 | 最低限度序列组成 baseline（基线）。 |
| step1000 | model embedding nearest centroid（模型向量最近中心） | 0.2143 | 0.1742 | 0.2143 | 比 1-mer Macro-F1 高 0.0199，弱阳性。 |
| step1000 | model region head argmax（区域预测头直接分类） | 0.1518 | 0.0777 | 0.1518 | 很弱，只能作健康检查。 |
| step2000 | 1-mer nearest centroid（单碱基组成最近中心基线） | 0.1875 | 0.1542 | 0.1875 | 同一采样和同一 baseline，便于比较 checkpoint。 |
| step2000 | model embedding nearest centroid（模型向量最近中心） | 0.2589 | 0.2053 | 0.2589 | 比 step1000 高 0.0311；比 1-mer baseline 高 0.0511。 |
| step2000 | model region head argmax（区域预测头直接分类） | 0.2232 | 0.1646 | 0.2232 | 比 step1000 明显提高，但仍只作辅助健康检查。 |
| step3000 | 1-mer nearest centroid（单碱基组成最近中心基线） | 0.1875 | 0.1542 | 0.1875 | 同一采样和同一 baseline，便于比较 checkpoint。 |
| step3000 | model embedding nearest centroid（模型向量最近中心） | 0.2679 | 0.2195 | 0.2679 | 比 step2000 高 0.0142；比 1-mer baseline 高 0.0653。 |
| step3000 | model region head argmax（区域预测头直接分类） | 0.2411 | 0.1853 | 0.2411 | 比 step2000 region head 继续提高，但仍只作辅助健康检查。 |

结论: step3000 在同一 2080Ti、同一采样、同一 probe 上继续优于 step2000，和 validation selection loss（验证选择损失）下降方向一致。这个结果支持继续训练到 step4000/5000，但不能单独证明正式下游成功。

### 5.2 2080Ti 下游结果文件

step1000:

- 图: [docs/training_progress/downstream_evaluations_2080/step_00001000/full_region_annotation_probe/figures/region_probe_macro_f1.png](docs/training_progress/downstream_evaluations_2080/step_00001000/full_region_annotation_probe/figures/region_probe_macro_f1.png)
- 指标: [docs/training_progress/downstream_evaluations_2080/step_00001000/full_region_annotation_probe/source_data/metrics_summary.tsv](docs/training_progress/downstream_evaluations_2080/step_00001000/full_region_annotation_probe/source_data/metrics_summary.tsv)
- run manifest（运行清单）: [docs/training_progress/downstream_evaluations_2080/step_00001000/full_region_annotation_probe/run_manifest.json](docs/training_progress/downstream_evaluations_2080/step_00001000/full_region_annotation_probe/run_manifest.json)

step2000:

- 图: [docs/training_progress/downstream_evaluations_2080/step_00002000/full_region_annotation_probe/figures/region_probe_macro_f1.png](docs/training_progress/downstream_evaluations_2080/step_00002000/full_region_annotation_probe/figures/region_probe_macro_f1.png)
- 指标: [docs/training_progress/downstream_evaluations_2080/step_00002000/full_region_annotation_probe/source_data/metrics_summary.tsv](docs/training_progress/downstream_evaluations_2080/step_00002000/full_region_annotation_probe/source_data/metrics_summary.tsv)
- run manifest（运行清单）: [docs/training_progress/downstream_evaluations_2080/step_00002000/full_region_annotation_probe/run_manifest.json](docs/training_progress/downstream_evaluations_2080/step_00002000/full_region_annotation_probe/run_manifest.json)

### 5.3 解释边界

这个 2080Ti probe 结果可以写成“checkpoint 质量早期改善的诊断证据”，不能写成“作物基因组基础模型已经在正式下游任务上成功”。正式论文主结论仍必须来自 CropGenome-Bench v1（作物基因组正式基准评测）中的 splice/promoter/TES、跨作物迁移、低样本效率和强外部模型对比。

## 6. 文件整理规则

为了避免 GitHub 再次变成一堆目录，后续固定执行下面规则:

1. GitHub 只看 `README.md`、`PROJECT_PLAN.md`、`MODEL_ARCHITECTURE.md`、`TRAINING_PROGRESS.md`。
2. `docs/training_progress/` 只跟踪少量核心 PNG（位图）曲线和必要 TSV（表格源数据）。
3. `docs/training_progress/downstream_evaluations_2080/` 只上传本次 2080Ti probe 的轻量 TSV/JSON/PNG；旧 `docs/training_progress/downstream_evaluations/` 明细目录仍只本地保留。
4. 旧 `docs/downstream/*`、`docs/training_curves/*`、临时 handoff（交接）文档、旧训练指标散表都不再作为 GitHub 入口。
5. checkpoint（模型存档点）、训练日志、PDF/SVG、逐样本预测和原始大数据一律不上传 GitHub；本次 per-class/confusion TSV 和 run manifest 体积很小，可作为可复核源数据上传。

## 7. 下一步: CropGenome-Bench v1 正式下游评估

正式任务注册文件: [training_server_transfer/configs/downstream_v2_benchmark.json](training_server_transfer/configs/downstream_v2_benchmark.json)。

目标: 不再只看 step1000 内部 probe（探针），而是建立 CropGenome-Bench v1（作物基因组正式基准），检验 CropGenome-FM 是否在作物任务上优于通用 DNA 大模型。

| 优先级 | 事件 | 要做什么 | 判断标准 |
|---|---|---|---|
| P0 | step4000 validation（验证） | 更新本文件和核心曲线 | val selection loss 是否继续低于 1.1591。 |
| P0 | core gene syntax（核心基因结构语法） | 构建 splice hard-negative、exon/intron/UTR segmentation、TIS/TTS context 三个任务 | 同一 split 下比较 random/majority/k-mer/CNN/DNABERT-2/HyenaDNA/Caduceus/CropGenome-FM。 |
| P0 | crop regulatory elements（作物调控元件） | 构建 promoter/TSS hard-negative、TES/polyA；ATAC/ACR 只在 assembly QC 通过后加入 | 主指标用 AUPRC/MCC/Macro-F1，不只看 accuracy。 |
| P0 | transfer and few-shot（迁移与少样本） | 做 species/genus holdout、monocot-to-dicot、1%/5%/10% label panels | 若作物预训练有效，跨作物保留率和少标签增益应高于通用模型。 |
| P1 | crop-specific structure（作物结构特色） | EDTA 高置信后做 TE boundary；Stage C/D 后做 long-context gene boundary | 标签 QC 未过关前不进主结论。 |
| P1 | variant/QTL ranking（变异和育种排序） | 只用已发表 processed VCF/GWAS/QTL/eQTL 表做 ref/alt delta 与候选基因排序 | 不下载 WGS/FASTQ/BAM 原始重测序，不重做 variant calling。 |
| P0 | 公平比较 | 所有模型同一 train/valid/test split、同一输入窗口、同一下游头、5 seeds mean ± std | 只有同任务同协议结果能进入论文主表。 |

论文主结论门槛: 至少 3 个 P0 作物任务完成固定 test（测试集）评估；CropGenome-FM 平均超过最强可运行通用/植物 DNA 模型至少 3% relative improvement（相对提升）；至少一个 species/genus holdout 或 low-homology holdout 中仍保留收益。


### Step4000 validation update

当前训练源数据已同步到 train step 4370，最近 validation checkpoint（验证检查点）为 step 4000。validation selection loss（验证选择损失）为 1.1468366，低于 step3000 的 1.1591112，说明预训练仍在改善。step4000 downstream probe（下游诊断探针）和 CropGenome-Bench v1 quick pilot（流程试跑）按相同口径执行；结果只作为诊断和流程验证，不作为正式 benchmark 主结果。

### Step4000 downstream update (诊断结果，不是正式主结果)

训练已同步到 train step 4370；最近 validation checkpoint 为 step 4000，validation selection loss = 1.1468366，低于 step3000 的 1.1591112。

#### Full-region diagnostic probe（完整区域诊断探针）

| checkpoint | embedding Macro-F1 | region head Macro-F1 | 说明 |
|---|---:|---:|---|
| step1000 | 0.1742 | 0.0777 | sampled region probe; diagnostic only |
| step2000 | 0.2053 | 0.1646 | sampled region probe; diagnostic only |
| step3000 | 0.2195 | 0.1853 | sampled region probe; diagnostic only |
| step4000 | 0.2315 | 0.1594 | sampled region probe; diagnostic only |

#### CropGenome-Bench v1 quick pilot（流程试跑；Stage_B proxy 标签，不是正式 benchmark）

| task | step3000 embedding F1 | step4000 embedding F1 | Δ | 解释 |
|---|---:|---:|---:|---|
| TES_polyA | 0.8467 | 0.8421 | -0.0046 | 下降/持平; pilot proxy only |
| promoter_TSS | 0.7111 | 0.7188 | +0.0076 | 提升; pilot proxy only |
| splice_acceptor | 0.6043 | 0.6000 | -0.0043 | 下降/持平; pilot proxy only |

当前口径：step4000 validation loss 继续改善，full-region diagnostic probe 也较 step3000 略好；CropGenome-Bench v1 quick pilot 是 mixed diagnostic signal（混合诊断信号），TES/polyA 与 promoter/TSS 基本持平或略升，splice proxy 略降。不能把 pilot 写成正式 benchmark 成功。下一步需要 GFF-derived hard negatives（由 GFF 精确构建的硬负样本）、partial metrics/progress log（分段落盘/进度日志）和 embedding cache（向量缓存）后再跑中等/正式规模。

### Step5000/6000 downstream update (诊断结果，不是正式主结果)

训练源数据已同步到 train step 7110；最新 validation checkpoint（验证检查点）为 step 6000，validation selection loss = 1.1252513。step5000/6000 checkpoint 均已在 RTX 2080 Ti 上完成 full-region diagnostic probe（完整区域诊断探针）和 CropGenome-Bench v1 quick pilot（流程试跑）。

#### Validation trend（验证趋势）

| checkpoint | validation selection loss | 说明 |
|---|---:|---|
| step1000 | 1.2376 | validation checkpoint |
| step2000 | 1.1890 | validation checkpoint |
| step3000 | 1.1591 | validation checkpoint |
| step4000 | 1.1468 | validation checkpoint |
| step5000 | 1.1354 | validation checkpoint |
| step6000 | 1.1253 | validation checkpoint |

#### Full-region diagnostic probe（完整区域诊断探针）

| checkpoint | embedding Macro-F1 | region head Macro-F1 | 说明 |
|---|---:|---:|---|
| step1000 | 0.1742 | 0.0777 | sampled region probe; diagnostic only |
| step2000 | 0.2053 | 0.1646 | sampled region probe; diagnostic only |
| step3000 | 0.2195 | 0.1853 | sampled region probe; diagnostic only |
| step4000 | 0.2315 | 0.1594 | sampled region probe; diagnostic only |
| step5000 | 0.2619 | 0.1588 | sampled region probe; diagnostic only |
| step6000 | 0.2316 | 0.2191 | sampled region probe; diagnostic only |

#### CropGenome-Bench v1 quick pilot（Stage_B proxy 标签；不是正式 benchmark）

| task | step3000 F1 | step4000 F1 | step5000 F1 | step6000 F1 | step6000 vs 1-mer | 解释 |
|---|---:|---:|---:|---:|---:|---|
| TES_polyA | 0.8467 | 0.8421 | 0.8550 | 0.8702 | +0.2548 | 超过 1-mer baseline; pilot proxy only |
| promoter_TSS | 0.7111 | 0.7188 | 0.7031 | 0.7143 | +0.0853 | 超过 1-mer baseline; pilot proxy only |
| splice_acceptor | 0.6043 | 0.6000 | 0.5735 | 0.5839 | +0.0054 | 超过 1-mer baseline; pilot proxy only |

当前口径：validation loss 从 step1000 到 step6000 持续下降；full-region diagnostic probe 在 step5000 达到当前最高 embedding Macro-F1，step6000 embedding 回落但 region head 明显恢复；CropGenome-Bench v1 quick pilot 三个 proxy task 在 step6000 仍均超过 1-mer baseline（单碱基组成基线），但不同 task 存在小样本波动。以上均为 diagnostic/pilot 证据，不能写成正式 CropGenome-Bench v1 主结果。正式结果仍需 GFF-derived hard negatives、固定 split、多 seed、外部模型同口径比较。

### Step7000 downstream update (诊断结果，不是正式主结果)

训练源数据已同步到 train step 7110；最新 validation checkpoint 为 step 7000，validation selection loss = 1.1148912，继续低于 step6000 的 1.1252513。step7000 checkpoint 已在 RTX 2080 Ti 上完成 full-region diagnostic probe（完整区域诊断探针）和 CropGenome-Bench v1 quick pilot（流程试跑）。

#### Full-region diagnostic probe（完整区域诊断探针）

| checkpoint | embedding Macro-F1 | region head Macro-F1 | 说明 |
|---|---:|---:|---|
| step1000 | 0.1742 | 0.0777 | sampled region probe; diagnostic only |
| step2000 | 0.2053 | 0.1646 | sampled region probe; diagnostic only |
| step3000 | 0.2195 | 0.1853 | sampled region probe; diagnostic only |
| step4000 | 0.2315 | 0.1594 | sampled region probe; diagnostic only |
| step5000 | 0.2619 | 0.1588 | sampled region probe; diagnostic only |
| step6000 | 0.2316 | 0.2191 | sampled region probe; diagnostic only |
| step7000 | 0.2605 | 0.1588 | sampled region probe; diagnostic only |

#### CropGenome-Bench v1 quick pilot（Stage_B proxy 标签；不是正式 benchmark）

| task | step5000 F1 | step6000 F1 | step7000 F1 | step7000 vs 1-mer | 解释 |
|---|---:|---:|---:|---:|---|
| TES_polyA | 0.8550 | 0.8702 | 0.8462 | +0.2308 | 超过 1-mer baseline; pilot proxy only |
| promoter_TSS | 0.7031 | 0.7143 | 0.7132 | +0.0841 | 超过 1-mer baseline; pilot proxy only |
| splice_acceptor | 0.5735 | 0.5839 | 0.5630 | -0.0155 | 未超过 1-mer baseline; pilot proxy only |

当前口径：step7000 validation 是当前最好验证点；full-region embedding Macro-F1 与 step5000 基本持平但略低，region head 低于 step6000；quick pilot 中 TES/polyA 继续增强，promoter/TSS 与 splice proxy 有小样本波动。以上仍是 diagnostic/pilot 证据，不能写成正式 CropGenome-Bench v1 主结果。

#### 正式 CropGenome-Bench v1 评测节奏

现在有必要正式开始“构建”CropGenome-Bench v1：冻结 GFF-derived hard negatives（由 GFF 精确构建的硬负样本）、固定 train/valid/test split（训练/验证/测试划分）、实现 embedding cache（向量缓存）、定义多 seed 与外部模型同口径协议。但不建议在每个 checkpoint 上都跑完整正式 test：正式 benchmark 应只在少数候选 checkpoint 上跑，例如 validation-best、step5000/step7000、以及最终停止点；checkpoint 选择只能看 validation/probe，不应反复用 test set 做选择。每个 checkpoint 可以继续跑轻量 diagnostic/pilot，用于训练监控。

### CropGenome-Bench v1 medium validation-only benchmark

| task | step5000 F1 | step7000 F1 | step10000 F1 | step10000 vs 1-mer | 趋势 |
|---|---:|---:|---:|---:|---|
| TES_polyA | 0.8309 | 0.8192 | 0.8308 | +0.1433 | step10000 最强 |
| promoter_TSS | 0.7578 | 0.7294 | 0.7317 | +0.1615 | step5000 最强 |
| splice_acceptor | 0.4615 | 0.5164 | 0.4530 | -0.0757 | step7000 最强但不稳 |

结论：step10000 在 validation loss 最好但目前 medium proxy 没有全面优势；TES_polyA 有新最高，但 promoter_TSS 低于 step5000，splice_acceptor 在三 checkpoint 中最差且低于 1-mer。正式 CropGenome-Bench v1 仍需 GFF hard negatives + 固定 split + 多 seed。

### Step11000–17000 early-stop downstream update (诊断/验证结果，不是正式 test 主结果)

训练已到 step17000 并 early stop（早停）。本轮新增两类真实产物：1）step11000–17000 full-region diagnostic probe（完整区域诊断探针）；2）step14000 与 step17000 medium validation-only benchmark（中等规模验证集基准，只用于 checkpoint 选择）。这一步的目的就是解决 `checkpoint_best.pt` 指向 step14000，但 TSV/probe 更偏向 step17000 的冲突。

#### Validation trend（验证趋势，step10000 以后）

| checkpoint | validation selection loss | val loss | region acc | 说明 |
|---|---:|---:|---:|---|
| step10000 | 1.0897 | 1.1647 | 0.4531 | validation checkpoint |
| step11000 | 1.0864 | 1.1645 | 0.4219 | validation checkpoint |
| step12000 | 1.0820 | 1.1573 | 0.4844 | validation checkpoint |
| step13000 | 1.0764 | 1.1531 | 0.4688 | validation checkpoint |
| step14000 | 1.0743 | 1.1470 | 0.5312 | 训练日志 checkpoint_best 指向该步；promoter_TSS medium 最强 |
| step15000 | 1.0753 | 1.1509 | 0.4375 | validation checkpoint |
| step16000 | 1.0766 | 1.1454 | 0.4844 | validation checkpoint |
| step17000 | 1.0729 | 1.1410 | 0.5312 | TSV 当前最低；但不是所有 downstream proxy 最强 |

#### Full-region diagnostic probe（完整区域诊断探针）

| checkpoint | embedding Macro-F1 | region head Macro-F1 | 说明 |
|---|---:|---:|---|
| step11000 | 0.2297 | 0.1478 | sampled region probe; diagnostic only |
| step12000 | 0.2543 | 0.1068 | sampled region probe; diagnostic only |
| step13000 | 0.2322 | 0.2018 | sampled region probe; diagnostic only |
| step14000 | 0.2410 | 0.2176 | sampled region probe; diagnostic only |
| step15000 | 0.2570 | 0.1318 | sampled region probe; diagnostic only |
| step16000 | 0.2393 | 0.2122 | sampled region probe; diagnostic only |
| step17000 | 0.3030 | 0.3213 | 当前 full-region probe 最强 |

#### CropGenome-Bench v1 medium validation-only benchmark（中等规模验证集；Stage_B proxy 标签，不是正式 test）

| task | step5000 F1 | step7000 F1 | step10000 F1 | step14000 F1 | step17000 F1 | 当前最好 |
|---|---:|---:|---:|---:|---:|---|
| TES_polyA | 0.8309 | 0.8192 | 0.8308 | 0.8263 | 0.8106 | step5000 |
| promoter_TSS | 0.7578 | 0.7294 | 0.7317 | 0.7592 | 0.7259 | step14000 |
| splice_acceptor | 0.4615 | 0.5164 | 0.4530 | 0.5197 | 0.6825 | step17000 |

#### 当前 checkpoint 选择判断

- step17000 是当前 primary candidate（主候选）：TSV validation selection loss 最低、full-region embedding/region-head probe 均最高，并且 splice_acceptor medium validation F1 从 step14000 的 0.5197 跳到 0.6825。
- step14000 仍需作为 formal benchmark 前的 paired candidate（配对候选）：训练日志的 `checkpoint_best.pt` 仍指向 step14000，且 promoter_TSS medium validation F1 = 0.7592，是当前 medium 表里最高。
- TES_polyA 在 medium validation 上没有随训练继续增强，step5000/step10000 略高于 step14000/17000；说明 proxy task（代理任务）存在任务差异，不能把任何单一 checkpoint 写成“全任务最优”。
- 推荐阶段动作：冻结 Stage_B 训练；正式 test（测试集）前只保留 step14000 与 step17000 两个候选，后续用固定 split（固定划分）、GFF-derived hard negatives（由 GFF 精确构建的硬负样本）、多 seed（多随机种子）和外部模型同口径比较决定最终论文主表。

公开轻量结果：full-region 总表 `docs/training_progress/downstream_evaluations_2080/summary_2080_full_region_probe.tsv`；medium validation 总表 `docs/training_progress/cropgenome_bench_v1_medium_validation/summary_cropgenome_bench_v1_medium_validation.tsv`；step14000/17000 medium 明细在 `docs/training_progress/cropgenome_bench_v1_medium_validation/evaluations/step_00014000/` 和 `docs/training_progress/cropgenome_bench_v1_medium_validation/evaluations/step_00017000/`。所有结果仍是 diagnostic/validation-only（诊断/验证集选择）证据，不是 formal test（正式测试集）主结果。

## CropGenome-Bench v1 pilot smoke test (流程试跑，不是正式主结果)

为了启动正式 CropGenome-Bench v1（作物基因组正式下游基准评测），已经完成一个小规模 pilot smoke test（试点冒烟测试）：从现有 Stage_B region_bucket（区域桶标签）构建 3 个 proxy task（代理任务），用 step3000 checkpoint 在 RTX 2080 Ti 上抽 frozen encoder embedding（冻结编码器向量），并与 1-mer baseline（单碱基组成基线）比较。

重要口径：这一步只验证 pipeline（数据构建、固定 split、模型抽向量、baseline、metrics、图表输出）能跑通；标签来自 Stage_B 区域桶代理，不是最终 GFF 边界 hard-negative benchmark（硬负样本正式基准），不能写成正式论文主结果。

| Pilot task | Train samples | Eval samples | 1-mer F1 | Frozen embedding F1 | Δ F1 | 判断 |
|---|---:|---:|---:|---:|---:|---|
| TES_polyA | 256 | 128 | 0.6154 | 0.8467 | +0.2313 | embedding 更好 |
| promoter_TSS | 256 | 128 | 0.6290 | 0.7111 | +0.0821 | embedding 更好 |
| splice_acceptor | 256 | 128 | 0.5785 | 0.6043 | +0.0258 | embedding 更好 |

本次 pilot 结果说明：promoter/TSS 和 TES/polyA proxy 上 frozen embedding 明显超过 1-mer baseline；splice proxy 只小幅超过 1-mer。它支持继续完善 CropGenome-Bench v1 流程，但还不能证明正式下游任务成功。下一步应把 proxy 标签替换为 GFF-derived hard negatives（由 GFF 精确构建的硬负样本），并在 step4000/step5000 或 validation-best checkpoint 上跑固定 split + 多 seed。

公开轻量结果：`docs/training_progress/cropgenome_bench_v1_pilot_smoke/summary_cropgenome_bench_v1_pilot_smoke.tsv`；明细：`docs/training_progress/cropgenome_bench_v1_pilot_smoke/evaluations/step_00003000/source_data/pilot_metrics_summary.tsv`；图：`docs/training_progress/cropgenome_bench_v1_pilot_smoke/evaluations/step_00003000/figures/pilot_task_f1.png`。
