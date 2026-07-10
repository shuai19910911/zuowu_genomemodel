# CropGenome-Bench v1：A100 正式下游评估详细解读

更新时间：2026-07-10 15:19 CST

这份文档回答四个问题：我们到底测了什么、结果好在哪里、哪些地方还不够好、是否可以进入 Stage C1 64K 训练。读者不需要机器学习背景；第一次出现的英文指标都会用白话解释。

## 1. 先说结论

1. **CropGenome-FM 的预训练确实学到了有用的作物基因组信息。**它在三个任务的平均 balanced accuracy（平衡准确率）上达到 `0.7386`（step14000）和 `0.7337`（step17000），高于 random-init（同结构但未预训练，`0.5967`）、DNABERT-2 117M（`0.6589`）和补充评估的 NT-v2 100M（`0.6813`）。
2. **最强证据是剪接位点识别。**step14000 的平衡准确率为 `0.8896`、AUROC 为 `0.9444`，明显高于 DNABERT-2 的 `0.7090/0.7862` 和 NT-v2 的 `0.7158/0.7948`。由于负样本也含有 GT/AG 典型剪接基序，这不是简单“看到 GT/AG 就猜正样本”。
3. **启动子任务有中等但稳定的优势。**step17000 的平衡准确率为 `0.6885`，比 DNABERT-2 高 `3.91` 个百分点，比 NT-v2 高 `1.95` 个百分点。
4. **TES/poly(A) 任务不能宣称全面领先。**step17000 为 `0.6455`，高于 DNABERT-2 和最佳 k-mer，但低于补充评估的 NT-v2 `0.6592`，落后 `1.37` 个百分点。这是当前明确的短板。
5. **两个 checkpoint（模型存档点）各有所长。**step14000 在剪接和三任务全量平均上更强；step17000 在启动子、TES/poly(A) 和 1% 标签平均上更强。因此不能写成“某一个 checkpoint 全任务最好”。
6. **Stage C1 64K 四项 gate（进入条件）通过。**真实 64K 执行、远距离依赖、按 token 归一化和固定验证选择均通过；A100 GPU2 峰值显存分配 `26,416.9 MiB`、预留 `27,946.0 MiB`，corrected 正式训练已启动。

## 2. 这次评估和以前的 proxy 有什么不同

以前的 formal-lite（轻量正式化评估）使用 Stage B proxy labels（代理标签），适合筛 checkpoint，但不能作为论文主结论。本次使用 raw GFF/GTF coordinates（原始结构注释坐标）和 FASTA 序列重新构建标签，不再使用 Stage B proxy 标签。

每个任务都有 `6,144` 个样本：

| split（数据划分） | 正样本 | 负样本 | 合计 | 用途 |
|---|---:|---:|---:|---|
| train（训练） | 2,048 | 2,048 | 4,096 | 训练简单线性分类头 |
| validation（验证） | 512 | 512 | 1,024 | 选择停止轮次和分类阈值 |
| test（正式测试） | 512 | 512 | 1,024 | 只用于最后报告 |

三个任务共 `18,432` 条序列。输入窗口统一为 `512 bp`（512 个碱基），所有方法使用完全相同的 train/validation/test 样本。

### 2.1 物种隔离，避免“见过同一种作物”造成虚高

- train：Brassica rapa（芸薹）、Prunus persica（桃）、Setaria italica（谷子）、Sorghum bicolor（高粱）、Vigna radiata（绿豆）、Vitis vinifera（葡萄）；剪接任务因可用注释结构不同使用其中 5 个物种。
- validation：Beta vulgaris（甜菜）、Daucus carota（胡萝卜）、Manihot esculenta（木薯）。
- test：Cucumis sativus（黄瓜）、Oryza sativa（水稻）、Solanum tuberosum（马铃薯）。

审计结果为 `species_disjoint=true`，即训练、验证和测试物种不重叠。这比随机拆分同一物种中的序列更难，也更接近“模型能不能迁移到没用于训练下游头的新作物”。

### 2.2 为什么叫 hard negatives（硬负样本）

普通负样本如果随便从基因组取一段，模型可能只靠 GC 含量或一个简单基序取巧。本次负样本更接近正样本：

- splice donor/acceptor（剪接供体/受体）：负样本同样含 canonical GT/AG motif（典型 GT/AG 基序），但位于已注释内含子内部并远离真实剪接边界。
- promoter/TSS（启动子/转录起始位点）：在同一 assembly（组装版本）中构建上游平移诱饵，并排除已注释基因边界。
- TES/poly(A)（转录终止/多聚腺苷酸化）：在同一 assembly 中构建下游平移诱饵，并排除已注释基因边界。

同时完成了标签平衡、坐标唯一性和 GC matching（GC 含量匹配）审计；各 task/split 的正负样本平均 GC 差最大约 `0.0011`。因此模型很难只靠“这段序列 GC 高不高”获得高分。

## 3. 比较了哪些方法

| 方法 | 小白解释 | 它回答的问题 |
|---|---|---|
| Best k-mer | 统计 1/3/6 个碱基组成，再训练线性分类器 | 不用大模型，仅靠短词频能做到什么程度？ |
| Random init | 使用 CropGenome-FM 相同结构，但权重没有预训练 | 提升来自模型结构，还是来自预训练学到的信息？ |
| DNABERT-2 117M | 公开通用 DNA 预训练模型 | 我们能否超过常用外部 DNA 模型？ |
| NT-v2 100M multi-species | 公开多物种 Nucleotide Transformer | 换一个更强的多物种模型后，结论是否还成立？ |
| CropGenome-FM step14000/17000 | 我们的两个预先保留候选 checkpoint | 哪个阶段的作物预训练表示更有用？ |

公平比较方式是 frozen embedding + linear probe（冻结模型向量 + 同一种线性分类头）：大模型本身不微调，只让每个模型把序列变成向量，再用同样简单的分类器判断正负。这样能尽量把差异归因于预训练表示，而不是“某个模型用了更复杂的下游头”。

说明：DNABERT-2、k-mer、random-init 和两个 CropGenome-FM checkpoint 属于读取正式 test 前锁定的主比较集合。NT-v2 是看到主表后补充的同口径外部模型，因此图表中用 `*` 标记，并作为 post-hoc supplementary（事后补充证据），不能反过来替换已锁定的主结果或用于继续挑 checkpoint。

## 4. 指标怎么读

- Balanced accuracy（平衡准确率）：正样本识别率和负样本识别率的平均值。`0.5` 接近随机猜，`1.0` 是完美。本次正负样本数量相同，所以它也很接近普通准确率，是主指标。
- MCC（Matthews correlation coefficient，马修斯相关系数）：综合考虑真阳性、真阴性、假阳性和假阴性。`0` 接近随机，`1` 是完美；比只看 accuracy 更不容易被类别比例误导。
- AUROC（受试者工作特征曲线下面积）：不固定某一个阈值，观察模型能否把正样本总体排在负样本前面。`0.5` 随机，越接近 `1` 越好。
- AUPRC（精确率-召回率曲线下面积）：更关注“找出的阳性有多准、真正阳性找回多少”。本数据正负平衡时随机参考约为 `0.5`。
- F1：precision（精确率）和 recall（召回率）的折中，受验证集所选阈值影响。它有用，但本报告不只靠 F1 下结论。
- SD（标准差）：不同随机种子结果的波动。数值越小，说明对少样本抽样越不敏感。

## 5. 全量标签主结果

下表是 100% train labels（使用全部 4,096 条训练样本）时的 balanced accuracy。粗体是每个任务的最高值。

| 任务 | Best k-mer | Random init | DNABERT-2 | NT-v2 100M* | step14000 | step17000 |
|---|---:|---:|---:|---:|---:|---:|
| promoter/TSS | 0.6113 | 0.5869 | 0.6494 | 0.6689 | 0.6875 | **0.6885** |
| splice donor/acceptor | 0.6797 | 0.6270 | 0.7090 | 0.7158 | **0.8896** | 0.8672 |
| TES/poly(A) | 0.6289 | 0.5762 | 0.6182 | **0.6592** | 0.6387 | 0.6455 |
| 三任务简单平均 | 0.6400 | 0.5967 | 0.6589 | 0.6813 | **0.7386** | 0.7337 |

![全量标签平衡准确率](figures/formal_full_data_balanced_accuracy.png)

### 5.1 promoter/TSS：有提升，但不是压倒性胜利

最佳 CropGenome-FM 是 step17000：

- balanced accuracy：`0.6885`
- MCC：`0.3999`
- AUROC：`0.7226`
- AUPRC：`0.7550`

相对 DNABERT-2 提升 `3.91` 个百分点，相对最佳 k-mer 提升 `7.71` 个百分点，相对 NT-v2 提升 `1.95` 个百分点。

白话解读：模型确实比短词频和通用 DNA 模型更会识别启动子上下文，但与 NT-v2 的差距不大，不能写成“遥遥领先”。启动子本身边界较分散，单个 512 bp 窗口也未必包含全部调控证据；后续仍需要开放染色质、表达或组织特异标签验证。

### 5.2 splice donor/acceptor：当前最可信的强项

最佳 CropGenome-FM 是 step14000：

- balanced accuracy：`0.8896`
- MCC：`0.7801`
- AUROC：`0.9444`
- AUPRC：`0.9513`

相对 DNABERT-2 提升 `18.07` 个百分点，相对最佳 k-mer 提升 `21.00` 个百分点，相对 NT-v2 提升 `17.38` 个百分点。

白话解读：如果负样本没有 GT/AG，模型可能只需记住剪接基序；但这里负样本也有相同的 GT/AG，因此高分更支持模型学到了基序周围的序列语法和作物剪接上下文。这是目前最适合放在论文主结果中的证据，但它仍是计算 benchmark，不等同于湿实验验证真实剪接事件。

### 5.3 TES/poly(A)：有信号，但外部模型更强

最佳 CropGenome-FM 是 step17000：

- balanced accuracy：`0.6455`
- MCC：`0.2984`
- AUROC：`0.6945`
- AUPRC：`0.7006`

它相对 DNABERT-2 提升 `2.73` 个百分点，相对最佳 k-mer 提升 `1.66` 个百分点；但比 NT-v2 的 balanced accuracy `0.6592` 低 `1.37` 个百分点。

白话解读：CropGenome-FM 学到了一些转录终止信号，但优势弱，而且不是所有外部模型中最好。这个任务当前应写成“有可检测信号、仍需改进”，不能写成“作物模型全面领先”。可能的后续方向包括更准确的 poly(A) 标签、转录组证据和更长下游上下文，但本次结果本身不能证明具体原因。

## 6. 少样本结果：只给很少标签时是否仍有用

1% 训练数据只约 `45–47` 条样本，10% 约 `414–415` 条。每个少样本设置使用 5 个 seed（随机种子）重新抽样，表中为三个任务的 balanced accuracy 简单平均。

| 标签比例 | DNABERT-2 | NT-v2 100M* | step14000 | step17000 |
|---|---:|---:|---:|---:|
| 1% | 0.5661 | 0.5663 | 0.6602 | **0.6669** |
| 10% | 0.6306 | 0.6225 | **0.7010** | 0.6876 |
| 100% | 0.6589 | 0.6813 | **0.7386** | 0.7337 |

![少样本标签效率](figures/formal_fewshot_balanced_accuracy.png)

重要观察：

- 1% 标签时，两个 CropGenome-FM checkpoint 都明显高于两个公开模型；step17000 相对最强公开模型高 `10.06` 个百分点。
- 10% 标签时，step14000 相对最强公开模型高 `7.04` 个百分点。
- 100% 标签时，step14000 相对最强公开模型高 `5.73` 个百分点。
- 优势在标签最少时更大，支持“作物预训练能降低下游标注需求”的方向。

边界：这里的 5 seeds 是线性 probe 的少样本子集抽样，不是 5 次独立预训练 CropGenome-FM。100% 设置下每个 seed 使用完全相同的数据和确定性优化路径，所以结果和 SD=0 完全一致；这不能包装成“5 个独立模型都稳定”。论文级模型稳定性仍需要独立预训练 seed 或至少独立微调 seed。

## 7. reverse complement（反向互补）结果怎么看

DNA 双链中，一段序列和它的 reverse complement（反向互补序列）代表同一遗传信息的两个方向。理想情况下，换方向后模型不应给出完全不同的判断。

- step14000、step17000 的平均 RC prediction agreement（反向互补预测一致率）均为 `1.000`，embedding cosine（向量余弦相似度）均为 `1.000`。
- DNABERT-2 的平均预测一致率为 `0.5869`，NT-v2 为 `0.5241`。
- 但 random-init 的两个 RC 指标也为 `1.000`。

因此，严格 RC 一致性主要证明 CropGenome-FM 的 RC-equivariant architecture（反向互补等变结构）实现正确，而不是单独证明预训练有效。预训练带来的价值要看它相对 random-init 的任务准确率提升；不能把架构自带的 `1.000` 当成学习效果。

## 8. 唯一 8K 最终版：early-stop step14000

如果只看不同下游任务，不能简单说一个全面胜出：

| 判断角度 | 更强候选 | 原因 |
|---|---|---|
| 全量三任务平均 | step14000 | `0.7386 > 0.7337`，主要由 splice 强优势推动 |
| splice | step14000 | `0.8896 > 0.8672` |
| promoter/TSS | step17000 | `0.6885 > 0.6875`，差距很小 |
| TES/poly(A) | step17000 | `0.6455 > 0.6387` |
| 1% 标签三任务平均 | step17000 | `0.6669 > 0.6602` |
| 10% 标签三任务平均 | step14000 | `0.7010 > 0.6876` |

但后续 64K/128K 训练必须有唯一初始化点。最终统一锁定 **early-stop step14000**：

1. 使用预先定义的 early-stopping 规则和它保存的 `checkpoint_best.pt`，checkpoint 元数据为 `step=14000`、`best_step=14000`、`best_selection_loss=1.074317`。
2. step17000 的原始 validation loss 后来略低，但改善没有达到 `min_delta=0.002` 的覆盖阈值；统一采用 early-stop checkpoint，避免在训练结束后反复改变阶段基座。
3. step14000 在正式评测的全量三任务平均和 splice 上更强，作为补充支持；但选择依据仍是预先定义的早停规则，不是 formal test 反选。
4. 论文结果继续报告两个 checkpoint，step17000 只作为敏感性对照；后续训练只使用 step14000。
5. 运营别名固定为 `checkpoint_stage_B_8k_final.pt → checkpoint_best.pt`，SHA-256=`c81bce39ec448845e929e755530bc7023a345cca42234ff7fb776f5f39c83fed`。

## 9. Stage C1 64K gate

A100 GPU2 上已完成真实 dry-run（单步验证）：

| 项目 | 结果 |
|---|---:|
| 输入 batch shape | `[1, 65536]` |
| checkpoint 加载 | 430 keys 全部匹配 |
| missing/unexpected keys | `0 / 0` |
| 初始化语义 | 严格继承模型权重；optimizer/global step/best tracking 重置 |
| 远距离依赖 | 128-chunk 等拓扑梯度支持由旧结构局部 992 个位置扩展至完整 8192/8192 |
| attention dilation | `1/2/4/8/16/32/64/128`，不新增参数，旧 checkpoint 严格兼容 |
| 目标归一化 | MLM/RC/region 分别按有效 token/标签数加权 |
| 固定验证面板 | 256 windows；22 assemblies、11 species、7 regions、76 个 64K |
| 总 loss | `0.717403` |
| MLM loss | `0.586707` |
| 峰值 allocated 显存 | `26,416.9 MiB` |
| 峰值 reserved 显存 | `27,946.0 MiB` |
| 前向/反向/optimizer step | 通过 |

Gate 结论：

- 工程执行、依赖跨度、目标归一化、模型选择：**4/4 PASS**。
- Stage B 表示是否有下游价值：**PASS**，尤其 splice 和少样本结果。
- 是否已经证明 64K 下游更有效：**没有**。梯度实验已证明 token 级依赖拓扑覆盖长程，但 64K 是否比 8K 更准仍需独立长程任务。
- Stage C1 正式训练：**RUNNING**。corrected 运行从唯一 8K 最终版 early-stop step14000 初始化，固定使用 A100 GPU2。

## 10. 当前不能过度声明什么

1. 只有 3 个 GFF-derived 二分类任务，尚未覆盖 TE boundary（转座元件边界）、长基因结构、variant effect（变异效应）和育种/QTL 任务。
2. 当前输入只有 512 bp，不能用这些结果证明 64K/128K 长上下文的优势。
3. 当前是 frozen embedding + linear probe，不等于 full fine-tuning（全模型微调）后的上限。
4. 只有一个 CropGenome-FM 预训练运行；少样本 5 seeds 不是独立预训练 seeds。
5. NT-v2 是读取主结果后补充的 post-hoc 外部模型；它可用于了解结果边界，但不能用于重新选择正式 checkpoint。
6. TES/poly(A) 未超过 NT-v2，作物模型“所有任务都优于公开模型”的说法不成立。
7. GFF/GTF 注释本身有物种间质量差异；正式论文还应补充注释质量敏感性分析。
8. 这些是计算评估结果，不替代转录组或湿实验功能验证。

## 11. 可复核文件

- [全量主指标表](source_data/headline_full_data_metrics.tsv)
- [1%/10%/100% 少样本表](source_data/fewshot_metrics.tsv)
- [逐任务相对提升](source_data/task_comparisons.tsv)
- [跨任务平均表](source_data/method_mean_balanced_accuracy.tsv)
- [Stage B 8K 最终 checkpoint 锁](source_data/stage_b_8k_final_checkpoint.json)
- [Stage C1 64K gate JSON](source_data/stage_c1_64k_gate.json)
- [轻量发布 manifest](source_data/publication_manifest.json)
- [全量对比 PNG](figures/formal_full_data_balanced_accuracy.png)
- [少样本 PNG](figures/formal_fewshot_balanced_accuracy.png)

GitHub 只发布聚合 TSV、JSON、PNG 和本说明，不上传原始 FASTA/GFF、逐样本预测、embedding cache、checkpoint 或训练日志。
