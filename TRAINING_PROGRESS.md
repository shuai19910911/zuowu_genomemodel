# CropGenome-FM训练进展

更新时间：2026-07-31 18:06 CST

## 1. 当前训练状态

|项目|当前值|
|---|---|
|运行状态|RUNNING|
|阶段|Stage B continuation；从step14000精确续训|
|最新训练点|step27590 / 50000|
|最新验证点|step27000|
|当前最佳验证点|step27000|
|best val selection loss|1.0156857|
|训练资源|gpu10，3×NVIDIA A100-SXM4-40GB，GPU0–2|
|实时核验|3个训练rank各约37.5GB；训练日志仍在增长|
|全局有效batch|36（每rank micro-batch 4 × 3 ranks × 梯度累积3）|
|采样|5,485,240个训练窗口；global no-replacement（全局无放回）|
|checkpoint|每500 step永久保存完整checkpoint|

本页是时间戳快照，训练继续后step会自然前进。训练曲线下降只说明预训练仍在改善，最终模型是否更有用仍要看固定下游任务。

## 2. Validation轨迹

`selection_loss = MLM loss + 0.02 × RC loss`，越低越好，是best checkpoint（最佳存档点）的主选择指标。

|step|val total loss|val MLM loss|val RC loss|val selection loss|region loss|region acc|
|---:|---:|---:|---:|---:|---:|---:|
|15000|1.1183|1.0371|0.1556|1.0402|1.5621|0.4323|
|16000|1.1177|1.0376|0.1497|1.0406|1.5407|0.4479|
|17000|1.1151|1.0363|0.1527|1.0394|1.5145|0.4531|
|18000|1.1120|1.0325|0.1662|1.0359|1.5238|0.4323|
|19000|1.1105|1.0299|0.1535|1.0330|1.5495|0.4115|
|20000|1.1094|1.0302|0.1589|1.0334|1.5204|0.4531|
|21000|1.1032|1.0257|0.1592|1.0289|1.4872|0.4375|
|22000|1.0978|1.0220|0.1585|1.0251|1.4531|0.4792|
|23000|1.0942|1.0196|0.1681|1.0230|1.4247|0.4688|
|24000|1.0936|1.0176|0.1614|1.0208|1.4548|0.5156|
|25000|1.0908|1.0153|0.1599|1.0185|1.4443|0.5052|
|26000|1.0887|1.0133|0.1629|1.0165|1.4422|0.4479|
|27000|1.0858|1.0123|0.1677|1.0157|1.4019|0.4896|

从step15000到27000，validation selection loss由1.0402384降至1.0156857，下降约2.36%。整体趋势继续改善，但中间并不是每个点都严格单调下降。

## 3. 曲线与源数据

- [总loss曲线](docs/training_progress/figures/stage_b_continuation_loss.png)
- [selection loss曲线](docs/training_progress/figures/stage_b_continuation_selection_loss.png)
- [全部训练/验证点TSV](docs/training_progress/source_data/stage_b_continuation_metrics.tsv)
- [曲线摘要TSV](docs/training_progress/source_data/stage_b_continuation_curve_summary.tsv)

曲线包含原始训练点、21点滚动中位数、全部validation checkpoint和最新训练点；图由TSV直接生成，只上传PNG，不上传SVG/PDF。

## 4. step19000完整非EDTA下游

step19000已完成A01–A10与B13–B17，共15个已执行编号；B11/B12等待EDTA标签。正式闭包包括：

- 10/10 embedding manifest；
- 2/2 evaluation run manifest；
- A组75个预测产物、B组全部5-seed产物；
- `FINAL_RECEIPT.json`状态ok；
- 控制器exit=0，2080Ti已释放。

主要结果概览：

- A01–A05分类F1为0.7058–0.8114；
- A06–A10多组织表达macro Pearson为0.5065–0.6508；
- B13基因结构七态macro-F1为0.3145 ± 0.0120；
- B14/B15排序MRR为0.5441/0.5425；
- B16三分类macro-F1为0.3851；
- B17当前低同源子集与主test相同，因此不是独立鲁棒性证据。

详细结果见[step19000非EDTA下游报告](docs/downstream_step19000/README.md)。

## 5. 下游v4进展

新的[下游任务通俗版总览](DOWNSTREAM_TASKS_CN.md)已经逐项解释全部56项任务，并明确哪些已经完成、哪些代码就绪、哪些等待许可证或EDTA。

|项目|当前进展|
|---|---|
|任务范围|A类10项＋B类7项＋C类36项＋D类3项，共56项；只含植物/作物|
|原有任务|原A+B共17项全部保留，没有被新增公开任务替换|
|已出step19000结果|A01–A10、B13–B17，共15个编号|
|EDTA阻塞|B11、B12|
|新增公开任务|39项来源、revision、split、指标和适配器已登记|
|许可证Gate|C17–C36、D01–D03共23项，正式主榜前必须人工复核|
|模型/基线|18个，包括CropGenome-FM、植物模型、通用DNA模型和k-mer基线|
|运行矩阵|56×18=1008行；每格均有明确状态|
|代码验证|v4专项47项通过；仓库全套314项＋20个子测试通过|

已完成的真实小规模链路：

- C03 FASTA、C06 CSV、C16逐碱基TSV和C17 Parquet均完成真实格式物化；
- CropGenome-FM、GPN-Brassicales、PlantBiMoE和k-mer在C06完成编码及统一probe；
- C17完成32条CropGenome-FM零样本motif恢复；
- 新launcher已完成“数据→编码→5-seed probe→最终回执”，第二次同参数执行可恢复而不重复计算。

这些是程序和接口验证，不是正式模型胜负。当前尚未在全部56项上完成18个模型的统一正式重跑。

## 6. EDTA与B11/B12

q04实时快照：2项EDTA完成、50项运行、67项等待；RepeatModeler明确禁用。B11/B12只在最终TE manifest、ID回映射和坐标质控全部通过后启动，不使用中间文件或替代标签。

## 7. 下一步

1. Stage B继续训练到step50000，每500步永久保存checkpoint、每1000步validation。
2. 持续完成EDTA；最终manifest关闭后运行B11/B12。
3. 人工复核23项公开数据许可证，未通过前不进入论文主榜。
4. 下载许可允许的公开模型，在同数据、同split、同上下文和同下游头预算下真实重跑。
5. 手动选择少量候选checkpoint做完整下游；不能用反复查看test的方式悄悄挑最优模型。

## 8. 解释边界

- 当前best只表示validation selection loss最低，不表示公开下游模型对比获胜。
- step19000下游test没有参与step19000 checkpoint选择；以后若据多个checkpoint的test结果选优，必须披露选择偏高。
- `region_loss/region_acc`只作weak-supervision（弱监督）健康检查，不进入主checkpoint选择。
- 冒烟测试只证明代码可运行，不能写成正式论文性能。
- 当前不能声称56项都已完成，也不能声称CropGenome-FM在所有任务上超过公开模型。
