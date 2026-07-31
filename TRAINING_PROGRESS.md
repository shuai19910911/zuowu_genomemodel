# CropGenome-FM训练进展

更新时间：2026-07-31 08:46 CST

## 1. 当前训练状态

|项目|当前值|
|---|---|
|运行状态|RUNNING|
|阶段|Stage B continuation；从step14000精确续训|
|最新训练点|step25090 / 50000|
|最新验证点|step25000|
|当前最佳验证点|step25000|
|best val selection loss|1.0185342|
|训练资源|gpu10，3×NVIDIA A100-SXM4-40GB，GPU0–2|
|实时核验|3个训练rank各约37.5GB，GPU利用率100%|
|全局有效batch|36（每rank micro-batch 4 × 3 ranks × 梯度累积3）|
|采样|5,485,240个训练窗口；global no-replacement|
|checkpoint|每500 step永久保存完整checkpoint|

训练进程、rank和显存均已实时核验；本页是时间戳快照，训练继续后step会自然前进。

## 2. Validation轨迹

`selection_loss = MLM loss + 0.02 × RC loss`，越低越好，是best checkpoint与early stopping的唯一主选择指标。

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

从step15000到25000，validation selection loss由1.0402384降至1.0185342，下降约2.09%；整体持续改善，但不是每个1000-step点都单调下降。

## 3. 曲线与源数据

- [总loss曲线](docs/training_progress/figures/stage_b_continuation_loss.png)
- [selection loss曲线](docs/training_progress/figures/stage_b_continuation_selection_loss.png)
- [全部训练/验证点TSV](docs/training_progress/source_data/stage_b_continuation_metrics.tsv)
- [曲线摘要TSV](docs/training_progress/source_data/stage_b_continuation_curve_summary.tsv)

曲线包含原始训练点、21点滚动中位数、全部validation checkpoint和最新训练点；图由TSV直接生成，只上传PNG，不上传SVG/PDF。

## 4. step19000完整非EDTA下游

step19000已完成A1–A10与B13–B17，共15个已执行编号；B11/B12等待EDTA标签。正式闭包包括：

- 10/10 embedding manifest；
- 2/2 evaluation run manifest；
- A组75个预测产物、B组全部5-seed产物；
- `FINAL_RECEIPT.json`状态ok；
- 21项相关单元/端到端测试通过；
- 控制器exit=0，2080Ti已释放。

详细结果见[step19000非EDTA下游报告](docs/downstream_step19000/README.md)。

## 5. EDTA与B11/B12

q04实时快照：42个EDTA运行、77个等待、本轮0个完成。普通任务使用10–12核/60GB；7个4.23–4.59GB超大输入使用12核/160GB；RepeatModeler明确禁用。B11/B12只在正式TE标签闭包后启动，不使用替代标签。

## 6. 解释边界

- 当前最佳只表示validation selection loss最低，不表示公开下游模型对比获胜。
- step19000下游test结果不参与step19000 checkpoint选择；后续若据多个checkpoint test结果选优，必须披露选择偏高。
- `region_loss/region_acc`仅作weak-supervision健康检查，不进入主选择指标。
- 训练目标仍是step50000；step25090只是中间快照，不称最终模型。