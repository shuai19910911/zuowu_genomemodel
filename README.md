# CropGenome-FM

更新时间：2026-07-31 08:46 CST

CropGenome-FM是面向作物基因组的长序列基础模型项目。本仓库只保留可公开、可读、可核验的轻量材料；checkpoint、embedding缓存、逐样本预测和大日志不上传。

## 当前状态

|模块|状态|最新事实|
|---|---|---|
|Stage B续训|RUNNING|step `25090/50000`；最新验证step25000；3×A100进程真实存活|
|step19000非EDTA下游|COMPLETE|A1–A10与B13–B17全部完成；10/10 embedding marker、2/2 evaluation marker、FINAL_RECEIPT闭包|
|EDTA结构注释|RUNNING|q04：42个运行、77个等待、本轮0个完成；RepeatModeler禁用|
|B11/B12|DEFERRED|等待正式EDTA TE标签，未使用替代标签补数|

2080Ti节点现在没有本轮下游程序是正常状态：非EDTA计算已完成，控制器exit=0，GPU已释放，而不是任务仍显示active但实际停跑。

## step19000非EDTA主结果

模型为`CropGenomeFM_step19000_continuation`，冻结编码器，只训练新线性/岭回归头。checkpoint只由validation selection loss选择；本轮test结果没有参与step19000的checkpoint选择。

|编号|任务|context|主指标|结果|
|---|---|---:|---|---:|
|A1|剪接供体/受体识别|8192|F1|0.7944 ± 0.0000|
|A2|启动子/TSS识别|8192|F1|0.7425 ± 0.0000|
|A3|TES/poly(A)识别|8192|F1|0.8114 ± 0.0000|
|A4|多物种lncRNA识别|6000|F1|0.7058 ± 0.0000|
|A5|木薯增强子识别|6000|F1|0.7696 ± 0.0000|
|A6|大豆基因表达预测|6000|macro Pearson|0.5761|
|A7|水稻基因表达预测|6000|macro Pearson|0.5065|
|A8|玉米基因表达预测|6000|macro Pearson|0.6508|
|A9|番茄基因表达预测|6000|macro Pearson|0.5912|
|A10|拟南芥基因表达预测|6000|macro Pearson|0.5652|
|B13|基因结构7态逐碱基分割|2048|macro-F1|0.3145 ± 0.0120|
|B14|长内含子剪接配对排序|4096|MRR|0.5441 ± 0.0000|
|B15|完整基因边界配对排序|4096|MRR|0.5425 ± 0.0000|
|B16|外显子顺序/同基因归属|2048|macro-F1|0.3851 ± 0.0000|

详细协议、全部context、逐seed源表、3张图、任务注册表和净化回执见：

- [step19000非EDTA下游详细结果](docs/downstream_step19000/README.md)

## 结果怎么理解

- A1/A2在2048 bp达到本轮最高F1，8192 bp没有继续提高；A3在8192 bp最好。不能把“更长context”统一写成必然更好。
- A6–A10从512 bp到6000 bp均明显提高，6000 bp macro Pearson为0.5065–0.6508，说明表达任务受益于更长局部上下文。
- B13类别识别有一定信号，但±20 bp boundary-F1仅约0.0115，边界定位仍弱。
- B14/B15排序和B16三分类只有有限信号，尚不能当作强结构注释结果。
- B17低同源筛选在当前test集保留了全部样本，因此与主track完全相同，不是独立鲁棒性证据。
- 本轮没有在同任务、同split、同head预算下重跑AgroNT、PlantCAD、NT-v2或Evo2，不能声称超过公开模型。

## 训练进展

- [训练状态、11个validation点和解释](TRAINING_PROGRESS.md)
- [总loss曲线](docs/training_progress/figures/stage_b_continuation_loss.png)
- [selection loss曲线](docs/training_progress/figures/stage_b_continuation_selection_loss.png)
- [曲线源数据](docs/training_progress/source_data/stage_b_continuation_metrics.tsv)

最新验证step25000：`val_selection_loss=1.0185342`，是当前最佳验证点；相对step15000的1.0402384下降约2.09%。训练仍在继续，不能把step25090称为最终模型。

## 其他材料

- [模型架构](MODEL_ARCHITECTURE.md)
- [正式few-shot结果](docs/results/formal_fewshot_metrics.tsv)
- [正式full-data结果](docs/results/formal_full_data_metrics.tsv)

## 公开边界

- 不上传checkpoint、optimizer/RNG状态、embedding、预测NPZ、逐seed模型头或大日志。
- `region_loss/region_acc`只作弱监督健康检查，不作为主checkpoint选择依据或正式下游胜利证据。
- A4–A10沿用公开数据原始split，不写成跨属迁移。
- B14/B15是候选组内排序，不等价于全基因组自动注释准确率。
- 后续若比较多个checkpoint的test结果来挑选模型，必须披露重复test查看带来的选择偏高风险。