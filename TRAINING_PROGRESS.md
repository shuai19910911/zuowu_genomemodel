# CropGenome-FM训练进展

更新时间：2026-08-04 21:28 CST

## 1. 当前训练状态

|项目|当前值|
|---|---|
|Stage B|COMPLETE：step14000精确续训至step50000|
|Stage B最终/最佳验证点|step50000；val selection loss 0.9875776|
|Stage C1|RUNNING：从Stage B step50000权重warm-start，阶段从step0重新计步|
|Stage C1最新训练日志|step1520 / 30000|
|Stage C1最新validation|step1500；val selection loss 0.9056264，当前Stage C1最低|
|Stage C1上下文|4K/8K/16K/32K/64K按预设比例混合的单一连续run|
|Stage C1曲线范围|step10–1520；152个train点、3个validation点|
|训练资源|gpu10，3×NVIDIA A100-SXM4-40GB，GPU0–2|
|Stage B checkpoint|每500 step永久保存；step40000、45000、50000身份与SHA已冻结用于完整下游|
|Stage B完整曲线范围|step10–50000；5000个train点、50个validation点|
|三checkpoint下游|RUNNING：54项可执行非EDTA任务、1572行、251个GPU组；快照时22行和2组闭合，终止失败0|

本页是时间戳快照，Stage C1和三checkpoint下游继续运行后数字会自然前进。Stage B预训练曲线下降只说明训练目标改善，最终模型是否更有用仍要看固定下游任务和公共模型公平重跑。

## 2. Stage B validation轨迹

`selection_loss = MLM loss + 0.02 × RC loss`，越低越好，是best checkpoint（最佳存档点）的主选择指标。下表聚焦step14000精确续训后的validation；step1000–14000的历史validation仍完整保留在全程曲线源数据中。

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
|28000|1.0834|1.0099|0.1701|1.0133|1.4013|0.4948|
|29000|1.0818|1.0086|0.1711|1.0120|1.3964|0.4948|
|30000|1.0805|1.0071|0.1767|1.0106|1.3988|0.4792|
|31000|1.0750|1.0037|0.1725|1.0072|1.3565|0.5052|
|32000|1.0733|1.0022|0.1671|1.0056|1.3556|0.5208|
|33000|1.0701|1.0006|0.1718|1.0041|1.3212|0.5260|
|34000|1.0685|0.9992|0.1715|1.0027|1.3170|0.5208|
|35000|1.0684|0.9977|0.1737|1.0011|1.3445|0.5104|
|36000|1.0673|0.9962|0.1725|0.9997|1.3537|0.5260|
|37000|1.0650|0.9949|0.1699|0.9983|1.3334|0.5365|
|38000|1.0641|0.9939|0.1731|0.9974|1.3346|0.4896|
|39000|1.0614|0.9923|0.1736|0.9958|1.3133|0.5365|
|40000|1.0622|0.9916|0.1742|0.9951|1.3405|0.5260|
|41000|1.0573|0.9903|0.1717|0.9938|1.2709|0.5104|
|42000|1.0572|0.9892|0.1744|0.9927|1.2913|0.5156|
|43000|1.0555|0.9885|0.1718|0.9919|1.2719|0.5625|
|44000|1.0558|0.9877|0.1741|0.9912|1.2914|0.5208|
|45000|1.0544|0.9867|0.1739|0.9902|1.2845|0.5312|
|46000|1.0538|0.9857|0.1731|0.9892|1.2919|0.5469|
|47000|1.0526|0.9852|0.1745|0.9887|1.2783|0.5573|
|48000|1.0522|0.9847|0.1761|0.9882|1.2792|0.5729|
|49000|1.0522|0.9843|0.1757|0.9878|1.2876|0.5417|
|50000|1.0517|0.9841|0.1752|0.9876|1.2817|0.5521|

从正式候选起点step16000到step50000，validation selection loss由1.0406386降至0.9875776，下降0.0530609（约5.10%）。总体趋势继续改善，但total loss和辅助region指标并非每个点都严格单调；step50000只是Stage B范围内最佳点，最终模型价值仍必须由固定下游任务验证。

## 3. 曲线与源数据

### Stage C1混合长度训练

- [Stage C1总loss曲线](docs/training_progress/figures/stage_c1_all_lengths_loss.png)
- [Stage C1 selection loss曲线](docs/training_progress/figures/stage_c1_all_lengths_selection_loss.png)
- [Stage C1源数据](docs/training_progress/source_data/stage_c1_all_lengths_metrics.tsv)
- [Stage C1曲线摘要TSV](docs/training_progress/source_data/stage_c1_all_lengths_curve_summary.tsv)

|Stage C1 step|val total loss|val MLM loss|val RC loss|val selection loss|region loss|region acc|
|---:|---:|---:|---:|---:|---:|---:|
|500|0.9746|0.9023|0.2095|0.9065|1.3624|0.5365|
|1000|0.9700|0.9016|0.2084|0.9058|1.2840|0.5417|
|1500|0.9667|0.9015|0.2073|0.9056|1.2223|0.5729|

从step500到step1500，validation selection loss下降0.0008364（约0.09%），total loss下降约0.80%。方向略有改善，但幅度很小且目前只有3个validation点，不能据此声称长上下文阶段已经带来明确模型增益。训练raw loss会受4K–64K混合长度与每步样本组成影响，判断趋势应优先看滚动中位数和固定validation。

Stage C1是新阶段：它从Stage B step50000加载模型权重，但重新建立优化器、混合长度采样和阶段步数。因此Stage C1单独画图，不把它与Stage B的loss硬连接成一条连续曲线。

### Stage B完整谱系与续训局部图

- [完整Stage B总loss曲线：step10–50000](docs/training_progress/figures/stage_b_full_lineage_loss.png)
- [完整Stage B源数据：5000个train点＋50个validation点](docs/training_progress/source_data/stage_b_full_lineage_metrics.tsv)
- [完整曲线摘要TSV](docs/training_progress/source_data/stage_b_full_lineage_curve_summary.tsv)
- [step14000后续训总loss局部图](docs/training_progress/figures/stage_b_continuation_loss.png)
- [step14000后续训selection loss局部图](docs/training_progress/figures/stage_b_continuation_selection_loss.png)
- [续训局部图源数据](docs/training_progress/source_data/stage_b_continuation_metrics.tsv)
- [续训局部图摘要TSV](docs/training_progress/source_data/stage_b_continuation_curve_summary.tsv)

此前GitHub只展示了当前续训日志，所以横轴从step14000之后开始，看起来像训练曲线缺了一段。完整图以原Stage B权威TSV只保留step10–14000，再从精确resume边界接入当前无替换续训日志；续训替代掉的旧step15000–17000不会混入新谱系。绿色虚线表示step14000精确续训边界。所有50个validation标记都显示，但只标注10个关键点，避免文字遮挡。

Stage B和Stage C1图都包含原始训练点、21点滚动中位数、validation checkpoint和最新训练点；图由各自TSV直接生成，只上传PNG，不上传SVG/PDF。现有Stage B续训图不删除，继续用于观察step14000之后的细节。

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

新的[下游任务通俗版总览](DOWNSTREAM_TASKS_CN.md)已经逐项解释全部56项任务，并明确哪些已经完成、哪些正在排队、哪些等待EDTA。

|项目|当前进展|
|---|---|
|任务范围|A类10项＋B类7项＋C类36项＋D类3项，共56项；只含植物/作物|
|原有任务|原A+B共17项全部保留，没有被新增公开任务替换|
|已出step19000结果|A01–A10、B13–B17，共15个编号|
|EDTA阻塞|B11、B12|
|新增公开任务|C/D共39项，已全部整理完成并生成统一数据回执|
|原始资产|11/11数据源已下载校验；14/14公共模型已下载并通过真实GPU前向冒烟|
|资源使用策略|所有可访问数据和公共模型直接使用；许可证元数据不构成执行Gate|
|模型/基线|16个：CropGenome-FM、14个公共模型和k-mer简单基线；不含内部消融|
|三checkpoint正式运行|step40000/45000/50000；1572行、251个GPU组、5个seed；2026-08-04 19:46 CST快照为22行和2组闭合，终止失败0|
|统一数据审计|COMPLETE：53/53非EDTA任务逐项检查通过，数据索引与汇总检查均已关闭|
|CPU调度边界|只允许SLURM q02–q05；禁止cu和fat，无资源时等待|

当前已经完成的资产和真实链路：

- 11/11正式数据源完成下载与校验；
- 14/14公共模型完成下载，并全部通过真实GPU前向冒烟；
- C01–C36和D01–D03共39项已经全部转换为统一数据格式并生成回执；
- 最后8个SLURM数据作业全部以`COMPLETED/0:0`结束；
- 正式非EDTA数据索引达到53/53 ready，53项逐任务完整性检查全部通过；
- 14/14公共模型真实GPU前向冒烟全部通过。

原串行作业8622446在24小时上限到达后超时，已通过保留18项有效结果、只分片补齐缺失任务的方式恢复。8个q05分片全部`COMPLETED/0:0`，随后完成53/53逐任务深度核验和一次汇总检查；没有从头重复已通过项目。当前数据准备已经关闭，三checkpoint评测正直接复用冻结数据和split运行。轻量状态快照见[`stage_b_checkpoint_set_downstream_status.tsv`](docs/training_progress/source_data/stage_b_checkpoint_set_downstream_status.tsv)。

当前已进入正式执行阶段，但完整矩阵尚未闭合，因此仍没有模型胜负结论。step40000、45000、50000都会访问完整test；这些结果按checkpoint-comparison monitoring/development evidence管理，不能把事后最优checkpoint写成未经test选择的独立最终模型。

## 6. EDTA与B11/B12

EDTA继续在q04执行，RepeatModeler明确禁用；119个本轮待处理组装中已有17个最终任务回执通过，最终汇总manifest尚未形成。B11/B12只在最终TE manifest、ID回映射和坐标质控全部通过后启动，不使用中间文件或替代标签。

## 7. 下一步

1. Stage C1继续按4K/8K/16K/32K/64K混合长度单一连续run训练，并按冻结门禁监控validation。
2. 持续执行step40000、45000、50000三checkpoint的1572行完整非EDTA矩阵，自动接力GPU embedding、CPU probe、B17敏感性分析和最终审计。
3. 持续完成EDTA；最终manifest、ID回映射和坐标质控关闭后运行B11/B12。
4. 不运行no-region、random-init或其他内部模型消融，把计算资源集中到作物专属下游任务和公共强模型公平比较。
5. 矩阵闭合后统一汇总三个checkpoint、公共模型和简单基线；如据test表现选择checkpoint，必须明确标注post-hoc（事后选择）和监控/开发证据属性。

## 8. 解释边界

- 当前best只表示validation selection loss最低，不表示公开下游模型对比获胜。
- step19000下游test没有参与step19000 checkpoint选择；以后若据多个checkpoint的test结果选优，必须披露选择偏高。
- `region_loss/region_acc`只作weak-supervision（弱监督）健康检查，不进入主checkpoint选择。
- 冒烟测试只证明代码可运行，不能写成正式论文性能。
- 当前不能声称56项都已完成，也不能声称CropGenome-FM在所有任务上超过公开模型。
