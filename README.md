# CropGenome-FM

更新时间：2026-08-11 15:27 CST

CropGenome-FM是面向作物基因组的长序列基础模型项目。本仓库只保留可公开、可读、可核验的轻量材料；checkpoint、embedding缓存、逐样本预测和大日志不上传。

## 论文引导的下游协议

旧下游执行已停止；基于 Nature Communications 2025 DNA foundation model benchmark 与 GENEB 2026 的 25 项 plant-only 新协议、2080Ti 低内存实现和单 checkpoint 启动说明见 [`training_server_transfer/docs/ARTICLE_GUIDED_DOWNSTREAM_V1_CN.md`](training_server_transfer/docs/ARTICLE_GUIDED_DOWNSTREAM_V1_CN.md)。默认命令只生成候选计划，不占 GPU；只有显式 `--execute --confirmation ACTIVATE_ARTICLE_GUIDED_CHECKPOINT` 才会通过正式 Gate 启动 gpu05。

## 当前状态

|模块|状态|最新事实|
|---|---|---|
|Stage B续训|COMPLETE|已到step `50000/50000`；最终validation selection loss为0.9875776，当前Stage B最佳点为step50000|
|Stage C1混合长度训练|PAUSED / DATA REBUILD|已在step25942通过训练器安全中断并保存完整checkpoint；GPU0–2已释放。当前在SLURM q03 CPU节点重建Stage C1-v2语料，完成去重、近重复聚类和35池采样检查后再恢复训练|
|step19000非EDTA下游|COMPLETE|A1–A10与B13–B17全部完成；10/10 embedding marker、2/2 evaluation marker、FINAL_RECEIPT闭包|
|下游任务目录|56项|A类10项、B类7项、C类36项、D类3项；逐项说明见`DOWNSTREAM_TASKS_CN.md`|
|新增C/D数据准备|39/39|原始数据源11/11、统一数据整理39/39、公共模型14/14及真实GPU冒烟14/14完成|
|全量数据审计|COMPLETE|53/53非EDTA任务逐项检查通过；8个恢复分片及最终汇总均完成|
|三checkpoint完整下游|RUNNING / AUDIT BLOCKED|冻结step40000/45000/50000；快照时108/1572行和9/251个GPU组闭合，终止失败0；gpu05主机内存不足限制派发|
|EDTA结构注释|RUNNING/QUEUED|本轮119个目标已有98个满足完整输出与回执条件；12个在q04运行、7个排队；RepeatModeler禁用|
|B11/B12|DEFERRED|等待正式EDTA TE标签，未使用替代标签补数|

C/D新增任务的数据整理和完整性检查已经完成。CPU-only下游任务在gpu05本机CPU执行：调度daemon运行中，52行就绪，全部隐藏CUDA设备、按内存限流；SLURM q05提交已禁用。GPU端B4:00.0故障卡保持隔离，健康卡动态派发已验证，但主机内存不足使CPU/GPU新任务均等待；单个长期claim等待仍会限制后续派发。

这三个checkpoint都会访问完整下游测试集，因此其比较结果按monitoring/development evidence（监控/开发证据）管理，不能把其中表现最好的一个包装成未经test选择的独立最终模型。轻量状态快照见[`stage_b_checkpoint_set_downstream_status.tsv`](docs/training_progress/source_data/stage_b_checkpoint_set_downstream_status.tsv)。

### 全量数据审计结果

这不是模型训练或正式模型对比，而是最后一轮**数据完整性检查**。程序逐项读取53个非EDTA任务的`samples.tsv`，检查重复样本、同一序列跨train/validation/test泄漏、候选组跨split泄漏、数据划分是否齐全，并重新核对文件一致性。

原串行作业`8622446`在24小时上限到达后超时。恢复流程保留了18项有效结果，只把34项缺失任务分成8个q05分片补齐；8个分片全部`COMPLETED/0:0`。随后53/53逐任务深度检查、53/53正式数据索引和一次最终汇总均通过。当前三checkpoint评测直接复用这批冻结数据，不重新生成split。

项目已取消no-region、random-init及其他内部预训练/架构消融，不再为第二条完整预训练轨迹消耗计算资源。正式证据集中在作物专属下游任务，以及CropGenome-FM与14个公共模型和适用简单基线的同数据、同split公平比较。

所有可访问的下游数据和公共模型直接进入准备与正式评测，许可证元数据只保留为来源记录，不作为执行Gate，也不再产生需要人工确认的任务状态。

## 当前做到哪一步了

一句话概括：**Stage B已完成到step50000；Stage C1已安全暂停在step25942，正在重建去重后的Stage C1-v2语料；step40000/45000/50000三checkpoint完整非EDTA下游仍未完成。**

1. **Stage B**：step14000无替换续训已完成到step50000；最终`val_selection_loss=0.9875776`，相对正式候选起点step16000下降约5.10%。
2. **Stage C1**：从Stage B step50000权重warm-start并重新计步，单一连续run混合4K/8K/16K/32K/64K上下文。step25000后曾以exact模式继续；2026-08-11 15:24 CST按数据重建要求安全暂停在step25942，完整中断checkpoint包含模型、430项optimizer状态、sampler游标和RNG策略，GPU0–2已释放。旧语料存在坐标重复、15B配额裁剪和采样比例不一致，不能直接作为新数据流程的精确续训语料。
3. **下游与公共模型**：53/53数据检查通过；14/14公共模型smoke完成。
4. **三checkpoint比较**：step40000/45000/50000身份和SHA已冻结；54项任务、1572行、251个GPU组。快照时108行和9个GPU组闭合，终止失败0。14/14正式smoke完成，但审计仍受实现哈希漂移+矩阵未闭合阻塞。gpu05主机内存不足限制新任务派发。
5. **还差什么**：在q03完成Stage C1-v2坐标去重、正反向互补归一、100%序列去重、≥95% identity且双向覆盖≥90%的近重复聚类与降权；验证5种长度×7种区域的动态采样后，才进入可恢复训练状态。在此之前不能声称新语料已经READY，也不能声称超过公共强基线。

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

- [Stage B与Stage C1训练状态、曲线和validation轨迹](TRAINING_PROGRESS.md)
- [56项下游任务通俗版总览与当前进展](DOWNSTREAM_TASKS_CN.md)
- [Stage C1总loss曲线](docs/training_progress/figures/stage_c1_all_lengths_loss.png)
- [Stage C1 selection loss曲线](docs/training_progress/figures/stage_c1_all_lengths_selection_loss.png)
- [Stage C1曲线源数据](docs/training_progress/source_data/stage_c1_all_lengths_metrics.tsv)
- [Stage C1曲线摘要TSV](docs/training_progress/source_data/stage_c1_all_lengths_curve_summary.tsv)
- [完整Stage B总loss曲线：step10到当前](docs/training_progress/figures/stage_b_full_lineage_loss.png)
- [完整曲线源数据：step10到当前](docs/training_progress/source_data/stage_b_full_lineage_metrics.tsv)
- [step14000后续训总loss局部图](docs/training_progress/figures/stage_b_continuation_loss.png)
- [step14000后续训selection loss局部图](docs/training_progress/figures/stage_b_continuation_selection_loss.png)
- [续训局部图源数据](docs/training_progress/source_data/stage_b_continuation_metrics.tsv)

Stage C1独立图当前覆盖step10–25940，共2594个train点和51个validation点；安全中断发生在step25942。最新validation为step25500，`val_selection_loss=0.8854862`；当前最低仍是step25000的0.8844738。趋势总体改善，但预训练loss仍不能替代完整下游比较。

Stage B完整图覆盖step10–50000，共5000个train点和50个validation点；绿色虚线标出step14000精确续训边界。图中step10–14000来自原Stage B权威源表，step14000之后来自无替换续训日志；旧训练中已经被续训替代的step15000–17000没有混入新谱系。原有两张续训图继续保留，作为右半段细节放大图。Stage C1改变了上下文混合、运行配置和优化器状态，并重新计步，因此不把Stage B与Stage C1的loss硬连成一条曲线。

Stage B最终验证step50000：`val_selection_loss=0.9875776`，是Stage B范围内最佳验证点。Stage C1从该checkpoint做权重warm-start并重新计步。本次GitHub更新同步了Stage C1安全暂停和Stage C1-v2数据重建状态；详细数据合同见[`docs/STAGE_C1_V2_DATA_REBUILD_CN.md`](docs/STAGE_C1_V2_DATA_REBUILD_CN.md)。

## 其他材料

- [下游v4公开状态与注册表](docs/downstream_v4/README_CN.md)
- [模型架构](MODEL_ARCHITECTURE.md)
- [正式few-shot结果](docs/results/formal_fewshot_metrics.tsv)
- [正式full-data结果](docs/results/formal_full_data_metrics.tsv)

## 公开边界

- 不上传checkpoint、optimizer/RNG状态、embedding、预测NPZ、逐seed模型头或大日志。
- `region_loss/region_acc`只作弱监督健康检查，不作为主checkpoint选择依据或正式下游胜利证据。
- A4–A10沿用公开数据原始split，不写成跨属迁移。
- B14/B15是候选组内排序，不等价于全基因组自动注释准确率。
- 后续若比较多个checkpoint的test结果来挑选模型，必须披露重复test查看带来的选择偏高风险。