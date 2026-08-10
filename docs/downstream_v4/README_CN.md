# CropGenome-FM 下游 v4：公开状态说明

更新时间：2026-08-10 15:57 CST

本页提供下游 v4 的轻量公开状态。仓库只保存可阅读的任务、来源、模型注册表及进展快照，不上传checkpoint、公共模型权重、数据集、embedding、逐样本预测、日志或完整运行目录。

## 当前正式范围

- 注册任务：56项植物/作物任务，A类10项、B类7项、C类36项、D类3项。
- 当前可执行：54项非EDTA任务；B11、B12等待正式EDTA标签和坐标质控。
- 独立数据审计：53/53通过；B17复用B13–B16结果做敏感性分析，不单独生成数据集。
- 模型与基线：CropGenome-FM、14个公开DNA模型和k-mer简单基线，共16个。
- 当前checkpoint集合：Stage B step40000、45000、50000。
- 冻结矩阵：1572行、251个GPU组；每项使用5个随机种子；内部预训练消融不进入本轮。

2026-08-10 15:57 CST快照：107/1572行、9/251个GPU组闭合，终止失败0。CPU-only下游任务在gpu05本机CPU执行：调度daemon运行中，1个worker活跃、52行就绪，全部隐藏CUDA设备、按内存限流；SLURM q05提交已禁用。GPU端健康卡动态派发已经恢复，4个组持有显存/主机内存claim并计算，另1个组等待主机内存。

最终审计当前为`BLOCKED`：14/14公共模型正式smoke回执和执行授权已经完成，剩余blocker是实现哈希漂移与矩阵未闭合。gpu05的B4:00.0故障卡保持隔离，NVML可见其余7/8块卡；健康卡的新CUDA上下文和连续动态派发已经通过实跑验证。当前GPU embedding提取占用大量主机内存，CPU-only新行和一个额外GPU组等待内存；单个长期claim等待仍会限制后续派发。

轻量状态：`docs/training_progress/source_data/stage_b_checkpoint_set_downstream_status.tsv`

## 可复核注册表

- `docs/cropgenome_downstream_v4/task_registry.tsv`：56项任务的任务类型、来源、split、指标和上下文；
- `docs/cropgenome_downstream_v4/model_registry.tsv`：16个模型/基线及冻结revision；
- `docs/cropgenome_downstream_v4/source_registry.tsv`：13个来源及引用、版本和许可证元数据。

## 证据边界

三个checkpoint都会访问完整test，因此其比较结果属于checkpoint-comparison monitoring/development evidence（checkpoint比较型监控/开发证据），不能把事后表现最好的checkpoint写成未经test选择的独立最终模型。

冒烟测试只证明模型可加载、可前向和目标上下文可运行，不代表性能。`not_applicable`表示模型能力与任务接口不匹配，不能记成0分。B11/B12在EDTA最终manifest和坐标质控通过前保持阻塞，不使用临时替代标签。许可证只作为来源元数据保存，不形成执行Gate；所有可访问的数据和权重按同一冻结协议真实重跑。

## 为什么不上传完整执行目录

正式运行绑定本地冻结数据、模型权重、checkpoint身份、环境收据和机器调度回执。只上传部分源码而不附这些依赖，会让公开仓库看似可运行、实际无法闭环。因此GitHub保持轻量；完整执行产物在本地审计闭合后，只汇总公开主结果、源数据表和必要图件。
