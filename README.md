# zuowu_genomemodel

作物基因组预训练大模型项目。目标是基于本地已整理的作物/植物 genome FASTA 和结构注释数据，直接训练正式的作物基因组基础模型，用于跨物种序列表示、调控元件识别、剪接/起始终止位点预测、基因表达相关序列建模、变异效应评分和作物改良候选位点优先级排序。

## 当前数据

数据说明文件: `/home/user/zhangzhishuai/data/plantDB/genome/README.md`

- 当前正式训练只使用带 genome 且至少有 GFF3/GTF 注释的完整 assembly: 262 个。
- 覆盖属: 26 个。
- 放弃缺少结构注释的 assembly: 1644 个，暂不进入预训练。
- 262 个正式数据的 genome gzip 约 213.48 GB，GFF/GTF gzip 合计约 5.40 GB，总压缩体积约 218.89 GB。
- GitHub 只保存项目介绍、计划、模型结构和进展，不上传基因组、注释、中间索引或训练产物。

## 核心方案

主模型采用结构注释感知、区域加权、长上下文、单碱基分辨率、反向互补等变的 Mamba2/Hyena + periodic attention 路线，而不是短窗口测试模型。正式训练重点:

1. 预处理只使用 262 个结构注释完整基因组，构建 CDS、splice、UTR、promoter/TSS、TES、intron、high-quality intergenic 区域。
2. 训练输入不全量采样所有序列，而是按区域比例和质量过滤进入 batch；high-quality intergenic 目标比例为 10%。
3. 模型输入包含 `input_ids`、`labels_mlm`、`loss_mask`、`region_ids`、`region_weights` 和坐标 metadata。
4. 训练上下文从 8K 扩展到 64K/128K，资源允许再到 256K。

下游任务主要使用 262 个带注释 genome 和外部公开植物功能基因组数据构建:

- 剪接 donor/acceptor、TIS/TTS、polyA 位点。
- promoter/terminator 强度或分类。
- lncRNA/mRNA 分类。
- ATAC/open chromatin、组织特异表达相关序列预测。
- SNP/indel/SV 零样本或微调变异效应评分。
- 跨属留一评测和作物小样本迁移。

详细训练方案见 [PROJECT_PLAN.md](PROJECT_PLAN.md)，其中包含数据过滤、严格防泄漏 split、区域加权采样、模型输入、loss、训练资源、搬运数据体积、下游任务和评测结果记录表；模型结构见 [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md)；技术路线图见 [TECHNICAL_ROADMAP.md](TECHNICAL_ROADMAP.md)。

## 集群使用边界

- 当前环境是 Slurm 登录节点。
- CPU 任务由用户或后续脚本提交到 `q07` 或 `q08`，每个计算节点最多 30 核、150G 内存，每批最多 6 个命令，例如 `sbatch -p q07 -c 30 run.sh`。
- Python 环境使用 `mamba` 环境 `zuowu_genomemodel`。
- GPU 训练命令先只生成给用户执行，例如 `CUDA_VISIBLE_DEVICES=1,2 python train.py ...`；暂不由本会话直接启动 GPU 训练。

## 项目进展

- 2026-06-07 23:26:04 CST: 读取本地 plantDB genome 数据说明，确认 1906 个可预训练 genome、262 个带注释完整 genome；完成第一版文献调研和正式训练方案设计文档。
- 2026-06-07 23:46:31 CST: 扩展训练方案和模型结构文档，加入端到端预处理、模型输入张量、前沿架构对比、分阶段 CPU/GPU 资源需求和总体训练耗时估算。
- 2026-06-08 11:42:31 CST: 根据新要求改为只使用 262 个结构注释完整基因组，放弃无注释 genome；补充区域加权、过滤标准、严格防泄漏 split、下游任务、基线优势预期、跨服务器搬运磁盘估算，并新增技术路线图。
- 2026-06-08 12:33:50 CST: 将 `assets/cropgenome_fm_roadmap.svg` 从简约概览图扩展为详细技术路线图，覆盖数据口径、QC、区域构建、采样权重、防泄漏 split、token shard、模型输入、架构、loss、训练资源、下游任务和基线优势。
- 2026-06-08 15:00:51 CST: 根据用户确认，放弃进一步压缩到核心 assembly 的方案；整理最终完整训练计划，采用 262 个结构注释完整基因组、原始压缩数据搬运、小索引、在线采样/tokenization 和 100-200GB 磁盘缓存。
- 2026-06-08 18:16:18 CST: 在训练计划中新增评测结果记录表和更新规则，后续预训练、下游任务和基线比较结果都写回 `PROJECT_PLAN.md` 的评测结果章节。
