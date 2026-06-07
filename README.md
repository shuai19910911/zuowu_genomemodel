# zuowu_genomemodel

作物基因组预训练大模型项目。目标是基于本地已整理的作物/植物 genome FASTA 数据，直接训练正式的作物基因组基础模型，用于跨物种序列表示、调控元件识别、剪接/起始终止位点预测、基因表达相关序列建模、变异效应评分和作物改良候选位点优先级排序。

## 当前数据

数据说明文件: `/home/user/zhangzhishuai/data/plantDB/genome/README.md`

- 当前可用于纯序列预训练的 genome assembly: 1906 个。
- 覆盖属: 27 个。
- 带 genome 且至少有 GFF3/GTF 注释的完整 assembly: 262 个。
- 缺少注释但可用于自监督预训练的 assembly: 1644 个。
- genome gzip 总量约 898 GB: 完整注释集合约 213 GB，只有 genome 集合约 685 GB。
- GitHub 只保存项目介绍、计划、模型结构和进展，不上传基因组、注释、中间索引或训练产物。

## 核心方案

主模型采用长上下文、单碱基分辨率、反向互补等变的双向状态空间/Hyena 路线，而不是短窗口测试模型。正式训练分为两个阶段:

1. 基础预训练: 全部 1906 个 genome，单碱基 token，双链随机增强，序列长度从 8K 逐步扩到 64K/128K。
2. 长上下文继续预训练: 使用 chromosome/scaffold 连续窗口，扩到 256K，条件允许再扩到 512K 或 1M。

下游任务主要使用 262 个带注释 genome 和外部公开植物功能基因组数据构建:

- 剪接 donor/acceptor、TIS/TTS、polyA 位点。
- promoter/terminator 强度或分类。
- lncRNA/mRNA 分类。
- ATAC/open chromatin、组织特异表达相关序列预测。
- SNP/indel/SV 零样本或微调变异效应评分。
- 跨属留一评测和作物小样本迁移。

详细训练方案见 [PROJECT_PLAN.md](PROJECT_PLAN.md)；模型结构见 [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md)。

## 集群使用边界

- 当前环境是 Slurm 登录节点。
- CPU 任务由用户或后续脚本提交到 `q07` 或 `q08`，每个计算节点最多 30 核、150G 内存，每批最多 6 个命令，例如 `sbatch -p q07 -c 30 run.sh`。
- Python 环境使用 `mamba` 环境 `zuowu_genomemodel`。
- GPU 训练命令先只生成给用户执行，例如 `CUDA_VISIBLE_DEVICES=1,2 python train.py ...`；暂不由本会话直接启动 GPU 训练。

## 项目进展

- 2026-06-07 23:26:04 CST: 读取本地 plantDB genome 数据说明，确认 1906 个可预训练 genome、262 个带注释完整 genome；完成第一版文献调研和正式训练方案设计文档。

