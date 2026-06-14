# zuowu_genomemodel

作物基因组预训练大模型项目。目标是基于本地已整理的作物/植物 genome FASTA 和结构注释数据，直接训练正式的作物基因组基础模型，用于跨物种序列表示、调控元件识别、剪接/起始终止位点预测、基因表达相关序列建模、变异效应评分和作物改良候选位点优先级排序。

## 当前数据

数据说明文件: `/home/user/zhangzhishuai/data/plantDB/genome/README.md`

- 当前正式训练只使用带 genome 且至少有 GFF3/GTF 注释的 crop assembly: 263 行 manifest，去重后 258 个 canonical assembly accession。
- 覆盖属: 26 个。
- 放弃缺少结构注释的 assembly: 1644 个，暂不进入预训练。
- 263 个正式数据的 genome gzip 约 214.11 GB，GFF/GTF gzip 合计约 2.89 GB，总压缩体积约 216.99 GB。
- GitHub 只保存项目介绍、计划、模型结构和进展，不上传基因组、注释、中间索引或训练产物。

## 核心方案

主模型采用结构注释感知、区域加权、长上下文、单碱基分辨率、反向互补等变的 Mamba2/Hyena + periodic attention 路线，而不是短窗口测试模型。正式训练重点:

1. 预处理只使用 263 行结构注释完整作物 assembly manifest，按 258 个 canonical assembly accession 防泄漏 split，构建 CDS、splice、UTR、promoter/TSS、TES、gene body、high-quality background 区域；TE/repeat、端粒、着丝粒、satellite/tandem repeat、rDNA、organellar insertion、segmental duplication 等结构基因组信息只在有可靠证据时启用。
2. 本服务器完成数据处理、候选池、split、防泄漏检查，并按 Stage B/C1/C2/D 固化每个 stage 的训练输入窗口或 `input_ids`。
3. 训练服务器只接收 `training_server_transfer/` 目录中的当前 stage 数据、配置和必要索引；训练时动态生成 mask、MLM labels、RC 增强和 batch 顺序。
4. 训练上下文从 8K 扩展到 64K/128K，资源允许再到 256K。

下游任务主要使用当前带注释 crop genome 和外部公开植物功能基因组数据构建:

- 剪接 donor/acceptor、TIS/TTS、polyA 位点。
- promoter/terminator 强度或分类。
- lncRNA/mRNA 分类。
- ATAC/open chromatin、组织特异表达相关序列预测。
- SNP/indel/SV 零样本或微调变异效应评分。
- TE family、TE insertion boundary、telomere/subtelomere、centromere/pericentromere、satellite repeat 等结构基因组任务。
- 跨属留一评测和作物小样本迁移。

详细训练方案见 [PROJECT_PLAN.md](PROJECT_PLAN.md)，其中包含数据过滤、严格防泄漏 split、区域加权采样、模型输入、loss、训练资源、搬运数据体积、下游任务和评测结果记录表；结构基因组自注释可行性见 [STRUCTURAL_ANNOTATION_FEASIBILITY.md](STRUCTURAL_ANNOTATION_FEASIBILITY.md)；模型结构见 [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md)；技术路线图见 [TECHNICAL_ROADMAP.md](TECHNICAL_ROADMAP.md)。
训练服务器搬运目录结构和每个文件用途见 [TRANSFER_DIRECTORY_STRUCTURE.md](TRANSFER_DIRECTORY_STRUCTURE.md)。

## 集群使用边界

- 当前环境是 Slurm 登录节点。
- CPU 任务由用户或后续脚本提交到 `q07` 或 `q08`，每个计算节点最多 30 核、150G 内存，每批最多 6 个命令，例如 `sbatch -p q07 -c 30 run.sh`。
- Python 环境使用 `mamba` 环境 `zuowu_genomemodel`。
- GPU 训练命令先只生成给用户执行，例如 `CUDA_VISIBLE_DEVICES=1,2 python train.py ...`；暂不由本会话直接启动 GPU 训练。

## 执行规则

- 2026-06-14 22:39:46 CST 起，凡是本项目中新生成或修改的训练、数据处理、注释、搬运脚本，正式运行前必须先完成代码逻辑核查、语法检查和最小可行运行检查；若任一检查失败，必须先修正并重新检查，不得直接启动正式任务。

## 项目进展

- 2026-06-07 23:26:04 CST: 读取本地 plantDB genome 数据说明，确认第一版可用数据范围；完成第一版文献调研和正式训练方案设计文档。
- 2026-06-07 23:46:31 CST: 扩展训练方案和模型结构文档，加入端到端预处理、模型输入张量、前沿架构对比、分阶段 CPU/GPU 资源需求和总体训练耗时估算。
- 2026-06-08 11:42:31 CST: 根据新要求改为只使用结构注释完整 crop genome，放弃无注释 genome；补充区域加权、过滤标准、严格防泄漏 split、下游任务、基线优势预期、跨服务器搬运磁盘估算，并新增技术路线图。
- 2026-06-08 12:33:50 CST: 将 `assets/cropgenome_fm_roadmap.svg` 从简约概览图扩展为详细技术路线图，覆盖数据口径、QC、区域构建、采样权重、防泄漏 split、token shard、模型输入、架构、loss、训练资源、下游任务和基线优势。
- 2026-06-08 15:00:51 CST: 根据用户确认，放弃进一步压缩到核心 assembly 的方案；整理最终完整训练计划，采用结构注释完整 crop genome 和跨服务器训练策略。
- 2026-06-08 18:16:18 CST: 在训练计划中新增评测结果记录表和更新规则，后续预训练、下游任务和基线比较结果都写回 `PROJECT_PLAN.md` 的评测结果章节。
- 2026-06-08 18:45:35 CST: 参考 `douke_genome` 的区域采样方案，更新为 TE/repeat 注释模式和当前默认无 TE fallback 模式。
- 2026-06-08 22:19:54 CST: 按用户当前确认的方案更新训练计划: 本服务器固化每个 stage 的输入，训练服务器接收 `training_server_transfer/` 并动态生成 mask/label/RC。
- 2026-06-08 22:43:02 CST: 开始 CPU 数据处理；因 genome 目录持续下载且可能混入非作物，manifest 阶段加入作物属白名单，今晚实际处理 263 个 crop assembly、26 个属；已提交 cu 分区 FASTA QC、annotation QC 和依赖合并任务。
- 2026-06-09 08:48:28 CST: FASTA QC 和 annotation QC 均完成 263/263，已在 q07 完成合并；继续按用户要求切换到 q07/q08，已在 q08 提交 region/sampling candidate 构建任务 `8469374`。
- 2026-06-09 10:28:40 CST: q08 region/sampling candidate 构建完成，q07 split/transfer manifest 和 SHA256 校验完成；生成第一版基础传输包。
- 2026-06-09 12:52:00 CST: 新增 `TRANSFER_DIRECTORY_STRUCTURE.md`，详细说明训练服务器只需整体搬运 `training_server_transfer/`，并解释最终目录下每个文件的用途、当前已完成内容和仍需生成的 Stage input。
- 2026-06-09 15:14:03 CST: 完成 263 个 crop assembly 的 seqid alias 修正和新版 region candidates 重建；`features_missing_contig` 从 23,713,267 降到 3,581,382；基础传输包 SHA256 校验通过。
- 2026-06-09 15:45:00 CST: 修正为 accession-level canonical split，263 行 manifest 折叠为 258 个唯一 assembly accession，train/val/test 为 192/35/31，cross-split duplicate 为 0；已提交 `cu` array `8470511` 生成 Stage B/C1/C2/D 窗口候选，并提交依赖 array `8470515` 在窗口候选完成后编码 `uint8 input_ids`。
- 2026-06-09 21:10:00 CST: Stage B/C1/C2/D 固化输入全部完成并整理为简约 `training_server_transfer/`；总目录约 50GB，Stage_B/C1/C2/D 分别约 29GB/15GB/4.8GB/2.0GB，`sha256sum -c SHA256SUMS` 全部通过，可整体搬运到训练服务器。
- 2026-06-09 21:42:00 CST: 在 `training_server_transfer/` 新增简洁训练脚本目录 `scripts/` 和正式训练配置 `configs/model_large.json`、`configs/train_stage_*.json`；完成语法检查、stage package quick check 和全目录 SHA256 校验。
- 2026-06-10 09:45:00 CST: 在 `zuowu_genomemodel` 环境安装 `numpy 2.2.6`、CUDA PyTorch `2.5.1` (`torch.version.cuda=12.4`) 和 `torchrun`；导出 `training_server_transfer/configs/zuowu_genomemodel_env.yml`，完成训练脚本 tiny dry-run、package quick check 和全目录 SHA256 校验。
- 2026-06-10 11:17:00 CST: 为 `zuowu_genomemodel` 补装 CUDA `nvcc 12.4.131` 和基础构建工具，并重新导出环境到 `training_server_transfer/configs/zuowu_genomemodel_env.yml`；尝试安装 `mamba-ssm`，但登录节点源码构建强制编译多架构导致 `cicc` 被系统终止，当前训练脚本使用 `hyena_lite` 后端；重新完成 package quick check、tiny dry-run 和全目录 SHA256 校验。
- 2026-06-10 12:45:11 CST: 在 `PROJECT_PLAN.md` 详细说明 Stage B/C1/C2/D 的长度组成比例是 stage 级 token 配方，不是候选池保留率或在某个长度桶内固定抽样比例；补充 Stage B 8K/4K/16K token 和等价窗口数估算。
- 2026-06-10 12:57:14 CST: 在 `PROJECT_PLAN.md` 补充 Stage B `30,600,306,688` token 的来源: train 约 30B、validation/test 各约 0.3B，并可由 `summary.tsv` 的 `written_tokens` 或 `manifest.tsv` 的 shard token 求和复现。
- 2026-06-10 13:59:12 CST: 在 `PROJECT_PLAN.md` 进一步解释 Stage B `30B` 是人为设定的训练 token 预算；用“300 个 8K 候选窗口”的例子说明 70% 是最终写入 token 份额，不是候选窗口数量份额。
- 2026-06-10 14:21:58 CST: 在 `PROJECT_PLAN.md` 补充 Stage B `30B` 预算的估算依据: 模型规模、Stage B 局部语法任务定位、结构注释高质量数据、磁盘体积、GPU 时间和后续 C1/C2/D 扩长训练共同约束。
- 2026-06-10 18:04:57 CST: 按用户确认改为“主 context 候选全部保留 + 更严格窗口质量过滤”；收紧 promoter distal、gene_body、background 保留比例和 N/连续 N/低复杂度阈值，预计最终搬运目录约 60-80GB。
- 2026-06-10 23:47:01 CST: 完成新策略下 Stage B/C1/C2/D 输入生成；实际写入 41.24B/20.47B/6.90B/2.78B token，`training_server_transfer/` 总体约 67G；package quick check、Stage_D 训练 dry-run 和全目录 `sha256sum -c SHA256SUMS` 均通过。
- 2026-06-11 08:20:04 CST: 更新训练计划，新增结构基因组增强层；将 telomere/subtelomere、centromere/pericentromere、TE family、TE boundary、satellite/tandem repeat、rDNA/organellar insertion、segmental duplication 和 synteny-breakpoint 纳入候选池、输入字段、loss、下游任务和评测记录模板。
- 2026-06-11 09:04:34 CST: 基于现有 assembly/contig/QC 元数据完成结构基因组自注释可行性初评；258 个 canonical assembly 中，236 个可做 TE/repeat 和 satellite/tandem repeat，229 个可做 telomere/centromere 候选，228 个可作为第一批结构增强全套候选。
- 2026-06-11 10:02:58 CST: 按“只处理可做结构注释的 assembly”启动正式结构注释批处理；q07/q08 提交结构扫描 237 个 targets，2 核 8G array 高并发运行，用于 telomere/subtelomere、centromere-like/pericentromere-like、repeat-rich 和 satellite proxy；同时提交 EDTA 236 个 targets，8 核 48G array 运行，用于 TE family、TE boundary 和 repeat 注释。任务 job ID: 结构扫描 `8541368`/`8541367`，EDTA `8541135`/`8541134`。
- 2026-06-11 14:05:52 CST: 根据 EDTA issue 281 风险检查，发现旧 EDTA 中间 `.fa.mod` 已出现纯数字序列 ID；已取消旧 EDTA array，改为先生成 `z000001` 这类长度 <=13 且非纯数字的 EDTA-safe FASTA，并保存原始 contig 映射表后重跑。安全版 EDTA 输出目录为本地 `structural_annotation/edta_safe/`，q07 job ID 为 `8551051`，资源仍为每任务 8 核 48G，array throttle 66；自动汇总 job ID 为 `8551052`。
- 2026-06-14 23:15:31 CST: 在 GPU 服务器 `gpu10` 启动正式 `v1-backbone` Stage_B 预训练；启动前完成 Stage_B package check、CUDA 可见性检查、GPU 2 空闲检查和 GPU dry-run。正式训练仅暴露 `CUDA_VISIBLE_DEVICES=2`，使用 1 张 NVIDIA A100-SXM4-40GB，远端主进程 PID `111856`，训练日志 `training_server_transfer/logs/v1_backbone_stage_B_gpu2_20260614_230834.log`。
