# Hermes 交接文档: zuowu_genomemodel

更新时间: 2026-06-22 12:30 CST
项目目录: `/home/user/zhangzhishuai/myhermes/zuowu_genomemodel`  
GitHub: `git@github.com:shuai19910911/zuowu_genomemodel.git`

## 0. 2026-06-22 最新状态覆盖说明

- GPU Stage_B（第二阶段训练）formal CaduceusRC（正式反向互补一致性模型）训练已按用户要求停止；GPU2（2号显卡）上的目标训练进程树已退出，停止后 GPU2 显存占用为 0 MiB。
- 停止时最后 train step（训练步）为 8500；最新完整 checkpoint（模型存档点）是 `training_server_transfer/runs/Stage_B_formal_caduceus_rc/checkpoints/step_00008000.pt`。
- validation loss（验证损失）最低点是 step5000，val loss（验证损失）为 1.079774；step8000 val loss（验证损失）为 1.122568。
- 128 bp（128 个碱基对）downstream probe（下游探针评测）最高点是 step5000，Macro-F1（类别平均 F1）为 0.242750；step8000 Macro-F1（类别平均 F1）为 0.208736，仍高于 1-mer baseline（单碱基组成基线）0.147429。
- 最终评价文档: `docs/TRAINING_EVALUATION_STOP_20260622.md`。
- 当前结论: 不建议继续盲目延长同一配置；下一步优先对 step5000 和 step8000 做更大样本、更长序列长度的正式 downstream benchmark（下游基准评测）。

## 1. 交接给 Hermes 的一句话

这是一个作物基因组预训练大模型项目。请在当前目录 `/home/user/zhangzhishuai/myhermes/zuowu_genomemodel` 继续工作，先完整阅读本文件、`README.md`、`PROJECT_PLAN.md`、`MODEL_ARCHITECTURE.md`、`TRANSFER_DIRECTORY_STRUCTURE.md`、`docs/TRAINING_METRICS.md`，然后接管正在运行的 GPU Stage_B 预训练和 CPU EDTA 结构注释任务。所有新脚本正式运行前必须做逻辑核查、语法检查和最小可行运行检查；CPU 未运行任务只允许提交到 `q07`；GPU 当前只允许使用 2 号卡，除非用户明确修改。

## 2. 用户核心目标和偏好

- 目标: 做一个正式的作物基因组预训练大模型，不做玩具版、不做测试版训练。
- 数据: 使用 `/home/user/zhangzhishuai/data/plantDB/genome/README.md` 中整理的 plantDB genome 数据；正式训练只纳入有 genome 且有结构注释 GFF/GTF 的作物 assembly。
- GitHub 只放项目介绍、计划、模型结构、路线图、进展、指标，不上传基因组原始数据、中间大数据、训练输入 shard、checkpoint。
- 进展记录必须带具体时间点。
- 尽量少生成进展文件，一两个主文件能看懂即可。
- 用户希望 Hermes 自动监控任务，运行结束后接着做后续命令。
- 用户要求: 每次生成/修改训练、数据处理、注释、搬运脚本后，必须先检查代码逻辑和准确性，再运行正式程序。这条规则已经写入 `README.md`。

## 3. 集群和资源规则

- 当前机器是 Slurm 登录节点。
- CPU 命令: 未运行任务只允许提交到 `q07`。不要再把新 pending 任务提交到 `q08/cu/q04/q05`。
- CPU 节点上限: 每个计算节点最多 30 核、150G 内存。EDTA 当前每任务 8 核、48G。
- Python 环境: `mamba` 环境 `zuowu_genomemodel`。
- EDTA 环境: `conda run -n EDTA2.0 EDTA.pl`，不要修改 EDTA2.0 环境；缺软件时安装到 `zuowu_genomemodel` 或用户允许的位置。
- GPU 服务器: `12.12.12.210`，主机名查询时显示 `gpu10`。本交接文档不保存密码；如果 Hermes 需要 SSH，请让用户重新提供或使用已有凭据。
- GPU 当前规则: 只使用 2 号卡，命令形如 `CUDA_VISIBLE_DEVICES=2 ...`。不要使用其他卡，除非用户明确允许。

## 4. 主要文档

- `README.md`: 项目介绍、核心方案、集群边界、执行规则、时间线进展。
- `PROJECT_PLAN.md`: 详细训练计划，包括数据过滤、split、区域权重、输入、loss、训练阶段、资源估算、下游任务、评测结果记录表。
- `MODEL_ARCHITECTURE.md`: 模型结构，包括当前 v1-backbone 和后续 v1.1 cross-strand midtraining 计划。
- `TECHNICAL_ROADMAP.md`: 技术路线图说明。
- `TRANSFER_DIRECTORY_STRUCTURE.md`: 训练服务器搬运目录结构。
- `STRUCTURAL_ANNOTATION_FEASIBILITY.md`: 结构基因组自注释可行性评估。
- `docs/TRAINING_METRICS.md`: Stage_B 训练指标看板。
- `assets/training_metrics/stage_B_loss.svg`: GitHub 上展示的 loss 曲线。

## 5. 数据和训练输入现状

正式训练数据口径:

- 只使用结构注释完整的 crop assembly。
- 263 行 manifest，去重后 258 个 canonical assembly accession。
- 覆盖 26 个属。
- 放弃缺少结构注释的 assembly 约 1644 个。
- 原始压缩 genome 约 214.11GB，GFF/GTF 压缩合计约 2.89GB。

训练搬运目录:

- `training_server_transfer/`
- 该目录是设计给训练服务器整体搬运或直接共享访问的简洁目录。
- 当前数据已固化为 Stage B/C1/C2/D 输入 shard，总体约 67GB。
- 2026-06-10 23:47 CST 已完成 `sha256sum -c SHA256SUMS`、package quick check、Stage_D dry-run。

当前固化输入规模:

- Stage_B: 约 41.24B token，主上下文 8K，保留全部 8K 主候选并混入其他长度。
- Stage_C1: 约 20.47B token，主上下文 64K。
- Stage_C2: 约 6.90B token，主上下文 128K。
- Stage_D: 约 2.78B token，主上下文 256K。

注意: `training_server_transfer/`、`slurm/`、`scripts/`、`analysis/`、`structural_annotation/` 等多数运行目录没有被 Git 追踪，GitHub 只保存计划和进展文档。

## 6. 模型和训练策略

当前正式模型:

- 名称: `CropGenome-FM-Large`
- 当前后端: `hyena_lite`
- 原计划偏向 Mamba2/Hyena + periodic attention；但 `mamba-ssm` 在登录节点源码构建时失败，所以当前训练脚本使用 `hyena_lite`。
- 输入: 单碱基/小词表 `input_ids`，stage 固化窗口，训练时动态 mask、labels、RC 增强、batch 顺序。
- 训练阶段: B -> C1 -> C2 -> D。
- 当前正在跑 Stage_B。

CrossDNA 启发的后续升级:

- 已调研 Nature Machine Intelligence 文章 `Explicit dynamic cross-strand interactions for DNA sequence language modelling` 和 CrossDNA 官方代码。
- 当前决策: 不打断正在跑的 v1-backbone Stage_B。
- 后续从 Stage_B checkpoint 做 `v1.1-cross-strand-midtraining`，加入 forward/reverse-complement 双分支和轻量 cross-strand communication。
- 用 RC consistency、splice、promoter/TSS、enhancer/open chromatin、variant effect probe 判断是否进入 C1/C2。

## 7. GPU Stage_B 当前状态

远程训练日志:

`training_server_transfer/logs/v1_backbone_stage_B_gpu2_20260614_230834.log`

远程运行信息:

- 启动时间: 2026-06-14 23:08 左右。
- README 记录时间: 2026-06-14 23:15:31 CST。
- 远程主 shell PID: `111856`
- mamba 进程 PID: `112297`
- Python 主训练进程 PID: `112300`
- dataloader worker: `112723` 等。
- 命令: `bash scripts/run_stage.sh Stage_B 1` -> `mamba run -n zuowu_genomemodel python scripts/train.py --data-root . --config configs/train_stage_B.json --model-config configs/model_large.json`
- GPU: 只使用 GPU 2。

最新检查状态，2026-06-16 13:10 CST:

- 最新 step: `3050`
- 最新 loss: `1.186195082962513`
- 最新 lr: `6.1e-05`
- GPU 2: `32561 MiB`, `100%`
- loss 已从约 1.24-1.25 下降到约 1.18-1.20，趋势正常。

checkpoint 状态:

- 当前还没有 checkpoint 文件。
- 当前运行进程启动时 `save_every=5000`，所以如果不中断，第一次 checkpoint 应在 step 5000。
- 用户后来要求 1000 step 保存一次，实际共享配置 `training_server_transfer/configs/train_stage_B.json` 已改为 `save_every=1000`，但正在运行的进程不会动态重读配置。
- 因此: 不重启则第一 checkpoint 仍约 step 5000；以后从 checkpoint resume 或重启后才按 1000 step。

常用 GPU 查询命令:

```bash
SSHPASS='用户重新提供密码后设置' sshpass -e ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/home/user/zhangzhishuai/myhermes/zuowu_genomemodel/.ssh_known_hosts 12.12.12.210 \
  "tail -n 20 /home/user/zhangzhishuai/myhermes/zuowu_genomemodel/training_server_transfer/logs/v1_backbone_stage_B_gpu2_20260614_230834.log"
```

```bash
SSHPASS='用户重新提供密码后设置' sshpass -e ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/home/user/zhangzhishuai/myhermes/zuowu_genomemodel/.ssh_known_hosts 12.12.12.210 \
  "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"
```

不要在文档或回答里泄露 GPU 密码。

## 8. CPU EDTA/结构注释当前状态

结构注释目标:

- 对可做结构注释的 assembly 做 telomere/subtelomere、centromere/pericentromere、TE family、TE boundary、satellite/tandem repeat 等结构信息。
- 结构扫描目标约 237 个。
- EDTA 目标 236 个。

EDTA issue 281 处理:

- 旧 EDTA array 曾发现 `.fa.mod` 出现纯数字序列 ID，存在 RunGRF.py 风险。
- 已取消旧 unsafe EDTA array。
- 新方案先生成 EDTA-safe FASTA，seq ID 形如 `z000001`，长度 <=13 且非纯数字，并保存原 contig 映射。
- 安全输出目录: `structural_annotation/edta_safe/`
- 安全脚本: `slurm/run_edta_safe_array.sh`
- 该脚本本地已补 `#SBATCH -p q07`，并通过 `bash -n`。

旧/新 EDTA job 关系:

- `8551051`: 最早提交的安全 EDTA array，原本覆盖 `1-236`。其中 `1-80` 已经启动，继续属于 `8551051`。部分已运行子任务被 Slurm 调度到 q04/q05，这是旧 array 的历史调度结果，不再取消，避免浪费已经跑了多天的计算。
- `8567363`: 2026-06-16 新提交的补跑 array，只覆盖旧作业中还未运行的 `81-236`，强制 `q07`。
- `8567367`: 新 summary job，依赖 `afterany:8551051:8567363`，等旧已运行部分和新补跑部分都结束后再汇总。

2026-06-16 13:10 CST 队列状态:

- `8551051_1-80` 中一批仍在运行，包含 q07/q04/q05 历史子任务。
- `8567363_81`, `8567363_82`, `8567363_83` 已在 q07 运行。
- `8567363_[84-236%66]` 在 q07 pending，原因 Resources。
- `8567367` 在 q07 pending，原因 Dependency。

重要: 后续未运行部分已经从旧 `8551051_[81-236]` 迁移到新 `8567363_[81-236]`，pending 只在 q07。不要再次取消正在运行的 `8551051_1-80`，除非用户明确要求。

常用 CPU 查询命令:

```bash
squeue -u zhangzhishuai -o "%.18i %.9P %.8j %.2t %.12M %.6D %R"
```

```bash
squeue -j 8551051,8567363,8567367 -o "%.18i %.9P %.8j %.2t %.12M %.6D %R"
```

如果需要数 EDTA done 文件，`find structural_annotation/edta_safe -name '*.EDTA.done'` 可能很慢，慎用；优先看 Slurm 状态和具体日志。

## 9. GitHub 同步状态

最近 Git commits:

- `1838aa4 restrict CPU submissions to q07`
- `11d46ec update Stage B progress and checkpoint policy`
- `4086a4c add Stage B training metrics dashboard`
- `d8d393a add CrossDNA inspired cross-strand plan`
- `49b72c6 document v1 backbone training start`
- `2199041 document code validation rule`
- `c157377 document EDTA safe sequence IDs`
- `8524682 document structural annotation batch jobs`

当前 Git 注意事项:

- `.ssh_known_hosts` 是本地未追踪文件，不要提交。
- `analysis/`、`structural_annotation/` 是本地大目录，不要提交。
- `training_server_transfer/` 被忽略，不要上传 GitHub。
- `slurm/` 当前不是 Git 追踪目录；里面脚本本地已改为 q07，但不会自动出现在 GitHub。
- 如果要把交接文档提交到 GitHub，只提交 `HERMES_HANDOFF.md` 和必要的 README/计划文档，避免误加大目录。

推荐提交检查:

```bash
git status --short
git diff --stat
```

只添加目标文档:

```bash
git add HERMES_HANDOFF.md
git commit -m "add Hermes handoff document"
git push
```

## 10. 下一步建议

1. 继续监控 GPU Stage_B，等 step 5000 产生第一个 checkpoint。
2. checkpoint 出现后，立即更新 `docs/TRAINING_METRICS.md` 和 `assets/training_metrics/stage_B_loss.svg`，并提交 GitHub。
3. 若用户坚持 1000 step 保存策略立即生效，需要评估是否从最新 checkpoint resume；但现在还没有 checkpoint，不建议中断。
4. 继续监控 `8551051` 和 `8567363`。等都结束后，确认 `8567367` summary 是否正常运行。
5. 结构注释完成后，将 TE/telomere/centromere/satellite 等信息整合回候选池和后续训练计划，但不要随意重做已经用于 Stage_B 的输入，除非用户决定启动结构增强版再训练或 midtraining。
6. 如果 Stage_B first checkpoint loss/val 正常，下一步可以按计划推进 C1，或者先做 v1.1 cross-strand midtraining 小阶段。

## 11. Hermes 接管时给它的提示词

用户切换到 Hermes 后，可以直接这样说:

```text
请接管作物基因组预训练大模型项目。项目目录是 /home/user/zhangzhishuai/myhermes/zuowu_genomemodel。请先阅读 HERMES_HANDOFF.md、README.md、PROJECT_PLAN.md、MODEL_ARCHITECTURE.md、TRANSFER_DIRECTORY_STRUCTURE.md 和 docs/TRAINING_METRICS.md，然后继续监控 GPU Stage_B 训练和 CPU EDTA 结构注释。注意: 新 CPU 未运行任务只允许 q07；GPU 当前只允许 2 号卡；所有新脚本正式运行前必须先做逻辑核查、语法检查和最小可行运行检查；不要提交大数据目录到 GitHub。请先检查当前进展并告诉我下一步。
```

如果 Hermes 需要登录 GPU 服务器，用户应另行提供凭据；不要让模型从旧聊天里复述或保存密码。

## 12. 高风险点

- 不要取消 `8551051` 正在运行的子任务，除非用户明确要求；它们已运行多天。
- 不要把 `training_server_transfer/`、`structural_annotation/`、原始 genome 数据、checkpoint 上传 GitHub。
- 不要在回答或文件里记录 GPU 密码。
- 不要误以为 `save_every=1000` 已对当前运行进程生效；当前进程大概率仍 step 5000 才保存。
- 不要用没有结构注释的 genome 进入正式训练。
- 不要让同一个 assembly/genome 片段跨 train/val/test 泄漏；split 是 accession-level canonical split。
- 不要对 CDS/splice/start/stop 做相似性丢弃，只做质量过滤；背景和 intergenic 才做更强去冗余和采样控制。

