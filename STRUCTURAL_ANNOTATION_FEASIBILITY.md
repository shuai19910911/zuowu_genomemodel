# Structural Annotation Feasibility

更新时间: 2026-06-11 08:45:00 CST

本文件评估当前 258 个 canonical crop assembly 中，有多少适合自行补充结构基因组注释，包括 TE/repeat、telomere/subtelomere、centromere/pericentromere、satellite/tandem repeat、rDNA/organellar insertion 和 segmental duplication。

评估基于现有本地 QC 文件，不是正式结构注释结果:

- `data_manifests/assemblies.canonical.tsv`
- `sequence_index/fasta_qc.summary.tsv`
- `sequence_index/contigs.tsv`

详细中间表保存在本地:

```text
analysis/structural_annotation_feasibility/
  structural_annotation_feasibility.tsv
  structural_annotation_feasibility.summary.tsv
  structural_annotation_feasibility.by_genus.tsv
```

## 总体结论

| 注释类型 | 可做数量 | 说明 |
|---|---:|---|
| TE/repeat de novo 注释 | 236 / 258 | 221 个 high，15 个 medium；适合 EDTA/RepeatModeler/RepeatMasker 流程 |
| satellite/tandem repeat 注释 | 236 / 258 | 适合 TRF/TRASH/RepeatOBserver 等 tandem repeat 检测 |
| telomere/subtelomere 候选注释 | 229 / 258 | 215 个 high_candidate，14 个 medium_candidate；还需端粒 motif 扫描确认 |
| centromere/pericentromere 候选注释 | 229 / 258 | 214 个 medium_candidate，15 个 weak_candidate；真正 high confidence 仍需 CENH3/Hi-C/T2T 或强外部证据 |
| TE + telomere + centromere + satellite 全套候选 | 228 / 258 | 推荐作为第一批结构增强训练输入候选 |
| 全部结构条件都差 | 21 / 258 | 不建议第一批做结构增强，只保留 gene/CDS/splice 等已有结构注释 |

## Assembly Level 分布

| assembly_level | assembly 数 | 全套候选 | TE/repeat 可做 | telomere 候选 | centromere 候选 | satellite 可做 |
|---|---:|---:|---:|---:|---:|---:|
| Complete Genome | 13 | 13 | 13 | 13 | 13 | 13 |
| Chromosome | 232 | 215 | 215 | 216 | 216 | 215 |
| Scaffold | 12 | 0 | 7 | 0 | 0 | 7 |
| Contig | 1 | 0 | 1 | 0 | 0 | 1 |

## 推荐执行分层

第一批直接做结构增强注释:

- 228 个 `TE + telomere + centromere + satellite` 全套候选。
- 这些 assembly 基本是 chromosome/complete genome 级别，适合先跑 TE/repeat、tandem repeat、端粒 motif、centromere-like repeat density 和结构候选整合。

第二批只做 TE/satellite:

- 额外 8 个 assembly 可做 TE/repeat 或 satellite/tandem repeat，但不适合作为端粒/着丝粒候选。
- 这些可以进入 TE family、TE boundary、satellite repeat 任务，但不进入 telomere/centromere 严格监督。

暂不做结构增强:

- 21 个 assembly 结构连续性或质量不足。
- 这些只保留当前 gene/CDS/splice/promoter/TES 等已有注释训练，不做端粒/着丝粒/TE family 监督正例。

## 分级规则

本次只是可行性分级，正式注释仍需后续运行工具。

| 类型 | high/medium 判定依据 |
|---|---|
| TE/repeat | genome 总长 >= 50 Mb，平均 N <= 5%；chromosome/complete 且 contig 不过度碎片化为 high，否则为 medium |
| telomere | chromosome/complete genome 且大 contig 覆盖度高；后续必须再做 `TTTAGGG/CCCTAAA` 端粒 motif 端部扫描 |
| centromere | chromosome/complete genome 且大 contig 覆盖度高；当前只能作为 centromere-like 候选，high confidence 需要 CENH3/Hi-C/T2T/外部注释 |
| satellite/tandem repeat | genome 总长 >= 50 Mb，平均 N <= 5%，并有 chromosome/complete 级别或足够 softmask/repeat 信号 |

## 后续建议

1. 先对 228 个全套候选 assembly 跑轻量端粒 motif + tandem repeat 扫描，确认端粒和 satellite 候选。
2. 对 236 个 TE/satellite 可做 assembly 分批跑 EDTA 或 RepeatModeler/RepeatMasker；大基因组如 Hordeum、Triticum、Saccharum 单独排队。
3. 着丝粒不要直接当 high-confidence 标签，只先生成 `centromere_like` 和 `pericentromere_like`，后续有 CENH3/Hi-C/T2T 注释时再升级。
4. 结构增强训练时只把 high/medium 证据写入 `structure_flags`；low 证据只作为 QC flag，不做监督正例。

## 执行进展

- 2026-06-11 10:02:58 CST: 已按本文件筛出的可做 assembly 启动正式批处理，不对全部 genome 目录盲目处理。结构扫描对 237 个 targets 运行，提交到 q07/q08，资源为每任务 2 核、8G 内存、高并发 array，用于生成 telomere/subtelomere、centromere-like/pericentromere-like、repeat-rich 和 satellite proxy 注释。EDTA 对 236 个 targets 运行，提交到 q07/q08，资源为每任务 8 核、48G 内存、每分区 9 个并发，用于生成 TE family、TE boundary 和 repeat 注释。`EDTA2.0` 环境只用于调用既有软件，不在该环境内安装或改动软件。
- 2026-06-11 14:05:52 CST: 根据 EDTA issue 281 检查输入序列 ID 风险，确认旧 EDTA 中间 `.fa.mod` 中已有纯数字 ID。为避免 `RunGRF.py` 因纯数字或超长 ID 失败，旧 EDTA array 已取消，正式结果改用 `structural_annotation/edta_safe/`。安全版流程在运行 EDTA 前把每条序列改名为 `z000001`、`z000002` 等长度 <=13 且非纯数字的 ID，并保存 `*.edta_safe.seqid_map.tsv` 用于回溯原始 contig。安全版 EDTA q07 job ID 为 `8551051`，每任务 8 核、48G，array throttle 66；后续汇总 job ID 为 `8551052`。
