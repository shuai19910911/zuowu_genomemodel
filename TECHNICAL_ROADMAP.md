# CropGenome-FM 技术路线图

更新时间: 2026-06-08 15:00:51 CST

![CropGenome-FM 技术路线图](assets/cropgenome_fm_roadmap.svg)

## 路线概览

1. 数据口径: 只使用 262 个有 FASTA + GFF3/GTF 的结构注释完整基因组；放弃 1644 个无结构注释基因组。
2. 数据 QC: 过滤短 contig、高 N、极端 GC、organelle contig、坏注释、低复杂度窗口和跨 split 近重复窗口。
3. 区域构建: 从 GFF/GTF 构建 CDS、exon、UTR、splice flank、promoter/TSS、TES/polyA、intron、high-quality intergenic 和 background。
4. 严格 split: 先按 assembly/species/genus/gene-family/LD block 分组，再窗口化，保证同一基因组片段不跨 train/val/test。
5. 区域采样: CDS 30%、splice 12%、promoter/TSS 15%、UTR 8%、TES 8%、intron 12%、high-quality intergenic 10%、background 5%。
6. 数据搬运: 搬原始压缩数据 + 小索引 + configs，不搬全量 token shards；训练服务器使用 100-200GB 磁盘 cache 在线生成 token/window shard。
7. 模型输入: `input_ids`、`labels_mlm`、`loss_mask`、`region_ids`、`region_weights`、`quality_scores` 和坐标 metadata。
8. 模型架构: RC-equivariant Mamba2/Hyena + periodic local/global sparse attention，Large 约 300M-450M 参数。
9. Loss: region-weighted MLM + causal next-token likelihood + reverse-complement consistency + optional region contrastive loss。
10. 训练计划: CPU 预处理 2-5 天；GPU 8K -> 64K -> 128K -> 256K 正式训练约 4-10 周；下游评测 1-2 周；推荐训练服务器 1.5TB 可用磁盘。
11. 下游任务: splice、TIS/TTS、promoter/TES、CDS/UTR/intron、lncRNA/mRNA、chromatin、expression、variant effect。
12. 基线和优势: 对比 CNN、DNABERT-2、AgroNT、PlantCAD2、HyenaDNA/Evo2；预计在 splice、TIS/TTS、结构区域分类、promoter/TES 和 gene-family holdout lncRNA/mRNA 上更强。

## 路线图维护规则

- 数据口径、模型架构、区域权重、训练阶段或资源估算发生变化时，同步更新 `PROJECT_PLAN.md`、`MODEL_ARCHITECTURE.md` 和本路线图。
- SVG 是仓库内可追踪的技术路线图；生成式图片可作为展示图，但正式计划以 Markdown 和 SVG 为准。
