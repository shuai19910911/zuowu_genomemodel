# CropGenome-FM作物基因组基础模型
## 详细研究设计、数据字典、模型架构、预训练与下游评估报告

- **文档版本：** 1.0
- **机器证据截止：** 2026-07-21 14:04 CST（UTC+08:00）
- **当前冻结基座：** `CropGenomeFM_step14000`
- **文档性质：** 当前事实审计＋下一阶段预注册式研究设计

**重要原则：** 所有“当前性能”来自已存在的机器结果；所有“计划任务、最低样本数、成功阈值”均为未来设计，不是已完成结果。

> 一句话状态：8K Stage B基座及10个正式下游任务（核心3项＋外部7项）已经完成；64K Stage C1只完成架构Gate和569个训练step，未到第一次验证；Stage C2/D只有物化数据与旧配置，没有正式训练。当前结果证明模型有有效作物序列表征，但尚不能宣称64K长程模型完成，也不能宣称全面超过PlantCAD2/PlantCaduceus，更没有NT-v2 500M的现有成绩。

![CropGenome-FM架构](figures/figure_01_architecture.png)

## 1. 阅读指南与证据等级

本报告把内容分成三层，避免把未来愿景写成已经实现的结果。

1. **F（formal，正式证据）**：数据、split、模型revision、checkpoint、主指标和测试结果均已冻结；test只用于最终报告。
2. **D（diagnostic，诊断证据）**：validation-only、轻量probe或checkpoint筛选结果，只用于调试和模型选择，不能替代正式测试。
3. **P（planned，计划）**：尚未完成数据冻结或训练；文中给出的窗口数是最低目标，不是现有样本数。

术语说明：token在本模型中基本对应单个碱基字符；AUPRC是精确率—召回率曲线下面积，越高越好，尤其适合不平衡分类；Pearson是预测与真实值的线性相关，越高越好；probe是冻结基础模型、只训练简单下游头；RC是reverse complement（反向互补）。

## 2. 研究目标、核心假设与不能预设的结论

### 2.1 总目标

构建一个真正面向作物和植物基因组的基础模型：既能在启动子、剪接、转录终止、增强子、lncRNA和表达预测等通用任务上接近或超过公开基因组大模型，也能在多倍体同源基因、TE插入调控、超长基因结构、NLR抗病基因簇、作物pan-genome结构变异等作物专属复杂问题上形成可验证优势。

### 2.2 可检验假设

- **H1：作物域预训练有效。** 同一架构的预训练模型应稳定超过random-init；当前正式数据支持H1。
- **H2：上下文增益真实。** 2K→8K→64K的提升必须来自额外远端序列，而不是padding、样本变化或probe自由度；当前核心nested-context支持到8K，但64K尚无正式测试。
- **H3：作物域优于纯规模。** 在作物专属任务上，中等规模作物模型可超过更大的通用模型；当前部分任务支持，但PlantCAD2/PlantCaduceus仍是强对手。
- **H4：多任务预训练改善结构理解。** MLM＋RC一致性＋区域辅助目标应改善启动子/剪接/TES表征；需通过消融而非只看单个最终模型验证。
- **H5：长程架构对复杂区域必要。** 64–256K完整上下文应在长基因、多倍体flank、TE/SV和NLR簇任务上超过短模型的native输入；必须同时提供chunked基线，不能把“模型跑不了”伪装成0分。

**不能预设的结论：** 新任务不能为了让本模型赢而事后筛选；PlantCAD2、PlantCaduceus、AgroNT、NT-v2 500M、Evo2必须在结果揭盲前注册；若本模型不超过它们，应如实报告。

## 3. 当前项目状态总表

| 模块 | 当前状态 | 可以支持的表述 | 不能支持的表述 |
| --- | --- | --- | --- |
| Stage B 8K | step17000触发早停；validation最佳并冻结step14000 | 8K基座训练和正式下游评估完成 | 完整遍历一遍语料或充分训练至收敛 |
| Stage C1 64K | 真实65,536 bp单步Gate通过；正式run停在step569 | 架构/显存/反向传播路径可运行 | 完成64K预训练、64K验证或64K下游收益 |
| Stage C2 128K | 数据包与旧训练配置存在；未训练 | 128K数据已经物化 | 已有128K模型 |
| Stage D 256K | 数据包与旧训练配置存在；未训练 | 256K数据已经物化 | 已有256K模型 |
| CropGenome-Bench-v1 | 3任务×3 nested contexts正式完成 | 核心结构任务正式比较 | 覆盖所有作物功能问题 |
| Plant Genomic Benchmark | 7任务正式完成；512/6000 bp按能力覆盖 | 外部增强子/lncRNA/表达证据 | 所有任务多seed稳定性 |
| NT-v2 500M | 官方模型标识/config已核验，但未下载冻结和评估 | 下一版必加基线 | 任何当前性能比较 |

## 4. 预训练原始数据：组成、来源与split

### 4.1 原始来源

- 来源是NCBI GenBank和NCBI RefSeq的公开plant/crop genome assembly FASTA与GFF3注释。
- canonical manifest含**258个assembly、30个物种、23个属**。
- split为train 206、validation 25、test 27个assembly。
- 来源构成为GenBank 209、RefSeq 49。
- manifest记录的压缩下载文件合计：genome约16.66 GB、annotation约2.32 GB；这是压缩artifact字节，不是解压后碱基总数。
- split策略名为`assembly_accession_level_genus_stratified_v2`。多个assembly来自同一物种/属，因此这不是“预训练完全genus-disjoint”。报告零样本跨属能力时必须另建从预训练中完全未见的外部属。

### 4.2 数据用途

- **train assembly**生成训练窗口并参与梯度更新。
- **validation assembly**生成固定验证窗口，用于loss、checkpoint和早停；不能用于拟合模型权重。
- **test assembly**保留为阶段性审计/正式评估候选；不能参与早停。
- GFF不仅用于构建下游标签，也用于预训练窗口的区域类型辅助标签。因此未来结构任务必须在assembly、同源家族和序列近重复层面去泄漏。

### 4.3 全部物种组成

完整30物种逐项表见附录A；表中bytes来自压缩下载artifact。

## 5. 预训练窗口数据包

![预训练数据量与状态](figures/figure_02_pretraining_data.png)

### 5.1 四阶段总量

| Stage | 执行状态 | 总窗口 | 总碱基token | train窗口 | validation窗口 | test窗口 | context窗口组成 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Stage_B | completed_early_stop_step17000_selected_step14000 | 5,594,781 | 41.243B | 5,485,240 | 54,695 | 54,846 | 4096:1494149;8192:3913853;16384:186779 |
| Stage_C1 | partial_graceful_stop_step569_no_validation | 779,304 | 20.470B | 764,070 | 7,576 | 7,658 | 4096:186779;8192:280160;16384:46705;32768:23360;65536:242300 |
| Stage_C2 | data_ready_training_not_started | 101,293 | 6.898B | 99,316 | 970 | 1,007 | 8192:31138;16384:15576;65536:11686;131072:42893 |
| Stage_D | data_ready_training_not_started | 18,400 | 2.778B | 18,040 | 176 | 184 | 8192:6236;65536:787;131072:2344;262144:9033 |

### 5.2 精确split token量

| Stage | train token | validation token | test token | 全部token |
| --- | ---: | ---: | ---: | ---: |
| Stage_B | 40,435,101,696 | 403,083,264 | 404,320,256 | 41,242,505,216 |
| Stage_C1 | 20,073,304,064 | 195,743,744 | 201,117,696 | 20,470,165,504 |
| Stage_C2 | 6,766,624,768 | 63,365,120 | 68,214,784 | 6,898,204,672 |
| Stage_D | 2,726,723,584 | 24,510,464 | 26,607,616 | 2,777,841,664 |

### 5.3 区域类型组成

区域标签固定为7类：background、coding、gene_body、promoter、splice、tes、utr。窗口总数按区域为：

| Stage | 区域窗口组成 |
| --- | --- |
| Stage_B | background:223487;coding:1790629;gene_body:559494;promoter:783111;splice:1175065;tes:447604;utr:615391 |
| Stage_C1 | background:31122;coding:249390;gene_body:77931;promoter:109128;splice:163650;tes:62340;utr:85743 |
| Stage_C2 | background:4045;coding:32425;gene_body:10127;promoter:14175;splice:21286;tes:8094;utr:11141 |
| Stage_D | background:742;coding:5880;gene_body:1842;promoter:2578;splice:3868;tes:1470;utr:2020 |

### 5.4 采样和“epoch”的正确解释

训练集是无限`IterableDataset`：先按shard窗口数加权采样shard，再在shard内有放回抽取窗口，并以0.5概率做RC增强。因此训练没有传统DataLoader epoch。

- step14000累计约503,000个Stage B窗口，约为训练池的0.0917 epoch-equivalent；估计约3.708B碱基token，约10.03 token/参数。
- step17000累计约611,000个窗口，约0.1114 epoch-equivalent；估计约4.504B token，约12.19 token/参数。
- Stage B大约要到step152,396才相当于抽取一遍训练窗口池；当前最大步数50,000本来也不足一遍。
- Stage C1 step569抽取72,832个窗口，约为C1训练池的0.0953 epoch-equivalent；但只完成计划180,000步的0.316%，且仍处在6,000步warmup内。

结论：没到1个epoch不等于没有学到有效表示，但不能宣称语料被完整遍历；Stage C1更不能视为完成。

## 6. 模型架构详解

### 6.1 总体结构

当前模型实际有**369,505,287个trainable parameters**。输入采用字符级7-token词表：A、C、G、T、N、MASK、PAD；隐藏维度1,024。主干由32个HyenaLite块和8个局部注意力插入块构成，总执行深度40个块。

| 组件 | 参数 | 数值 | 实现状态 | 白话解释 |
| --- | --- | --- | --- | --- |
| model_total | trainable_parameters | 369505287 | measured_from_training_runtime | 实际可训练参数总数 |
| tokenizer | vocabulary | A,C,G,T,N,MASK,PAD (7) | implemented | 单碱基字符级输入；输出头预测A/C/G/T/N |
| embedding | d_model | 1024 | implemented | 每个位置1024维隐藏表示 |
| backbone | HyenaLite_blocks | 32 | implemented | 线性复杂度局部卷积门控块 |
| HyenaLite | expand/kernel/mlp_ratio | 2 / 127 / 1.25 | implemented | 通道扩展、深度卷积核、SwiGLU前馈宽度 |
| local_attention | insertions | 8 (after every 4 HyenaLite blocks) | implemented | 总执行块数40=32+8 |
| local_attention | heads/chunk | 8 heads / 512 positions | implemented | 每个注意力子块在重排后的512位置组内计算 |
| local_attention | dilations | 1,2,4,8,16,32,64,128 | implemented_in_C1_config | 逐层扩大跨chunk依赖跨度且不新增参数 |
| normalization | norm | RMSNorm eps=1e-6 | implemented | 稳定长序列训练 |
| regularization | dropout | 0.05 | implemented | Hyena、注意力和前馈残差dropout |
| strand_symmetry | RC_equivariance | direct+reverse-complement dual pass, aligned average | implemented | 正向与反向互补预测/隐藏表示对齐后平均 |
| MLM_head | output | 5 nucleotide classes | implemented | 只预测A/C/G/T/N，不预测MASK/PAD |
| region_head | classes | 7 | implemented | background,coding,gene_body,promoter,splice,tes,utr |
| memory | gradient_checkpointing | true | implemented | 以额外计算换显存，支持64K |
| context | Stage_B_evaluated | 8192 bp | completed | 冻结step14000基座 |
| context | Stage_C1_gate | 65536 bp | gate_pass_partial_training_only | 单步前向/反向/optimizer通过；正式训练仅到step569 |

### 6.2 HyenaLite块

每个HyenaLite块先RMSNorm，再把通道扩展为2倍并分成内容/门控两支；内容支经过kernel=127的depthwise 1D卷积，和SiLU门控相乘后投回1,024维，最后接SwiGLU前馈残差。该实现是“HyenaLite”，不是完整长隐式滤波Hyena；单块主要提供局部卷积建模，长程依赖还依赖层叠与稀疏注意力。

### 6.3 局部扩张注意力

每4个HyenaLite块插入1个attention块，共8个。attention有8头，每组最多512个位置。Stage C1配置把8个attention插入依次设为dilation 1、2、4、8、16、32、64、128：先按步长重排位置，再在每个512位置组内注意，从而用近线性开销跨更远距离。dilation不增加参数，但改变计算拓扑，所以Stage B和C1不是完全相同的前向图。

### 6.4 RC等变设计

模型分别处理原序列和反向互补序列，把RC输出翻转回原坐标，并把A↔T、C↔G的类别通道对齐，再平均logits和hidden state。这样获得方向稳健性，但近似使一次样本进行两次主干forward，训练成本也近似翻倍。

### 6.5 三个输出用途

1. **MLM head：** 每个位置输出A/C/G/T/N五类，用于遮盖碱基恢复。
2. **Region head：** 对全序列池化后预测7类窗口来源区域，是窗口级辅助任务，不是逐碱基标注。
3. **Frozen embedding：** 正式下游冻结encoder，用简单线性/岭回归probe比较不同基础模型的表示质量。

## 7. 预训练任务与损失

| 目标 | 版本 | 预测内容 | 损失 | 权重 | 用于checkpoint选择 | 状态 | 解释边界 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MLM | current | 随机遮盖15%的有效A/C/G/T位置并恢复原碱基 | cross-entropy over masked positions | 1.0 | yes | implemented | 自监督序列建模，不直接等于功能标签 |
| RC-consistency | current | 正向与反向互补方向的对齐分布一致 | symmetric KL divergence over valid positions | 0.02 | 0.02 | implemented | 主要验证链方向稳健性；random-init也可能因结构获得一致性 |
| region-classification | current | 每个窗口的7类来源区域 | label-smoothed cross-entropy | 0.05 | no | implemented | 窗口级辅助标签，不是逐碱基结构标注 |
| token-region-segmentation | recommended_v3 | 逐碱基/分块预测promoter/UTR/CDS/intron/splice/TES/background | class-balanced focal or cross-entropy + Dice | TBD by validation-only pilot | candidate | not_implemented | 必须对预训练与下游benchmark标签来源做隔离 |
| boundary-distance | recommended_v3 | 到TSS/TES/splice/TE边界的有符号距离或区间 | Huber + boundary classification | TBD | candidate | not_implemented | 只用train assemblies生成标签 |
| multi-context-consistency | recommended_v3 | 同一中心位点在1K/8K/32K/64K/128K上下文保持局部表示一致，同时允许远端全局表示增益 | stop-gradient contrastive/cosine consistency | TBD | candidate | not_implemented | 不能强迫所有层完全相同而抹除长程信息 |
| allele-pair-contrastive | recommended_v3 | 同一locus ref/alt或同源单倍型的表示差异与真实序列变化对应 | contrastive + delta reconstruction | TBD | candidate | not_implemented | 若使用功能效应标签则属于监督适配，不得混入相同formal test |

### 7.1 MLM

有效A/C/G/T位置以15%概率被选中：80%替换成MASK、10%替换成随机A/C/G/T/N、10%保持原字符。交叉熵只在被选择位置计算。每个样本至少强制一个mask，避免短序列没有监督。

### 7.2 RC一致性

在所有有效位置计算正向分布与RC对齐分布的双向KL平均：它约束两条链方向的预测一致。需要注意，架构本身已经做RC平均，random-init也可能显示一定一致性；必须用下游性能而非单一RC loss证明生物学价值。

### 7.3 区域分类

对有区域标签的窗口计算带0.05 label smoothing的7分类交叉熵。Stage B总loss为`1.0×MLM + 0.02×RC + 0.05×region`；早停selection loss只用`MLM + 0.02×RC`，避免辅助region头主导基座选择。

## 8. 训练超参数、验证与当前终态

### 8.1 Stage B 8K

- 计划50,000 optimizer steps；前1,000步micro-batch=5、gradient accumulation=7，之后为4×9。
- AdamW：peak LR=1e-4、min LR=1e-5、weight decay=0.1、1,000步warmup、bf16。
- 每1,000步固定validation 64 batches并保存checkpoint。
- 早停：最早5,000步；patience=3次验证；min_delta=0.002。
- validation最佳为step14000；step17000的selection loss虽略低，但相对最佳改善小于min_delta，连续3次未达到显著改善，故早停。
- 正式主模型固定step14000；step17000只保留作sensitivity，不可事后替换主结论。

### 8.2 Stage C1 64K

- 从Stage B step14000权重启动新阶段；micro-batch=1、accumulation=128。
- 计划180,000步；peak LR=1.2e-4、min LR=1.2e-5、warmup=6,000、validation every 1,500 steps×256 batches。
- 要求dry-run必须抽到真实65,536 bp样本；gradient checkpointing开启。
- 正式run收到SIGTERM后优雅停在step569并保存interrupted checkpoint；没有到step1500，所以没有任何C1 validation结果。
- 旧phase-transition配置设`resume_optimizer=false`和`reset_step_on_resume=true`；若未来要从step569“精确续训”，必须另行冻结明确的optimizer/step恢复合同，不能直接把旧默认当精确resume。

### 8.3 Stage C2/D

C2旧配置为90,000步、1×128、LR 8e-5、context最高128K；D旧配置为45,000步、1×128、LR 5e-5、context最高256K。两者都未启动，旧配置只能视为草案，不应在没有真实H20 preflight和P0下游任务前直接执行。

## 9. 下游评估总原则

1. encoder全部冻结，只拟合线性分类或ridge regression；这比较“表示可用性”，不是完整fine-tuning上限。
2. 超参数只能由validation选择，test不参与选择。
3. 同一模型必须先通过zero-truncation：如果tokenizer产生的token数超过模型上限，该模型在该任务/上下文记为N/E，不允许静默截断。
4. 核心三任务使用完全相同的样本身份做512/2,048/8,192中心裁剪，保证context比较不换样本。
5. 正式主指标：二分类AUPRC；表达回归macro Pearson。AUROC、Spearman、R²、MAE是支持指标。
6. 核心任务有5个probe seeds。外部二分类任务有5个probe seeds（13/29/47/71/101），full-data结果在现有实现下完全相同；五个表达回归任务使用确定性ridge并只保留seed 13。回归任务仍不能宣称多seed稳定性。
7. checkpoint候选先在validation-only诊断上筛；formal test不按每个checkpoint重复跑，避免测试集泄漏。

## 10. 核心正式任务：CropGenome-Bench-v1

### 10.1 数据构建

三项任务都从原始FASTA和GFF/GTF坐标构建，不使用Stage B的proxy label。每任务每个context有6,144个样本：train 4,096、validation 1,024、test 1,024；每个split正负1:1。

| 任务 | 标签来源 | nested窗口 | train | validation | test | hard negative定义 |
| --- | --- | --- | --- | --- | --- | --- |
| splice_donor_acceptor | raw GFF/GTF coordinates plus FASTA | 512;2048;8192 nested | 4,096 | 1,024 | 1,024 | same canonical GT/AG motif inside annotated introns, away from annotated junctions |
| promoter_TSS | raw GFF/GTF coordinates plus FASTA | 512;2048;8192 nested | 4,096 | 1,024 | 1,024 | same-assembly shifted upstream promoter decoy, excluding annotated gene boundaries |
| TES_polyA | raw GFF/GTF coordinates plus FASTA | 512;2048;8192 nested | 4,096 | 1,024 | 1,024 | same-assembly shifted downstream decoy, excluding annotated gene boundaries |

### 10.2 split物种

- **train：** Prunus persica、Setaria italica、Sorghum bicolor、Vigna radiata、Vitis vinifera；promoter/TES另含Brassica rapa。
- **validation：** Beta vulgaris、Daucus carota、Manihot esculenta。
- **test：** Cucumis sativus、Oryza sativa、Solanum tuberosum。

这是下游probe的species-disjoint split；它不等于test物种从未出现在基础模型预训练中。未来“严格跨物种零样本”必须另外检查预训练manifest。

### 10.3 三个任务的生物学意义

- **Promoter/TSS：** 判断窗口中心是否为注释转录起始位点。它检验启动子局部motif和更远的上游/基因背景。
- **TES/poly(A)：** 判断中心是否为转录终止/多聚腺苷酸相关末端。它检验3′端信号和下游基因组背景；GFF标签不等于实验测得的精确poly(A) cleavage site。
- **Splice donor/acceptor：** 正例是注释内含子边界，负例是内含子中相同GT/AG canonical motif但非真实边界的hard decoy。这避免模型只记“GT/AG”捷径。

GC均值按任务/split匹配，跨split坐标不重复，N比例≤5%。

## 11. 核心任务全部正式性能

![核心任务性能](figures/figure_03_core_performance.png)

以下全部值是formal test、full-data、5个probe seeds的AUPRC均值。模型缺少某context能力时不列入该context排名。

#### 512 bp

| 排名 | 模型/方法 | TES/poly(A) | promoter/TSS | splice donor/acceptor | 宏平均AUPRC |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | PlantCAD2_Small | 0.8526 | 0.8627 | 0.8197 | 0.8450 |
| 2 | PlantCaduceus_l32 | 0.8658 | 0.8628 | 0.8049 | 0.8445 |
| 3 | CropGenomeFM_step14000 | 0.8015 | 0.8513 | 0.8455 | 0.8328 |
| 4 | AgroNT_1B | 0.8180 | 0.8231 | 0.7934 | 0.8115 |
| 5 | NTv2_100M_multi_species | 0.7797 | 0.7958 | 0.7403 | 0.7719 |
| 6 | central_33bp_linear_probe | 0.6997 | 0.7271 | 0.8476 | 0.7582 |
| 7 | HyenaDNA_medium_160k | 0.7775 | 0.7700 | 0.7037 | 0.7504 |
| 8 | Caduceus_PS_131k | 0.7494 | 0.7580 | 0.7324 | 0.7466 |
| 9 | DNABERT2_117M | 0.7734 | 0.7399 | 0.7174 | 0.7436 |
| 10 | 3mer_linear_probe | 0.7494 | 0.6686 | 0.6583 | 0.6921 |
| 11 | 6mer_linear_probe | 0.6925 | 0.6717 | 0.6531 | 0.6724 |
| 12 | CropGenomeFM_random_init | 0.6660 | 0.6677 | 0.6717 | 0.6685 |
| 13 | composition_linear_probe | 0.6141 | 0.5604 | 0.6644 | 0.6130 |
| 14 | 1mer_linear_probe | 0.5980 | 0.5555 | 0.6186 | 0.5907 |
| 15 | Evo2_1B_base | 0.5761 | 0.5554 | 0.6161 | 0.5826 |
| 16 | majority_baseline | 0.3073 | 0.3073 | 0.3073 | 0.3073 |
#### 2,048 bp

| 排名 | 模型/方法 | TES/poly(A) | promoter/TSS | splice donor/acceptor | 宏平均AUPRC |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | CropGenomeFM_step14000 | 0.9064 | 0.8892 | 0.8548 | 0.8835 |
| 2 | AgroNT_1B | 0.9016 | 0.8722 | 0.7805 | 0.8514 |
| 3 | PlantCAD2_Small | 0.9006 | 0.8823 | 0.7638 | 0.8489 |
| 4 | PlantCaduceus_l32 | 0.9049 | 0.8621 | 0.7768 | 0.8480 |
| 5 | NTv2_100M_multi_species | 0.8577 | 0.8465 | 0.7344 | 0.8129 |
| 6 | Caduceus_PS_131k | 0.8249 | 0.8081 | 0.6895 | 0.7741 |
| 7 | HyenaDNA_medium_160k | 0.8372 | 0.7866 | 0.6896 | 0.7712 |
| 8 | central_33bp_linear_probe | 0.6997 | 0.7271 | 0.8476 | 0.7582 |
| 9 | CropGenomeFM_random_init | 0.7432 | 0.7527 | 0.6046 | 0.7002 |
| 10 | 3mer_linear_probe | 0.7590 | 0.6171 | 0.6264 | 0.6675 |
| 11 | 6mer_linear_probe | 0.6395 | 0.6372 | 0.5875 | 0.6214 |
| 12 | Evo2_1B_base | 0.5921 | 0.6162 | 0.6017 | 0.6033 |
| 13 | composition_linear_probe | 0.5476 | 0.5097 | 0.6480 | 0.5684 |
| 14 | 1mer_linear_probe | 0.5401 | 0.5108 | 0.5945 | 0.5485 |
| 15 | majority_baseline | 0.3073 | 0.3073 | 0.3073 | 0.3073 |
#### 8,192 bp

| 排名 | 模型/方法 | TES/poly(A) | promoter/TSS | splice donor/acceptor | 宏平均AUPRC |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | CropGenomeFM_step14000 | 0.9036 | 0.8984 | 0.7628 | 0.8549 |
| 2 | PlantCaduceus_l32 | 0.9258 | 0.8809 | 0.6601 | 0.8222 |
| 3 | PlantCAD2_Small | 0.9257 | 0.9025 | 0.6151 | 0.8144 |
| 4 | Caduceus_PS_131k | 0.8582 | 0.8574 | 0.5994 | 0.7717 |
| 5 | central_33bp_linear_probe | 0.6997 | 0.7271 | 0.8476 | 0.7582 |
| 6 | CropGenomeFM_random_init | 0.8090 | 0.7875 | 0.5230 | 0.7065 |
| 7 | HyenaDNA_medium_160k | 0.8284 | 0.6813 | 0.5495 | 0.6864 |
| 8 | 3mer_linear_probe | 0.7495 | 0.6950 | 0.5293 | 0.6579 |
| 9 | 6mer_linear_probe | 0.6561 | 0.6085 | 0.5554 | 0.6067 |
| 10 | composition_linear_probe | 0.5533 | 0.5116 | 0.5686 | 0.5445 |
| 11 | 1mer_linear_probe | 0.5469 | 0.5104 | 0.5625 | 0.5400 |
| 12 | Evo2_1B_base | 0.5220 | 0.5599 | 0.5337 | 0.5386 |
| 13 | majority_baseline | 0.3073 | 0.3073 | 0.3073 | 0.3073 |

### 11.1 关键解释

- 512 bp：本模型宏平均0.8328，排名3；PlantCAD2为0.8450、PlantCaduceus为0.8445，本模型比两者约低0.012。它高于AgroNT 1B的0.8115。
- 2,048 bp：本模型0.8835，排名1；高于AgroNT 1B约0.0320，也高于PlantCAD2/PlantCaduceus约0.0345/0.0355。
- 8,192 bp：本模型0.8549，排名1；高于PlantCaduceus约0.0327、PlantCAD2约0.0405。
- 8,192 bp宏平均优势主要来自splice任务：本模型0.7628，PlantCaduceus 0.6601，PlantCAD2 0.6151；而TES上两个植物基线仍约0.9258，高于本模型0.9036。不能只报宏平均而隐藏任务异质性。
- 相对同架构random-init，本模型在512/2,048/8,192 bp宏平均分别提高约0.1642/0.1833/0.1484，支持预训练有效。
- Evo2在这个“冻结embedding＋线性probe＋本地8K runtime”协议下较弱，不代表Evo2生成能力或其他fine-tuning协议普遍较弱。

### 11.2 AUPRC并列分数完整性审计

本次整理发现，核心formal evaluator的自定义AUPRC实现按稳定排序逐个累积正例，但没有把相同probability的样本作为一个并列阈值组处理。因此，存在大量完全相同分数的方法会受文件行顺序影响。最明显的信号是：每个test split有512正例和512负例，恒定分数`majority_baseline`的标准Average Precision应等于正例率0.5，而锁定正式表给出0.30734。

本报告没有覆盖或改写既有formal artifact，而是从3个context的`formal_test_predictions.tsv`对full-data全部预测做了tie-safe重算，并保存为`source_data/core_auprc_tie_audit_summary.tsv`。结果如下：

- `majority_baseline`应由0.30734校正为0.50000。
- 1/3/6-mer、composition等容易产生并列分数的baseline会有较明显变化；最大单任务/seed差异约0.06245。
- CropGenome-FM、PlantCAD2、PlantCaduceus、AgroNT等连续概率学习模型只发生极小数值变化；512/2,048/8,192 bp的主学习模型排名均不变。
- 外部二分类复用了同一metric函数；对34个task×model×context的primary-seed prediction重算后，最大绝对差异约0.000384，模型排名不变。
- 下一正式release必须改用经过单元测试的标准Average Precision实现并重新生成全部表、图、manifest；在此之前，旧表中的tie-sensitive baseline AUPRC只作历史记录，不用于科学结论。

## 12. 外部任务：Plant Genomic Benchmark

### 12.1 来源与去泄漏

数据来自Hugging Face `InstaDeepAI/plant-genomic-benchmark`，实际下载manifest和dataset manifest共同绑定revision `78ec8156c2ffb3e5475277fdb7eb603294224e53`，共33个原始artifact，约1.356 GB。构建后对跨split完全相同序列去重；删重只删除train/validation中的冲突样本，test保持冻结；处理后每任务跨split exact duplicate=0。

| 任务 | 类型/维度 | 总样本 | train | validation | test | 长度min/median/max(bp) | 去重前跨split重复 | 删除train/val | 去重后重复 | 验证集策略 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lncrna_multispecies | binary / 1 | 56,795 | 45,588 | 5,112 | 6,095 | 102/498/6000 | 1074 | 1260/7 | 0 | sha256(sample identity) modulo 10 from official train; official test preserved |
| enhancer_cassava_proseq | binary / 1 | 18,893 | 16,852 | 1,229 | 812 | 1000/1000/1000 | 0 | 0/0 | 0 | official validation split |
| gene_expression_arabidopsis_thaliana | multi_regression / 56 | 32,529 | 25,728 | 3,399 | 3,402 | 6000/6000/6000 | 5 | 3/2 | 0 | official validation split |
| gene_expression_glycine_max | multi_regression / 14 | 56,709 | 47,105 | 4,801 | 4,803 | 6000/6000/6000 | 5 | 31/2 | 0 | official validation split |
| gene_expression_oryza_sativa | multi_regression / 7 | 38,626 | 31,224 | 3,700 | 3,702 | 6000/6000/6000 | 20 | 20/2 | 0 | official validation split |
| gene_expression_solanum_lycopersicum | multi_regression / 10 | 34,960 | 27,307 | 3,826 | 3,827 | 6000/6000/6000 | 15 | 14/1 | 0 | official validation split |
| gene_expression_zea_mays | multi_regression / 23 | 43,452 | 34,487 | 4,482 | 4,483 | 6000/6000/6000 | 7 | 6/1 | 0 | official validation split |

### 12.2 七项任务含义

1. **Cassava PRO-seq enhancer：** 木薯PRO-seq活性增强子二分类；manifest中的序列长度固定为1,000 bp。6000-bp运行只能补PAD，不会新增真实远端序列，因此不能用其512→6000差异证明长上下文。
2. **Multi-species lncRNA：** 长度102–6,000 bp、中位数498 bp的lncRNA二分类，检验跨植物物种非编码转录本特征；6000-bp运行包含每条样本的全部可用序列，但很多样本远短于6K。
3. **Arabidopsis expression：** 56个输出维度的表达回归。
4. **Soybean expression：** 14个输出维度，是最直接的大豆相关外部任务之一。
5. **Rice expression：** 7个输出维度。
6. **Tomato expression：** 10个输出维度。
7. **Maize expression：** 23个输出维度。

表达任务用各输出维度Pearson后做macro average。由于维度间样本可用性不同，还应检查per-output结果，不能只看一个宏平均。

## 13. 外部任务全部正式性能

![外部任务性能](figures/figure_04_external_performance.png)

表中二分类列是AUPRC，五个表达列是macro Pearson。为使任务共用一张表，展示的是full-data primary seed 13；二分类另有29/47/71/101四个seed且现有结果与seed 13完全相同，表达回归为确定性ridge、只保留seed 13。`—`表示zero-truncation协议下未评估，不是0分。

#### 512 bp

| 模型 | Cassava enhancer | lncRNA | Arabidopsis expr. | Soybean expr. | Rice expr. | Tomato expr. | Maize expr. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CropGenomeFM_step14000 | 0.7842 | 0.6947 | 0.1066 | 0.1286 | 0.0908 | 0.2367 | 0.1611 |
| CropGenomeFM_random_init | 0.7183 | 0.5883 | 0.0462 | 0.1055 | 0.0494 | 0.2312 | 0.1286 |
| AgroNT_1B | 0.7856 | 0.6562 | 0.0439 | 0.1135 | 0.0967 | 0.2113 | 0.1479 |
| PlantCAD2_Small | 0.8572 | 0.7419 | 0.1151 | 0.1668 | 0.1235 | 0.2932 | 0.1921 |
| PlantCaduceus_l32 | 0.8207 | 0.7480 | 0.1534 | 0.1568 | 0.1088 | 0.2952 | 0.2019 |
| NTv2_100M_multi_species | 0.7487 | 0.6485 | 0.0585 | 0.1133 | 0.0725 | 0.2085 | 0.1328 |
| DNABERT2_117M | 0.7404 | 0.6570 | 0.0536 | 0.0965 | 0.0739 | 0.2231 | 0.1340 |
| HyenaDNA_medium_160k | 0.6886 | 0.6198 | 0.0309 | 0.1154 | 0.0652 | 0.2316 | 0.1200 |
| Caduceus_PS_131k | 0.7543 | 0.6177 | 0.0697 | 0.1223 | 0.0557 | 0.2529 | 0.1240 |
| Evo2_1B_base | 0.5387 | 0.5499 | 0.0439 | 0.0954 | 0.0470 | 0.1989 | 0.0471 |
#### 6,000 bp

| 模型 | Cassava enhancer | lncRNA | Arabidopsis expr. | Soybean expr. | Rice expr. | Tomato expr. | Maize expr. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CropGenomeFM_step14000 | 0.8321 | 0.7072 | 0.5520 | 0.5673 | 0.5028 | 0.5891 | 0.6350 |
| CropGenomeFM_random_init | 0.6854 | 0.5894 | 0.3654 | 0.4298 | 0.3595 | 0.4560 | 0.4187 |
| PlantCAD2_Small | 0.8615 | 0.7400 | 0.5518 | 0.5855 | 0.5181 | 0.6306 | 0.6739 |
| PlantCaduceus_l32 | 0.8263 | 0.7535 | 0.5596 | 0.5988 | 0.5210 | 0.6352 | 0.6711 |
| HyenaDNA_medium_160k | 0.7033 | 0.6226 | 0.4967 | 0.5164 | 0.4455 | 0.4942 | 0.4577 |
| Caduceus_PS_131k | 0.7589 | 0.6280 | 0.3956 | 0.4432 | 0.4195 | 0.4814 | 0.4804 |
| Evo2_1B_base | 0.4948 | 0.5554 | 0.1688 | 0.1818 | 0.0727 | 0.2395 | 0.1032 |

### 13.1 外部结果解释

- 512 bp Cassava enhancer：本模型0.7842，接近AgroNT 0.7856，但低于PlantCAD2 0.8572和PlantCaduceus 0.8207。
- 6,000 bp Cassava enhancer：本模型0.8321，高于PlantCaduceus 0.8263，但低于PlantCAD2 0.8615；因原始序列固定为1,000 bp，所谓6K输入主要是padding，这不是长程证据。
- 6,000 bp lncRNA：本模型0.7072，低于PlantCaduceus 0.7535和PlantCAD2 0.7400。
- 五个6,000 bp表达任务：本模型分别为Arabidopsis 0.5520、soybean 0.5673、rice 0.5028、tomato 0.5891、maize 0.6350；全部低于PlantCaduceus，除Arabidopsis与PlantCAD2几乎相同外，其余也低于PlantCAD2。
- 本模型在大多数外部任务明显超过同架构random-init，说明预训练仍有效；但植物专属外部泛化目前不是第一，主要差距应作为下一阶段优化目标，而不是用新任务掩盖。
- AgroNT和NT-v2 100M在6000 bp外部面板因tokenization后存在超过模型上限的样本而被整体排除，避免不透明截断。
- NT-v2 500M没有任何当前结果；下一版必须加入后再谈与“Nucleotide Transformer v2 500M”的正式比较。

## 14. 基线模型、参数量与公平角色

| 模型 | 角色 | 参数量 | 架构 | 当前native/评估上下文 | 预训练域 | 当前证据状态 |
| --- | --- | --- | --- | --- | --- | --- |
| CropGenomeFM_step14000 | our_frozen_model | 369505287 | 32 HyenaLite blocks + 8 dilated local-attention blocks; RC-equivariant | 8192 bp Stage B; 65536 bp architecture gate | 258 crop/plant assemblies; 30 species | evaluated |
| CropGenomeFM_random_init | same_architecture_ablation | 369505287 | same as CropGenomeFM | 512/2048/8192 and 512/6000 | none | evaluated |
| AgroNT_1B | edible_plant_primary | 991261206 | 40-layer ESM-style Transformer; 6-mer tokenizer | 1024 tokens; evaluated 512/2048 bp; external 6000 excluded by zero-truncation audit | about 10.5M sequences from 48 primarily edible plant species; Ensembl Plants | evaluated |
| PlantCAD2_Small | plant_primary | 175980688 | 24-layer bidirectional RCPS Caduceus/Mamba2; d_model 768 | evaluated to 8192 bp core and 6000 bp external | plant model; exact public model release bound by revision | evaluated |
| PlantCaduceus_l32 | plant_primary | 426689552 | 32-layer bidirectional RCPS Caduceus; d_model 1024 | evaluated to 8192 bp core and 6000 bp external | plant model; exact public model release bound by revision | evaluated |
| NTv2_100M_multi_species | general_multispecies | 97889484 | 22-layer Transformer; rotary position; 6-mer tokenizer | 2050 positions; evaluated 512/2048 bp; external 6000 excluded by zero-truncation audit | 850 non-plant/non-virus genomes; 300B training tokens | evaluated |
| NTv2_500M_multi_species | required_next_release_general_multispecies | 500M nominal; exact local weights not yet frozen | 29-layer Transformer; d_model 1024; rotary position; 6-mer tokenizer | 2048 tokens; must pass zero-truncation audit per task | 850 non-plant/non-virus genomes; 900B training tokens | planned_not_evaluated |
| DNABERT2_117M | general_short_context | 120219904 | 12-layer BERT; BPE-like DNA tokenizer | 512 positions; evaluated 512 bp only | general DNA | evaluated |
| HyenaDNA_medium_160k | general_long_context | 14236768 | Hyena sequence model | nominal 160k; evaluated 8192/6000 | general DNA | evaluated |
| Caduceus_PS_131k | general_rc_long_context | 7725344 | bidirectional RCPS Caduceus | nominal 131k; evaluated 8192/6000 | general DNA | evaluated |
| Evo2_1B_base | evolution_scale_general | 1B nominal | 25-layer StripedHyena2; character tokenizer | evaluated configuration max 8192 bp | multi-domain genomic sequence | evaluated |

### 14.1 三个最关键植物/通用基线

- **AgroNT 1B：** 本地权重精确统计991,261,206个参数；40层ESM-style Transformer；6-mer tokenizer；官方model card说明基于48种主要可食用植物、约10.5M序列和472.5B token训练，native 1,024 tokens约6,144 bp。它是“大规模植物短上下文”关键基线。
- **PlantCAD2 Small：** 本地权重精确统计175,980,688个参数；24层、d_model=768的bidirectional RCPS Caduceus/Mamba2类模型。它在当前多个外部任务最强，是必须正面超过而不能规避的基线。
- **NT-v2 500M multi-species：** 官方config为29层、d_model=1,024、16头、max_position_embeddings=2,050，官方card称2048 tokens和约500M参数、850个非植物/非病毒物种基因组、900B训练token。当前只核验了官方revision，尚未冻结权重或跑分。

PlantCaduceus l32本地权重为426,689,552个参数，也是当前强植物长上下文基线。Evo2、HyenaDNA和Caduceus用于覆盖通用进化规模或长上下文架构。

## 15. 当前证据能回答什么

### 15.1 已支持

- CropGenome-FM预训练显著优于同架构random-init。
- 2K和8K核心结构任务上，当前冻结模型宏平均超过已评估植物/通用基线。
- 在部分长窗口外部任务上，模型学到有用作物表示。
- 369.5M混合架构可以执行真实64K前向、反向和optimizer step。

### 15.2 尚未支持

- 64K/128K/256K预训练模型完成。
- 64K带来真实下游增益。
- 全面超过PlantCAD2、PlantCaduceus或AgroNT。
- 超过NT-v2 500M；它尚未评估。
- Evo2在所有协议中较弱。
- 表达预测可用于直接育种决策或因果基因发现。

## 16. 当前研究设计的主要局限

1. Stage B只相当于约0.09–0.11遍训练窗口池；是否欠训练需看validation和token/parameter曲线，而非只看epoch。
2. Stage C1只有569步，无validation，不能参与任何科学结论。
3. 预训练split是assembly级分层，不是严格全属隔离；跨属零样本能力尚未充分证明。
4. 核心任务都来自GFF结构注释，和预训练区域辅助标签存在任务相关性；必须用未见assembly/同源家族和外部功能数据控制。
5. 外部二分类虽有5个seed，但现有full-data值完全相同；表达回归只有确定性seed 13。按species/study bootstrap和独立训练重复评估的不确定性仍不足。
6. 当前正式下游是frozen probe，不等于全参数fine-tuning或LoRA上限。
7. 当前外部表达面板上PlantCAD2/PlantCaduceus普遍更强。
8. 当前缺NT-v2 500M；任何“全面基线矩阵”都还不完整。
9. 某些公开模型因native上下文/Tokenizer限制未评估长窗口；不能把N/E按0分计入排名。
10. 目前没有真正针对作物多倍体、TE/SV和育种locus的冻结formal任务。

## 17. 下一版作物专属下游任务矩阵

下面9项是预注册候选。最低样本数是任务建设门槛，不是现有数据。P0先做，P1在公开数据release/accession冻结后做，P2缺可靠标签时明确暂停。

| 优先级 | 任务ID | 科学问题 | 标签 | 输入长度 | 最低计划train/val/test | 数据状态 | 主指标 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0_data_ready | CROP-LONGGENE-SEG | 能否在完整长基因上下文中联合标注启动子、UTR、CDS、内含子、剪接边界和TES | token-level multiclass segmentation + boundary detection | 8192;32768;65536;131072 | 100000 windows / 10000 windows / 10000 windows | current source available; task release not built | macro-F1; boundary-F1@2/10bp; segment IoU |
| P0_buildable_plus_functional_labels | CROP-DISTAL-CIS-PAIR | 局部2K序列近似相同时，远端顺式调控背景能否决定启动子/TES活性 | paired binary/ranking with local-context matched pairs | 2048;8192;32768;65536 | 40000 paired windows / 5000 pairs / 5000 pairs | candidate sources partly available; exact functional accessions must be frozen | paired AUPRC; pair accuracy; context gain; calibration |
| P1_needs_release_freeze | CROP-ISOFORM-LR | 能否利用完整基因上下文预测组织/胁迫条件下的可变剪接和可变poly(A)使用 | multi-label junction/polyA usage; delta-PSI or usage regression | 8192;65536;131072 | 30000 genes/isoform windows / 5000 / 5000 | not frozen; no current formal result | macro-AUPRC; Pearson(delta-PSI); junction boundary F1 |
| P1_crop_specific | CROP-POLYPLOID-HOMEOLOG | 能否识别小麦/棉花/油菜同源亚基因组基因的表达优势、抑制或平衡状态 | 3-class dominance + continuous homoeolog expression difference | 8192;32768;65536 | 20000 homoeolog sets / 4000 sets / 4000 sets | not frozen; no current formal result | macro-F1; macro Pearson; within-triad ranking accuracy |
| P1_crop_specific_long_context | CROP-TE-SV-REG | 能否从含/不含TE插入的等位/单倍型序列对预测表达方向和幅度变化 | paired ref/alt delta regression + direction classification + TE boundary | 32768;131072;262144 | 20000 allele pairs / 4000 pairs / 4000 pairs | blocked until EDTA QC and paired functional labels are frozen | delta-expression Pearson; sign AUROC/AUPRC; boundary F1; calibration |
| P1_crop_specific_repeat_rich | CROP-NLR-CLUSTER | 能否在重复富集的抗病NLR基因簇中识别完整基因、伪基因和簇边界 | token segmentation + intact-copy count + cluster classification | 65536;131072;262144 | 20000 cluster/background windows / 3000 / 3000 | not frozen; computational labels require manual QC tier | boundary F1; copy-count MAE; macro-F1; low-homology sensitivity |
| P1_functional | CROP-ACR-STRESS | 能否跨作物预测组织/胁迫特异开放染色质与远端调控元件 | multi-label accessibility/activity classification | 2048;8192;32768;65536 | 50000 peaks+matched negatives / 10000 / 10000 | one cassava task available; cross-crop panel not frozen | macro-AUPRC; per-tissue AUPRC; calibration; cross-species transfer |
| P2_breeding_relevance | CROP-PANGENOME-SV | 能否从成对单倍型长序列预测基因PAV/SV类别及其转录影响 | paired structural variant classification + expression delta | 32768;131072;262144 | 30000 locus pairs / 5000 pairs / 5000 pairs | not frozen; no current formal result | macro-F1; delta-expression Pearson; top-k locus ranking |
| P2_requires_curated_labels | CROP-QTL-VAR-RANK | 能否在QTL/GWAS区间内排序有实验或精细定位支持的候选基因/变异 | locus-wise ranking; ref-alt delta scoring | 8192;65536;131072 | at least 500 independent loci / at least 100 loci / at least 100 loci | currently blocked; missing locked labels/crosswalk | MRR; hit@1/5/10; locus-level AUCPR; bootstrap by locus |

## 18. 每个新增任务的详细设计

### 1. CROP-LONGGENE-SEG（P0_data_ready）

**要回答的问题。** 能否在完整长基因上下文中联合标注启动子、UTR、CDS、内含子、剪接边界和TES。

- **标签与输出：** token-level multiclass segmentation + boundary detection。
- **候选公开来源：** 现有258个NCBI GenBank/RefSeq assembly的FASTA+GFF3；只纳入高质量可调用区域。
- **当前数据状态：** current source available; task release not built。
- **输入窗口：** 8192;32768;65536;131072 bp。
- **最低计划规模（不是现有样本数）：** train=100000 windows，validation=10000 windows，test=10000 windows。若真实可用数据达不到该门槛，应缩小声明或停止任务，不能用复制样本凑数。
- **最高层split与去泄漏：** 最高层genus/assembly隔离；蛋白家族和15-mer近重复去泄漏；长基因长度分层。
- **主指标：** macro-F1; boundary-F1@2/10bp; segment IoU。
- **为什么能形成能力差异：** AgroNT/NT-v2短上下文不能一次看到完整64K/128K；必须另报chunked track；PlantCAD2/PlantCaduceus/HyenaDNA/Caduceus为真正长上下文对手。
- **预注册成功规则：** 相对最强native plant baseline macro-F1下界>0且绝对提升>=0.02；相对最佳chunked短模型>=0.03；Holm校正。
- **禁止越界的结论：** GFF注释预测，不等于新基因功能或湿实验验证。

### 2. CROP-DISTAL-CIS-PAIR（P0_buildable_plus_functional_labels）

**要回答的问题。** 局部2K序列近似相同时，远端顺式调控背景能否决定启动子/TES活性。

- **标签与输出：** paired binary/ranking with local-context matched pairs。
- **候选公开来源：** 现有GFF/FASTA用于结构候选；Plant Genomic Benchmark cassava PRO-seq及后续冻结的作物PRO-seq/CAGE/PAS-seq作为功能标签。
- **当前数据状态：** candidate sources partly available; exact functional accessions must be frozen。
- **输入窗口：** 2048;8192;32768;65536 bp。
- **最低计划规模（不是现有样本数）：** train=40000 paired windows，validation=5000 pairs，test=5000 pairs。若真实可用数据达不到该门槛，应缩小声明或停止任务，不能用复制样本凑数。
- **最高层split与去泄漏：** 按物种/实验study隔离；中心2K GC、motif、同源度匹配；远端序列置换作为反事实控制。
- **主指标：** paired AUPRC; pair accuracy; context gain; calibration。
- **为什么能形成能力差异：** 短模型在native track看不到被设计为决定标签的远端区域；PlantCAD2/PlantCaduceus等长模型仍可公平竞争。
- **预注册成功规则：** 64K相对2K paired AUPRC提升CI下界>0；相对最强native baseline>=0.02；远端置换必须显著消除增益。
- **禁止越界的结论：** 只有功能测序标签可支持调控活性；纯GFF版本只能称结构上下文任务。

### 3. CROP-ISOFORM-LR（P1_needs_release_freeze）

**要回答的问题。** 能否利用完整基因上下文预测组织/胁迫条件下的可变剪接和可变poly(A)使用。

- **标签与输出：** multi-label junction/polyA usage; delta-PSI or usage regression。
- **候选公开来源：** 公开作物Iso-Seq/long-read RNA-seq与配套短读长RNA-seq；NCBI SRA/ENA；exact accessions TBD。
- **当前数据状态：** not frozen; no current formal result。
- **输入窗口：** 8192;65536;131072 bp。
- **最低计划规模（不是现有样本数）：** train=30000 genes/isoform windows，validation=5000，test=5000。若真实可用数据达不到该门槛，应缩小声明或停止任务，不能用复制样本凑数。
- **最高层split与去泄漏：** study×species隔离；gene-family cluster disjoint；表达可检测性callable universe。
- **主指标：** macro-AUPRC; Pearson(delta-PSI); junction boundary F1。
- **为什么能形成能力差异：** 长内含子和多个候选位点超出AgroNT/NT-v2单窗口；chunking作为弱等价对照。
- **预注册成功规则：** 相对最强native baseline AUPRC>=+0.02或Pearson>=+0.03且CI下界>0。
- **禁止越界的结论：** 预测转录本使用，不直接证明表型因果。

### 4. CROP-POLYPLOID-HOMEOLOG（P1_crop_specific）

**要回答的问题。** 能否识别小麦/棉花/油菜同源亚基因组基因的表达优势、抑制或平衡状态。

- **标签与输出：** 3-class dominance + continuous homoeolog expression difference。
- **候选公开来源：** NCBI/Ensembl Plants参考与亚基因组注释；公开多组织RNA-seq/Expression Atlas；exact releases/accessions TBD。
- **当前数据状态：** not frozen; no current formal result。
- **输入窗口：** 8192;32768;65536 bp。
- **最低计划规模（不是现有样本数）：** train=20000 homoeolog sets，validation=4000 sets，test=4000 sets。若真实可用数据达不到该门槛，应缩小声明或停止任务，不能用复制样本凑数。
- **最高层split与去泄漏：** 整套homoeolog group不可跨split；按gene family、品种和study隔离；表达批次train-only归一化。
- **主指标：** macro-F1; macro Pearson; within-triad ranking accuracy。
- **为什么能形成能力差异：** 作物多倍体特有；通用非植物NT-v2无植物预训练，AgroNT虽植物专用但native上下文较短；PlantCAD2/PlantCaduceus为关键强基线。
- **预注册成功规则：** 超过max(AgroNT,PlantCAD2,PlantCaduceus,NT-v2-500M,Evo2)的同口径最强值，MMED=0.02 F1或0.03 Pearson，CI下界>0。
- **禁止越界的结论：** 表达优势不是亚基因组进化机制或育种价值的直接证明。

### 5. CROP-TE-SV-REG（P1_crop_specific_long_context）

**要回答的问题。** 能否从含/不含TE插入的等位/单倍型序列对预测表达方向和幅度变化。

- **标签与输出：** paired ref/alt delta regression + direction classification + TE boundary。
- **候选公开来源：** 作物pan-genome/SV release + EDTA高置信TE + matched RNA-seq/eQTL；exact source releases TBD。
- **当前数据状态：** blocked until EDTA QC and paired functional labels are frozen。
- **输入窗口：** 32768;131072;262144 bp。
- **最低计划规模（不是现有样本数）：** train=20000 allele pairs，validation=4000 pairs，test=4000 pairs。若真实可用数据达不到该门槛，应缩小声明或停止任务，不能用复制样本凑数。
- **最高层split与去泄漏：** 同一variant pair同split；按locus/gene family/品种隔离；仅callable matched negatives。
- **主指标：** delta-expression Pearson; sign AUROC/AUPRC; boundary F1; calibration。
- **为什么能形成能力差异：** 256K超出当前AgroNT/NT-v2和本地Evo2-8K；PlantCAD2/PlantCaduceus/HyenaDNA/Caduceus按真实native能力进入。
- **预注册成功规则：** 相对最强native长模型Pearson>=+0.03且CI下界>0；相对序列组成/SV长度基线显著。
- **禁止越界的结论：** 不得把缺失TE或无表达测量样本当负例；关联不等于因果。

### 6. CROP-NLR-CLUSTER（P1_crop_specific_repeat_rich）

**要回答的问题。** 能否在重复富集的抗病NLR基因簇中识别完整基因、伪基因和簇边界。

- **标签与输出：** token segmentation + intact-copy count + cluster classification。
- **候选公开来源：** 高质量作物组装；经NLR-Annotator/人工注释交叉确认的NLR registry；exact release TBD。
- **当前数据状态：** not frozen; computational labels require manual QC tier。
- **输入窗口：** 65536;131072;262144 bp。
- **最低计划规模（不是现有样本数）：** train=20000 cluster/background windows，validation=3000，test=3000。若真实可用数据达不到该门槛，应缩小声明或停止任务，不能用复制样本凑数。
- **最高层split与去泄漏：** 按NLR family cluster和作物genus隔离；重复相似度分层；近重复去泄漏。
- **主指标：** boundary F1; copy-count MAE; macro-F1; low-homology sensitivity。
- **为什么能形成能力差异：** 长重复簇是作物育种专属难点；短Transformer只能chunk；长植物模型是主要对手。
- **预注册成功规则：** boundary-F1>=+0.03或copy-count MAE相对降低>=10%，bootstrap CI支持且低同源子集保持。
- **禁止越界的结论：** 计算预测NLR结构，不代表抗病表型或功能验证。

### 7. CROP-ACR-STRESS（P1_functional）

**要回答的问题。** 能否跨作物预测组织/胁迫特异开放染色质与远端调控元件。

- **标签与输出：** multi-label accessibility/activity classification。
- **候选公开来源：** 公开作物ATAC-seq/DNase-seq/PRO-seq；木薯PRO-seq现有；其余exact accessions TBD。
- **当前数据状态：** one cassava task available; cross-crop panel not frozen。
- **输入窗口：** 2048;8192;32768;65536 bp。
- **最低计划规模（不是现有样本数）：** train=50000 peaks+matched negatives，validation=10000，test=10000。若真实可用数据达不到该门槛，应缩小声明或停止任务，不能用复制样本凑数。
- **最高层split与去泄漏：** study/site-year/species隔离；GC、mappability、distance-to-TSS匹配；blacklist和可调用区限定。
- **主指标：** macro-AUPRC; per-tissue AUPRC; calibration; cross-species transfer。
- **为什么能形成能力差异：** 检验作物特异调控语法；NT-v2不含植物，AgroNT是强植物短上下文基线，PlantCAD2/PlantCaduceus是强植物长上下文基线。
- **预注册成功规则：** 通用任务对最强植物基线非劣（AUPRC差CI下界>-0.02）；作物长程子集优效>=+0.02。
- **禁止越界的结论：** 开放染色质不等于增强子，也不等于目标基因因果调控。

### 8. CROP-PANGENOME-SV（P2_breeding_relevance）

**要回答的问题。** 能否从成对单倍型长序列预测基因PAV/SV类别及其转录影响。

- **标签与输出：** paired structural variant classification + expression delta。
- **候选公开来源：** 水稻/玉米/大豆/小麦公开pan-genome和配套表达；exact release/crosswalk TBD。
- **当前数据状态：** not frozen; no current formal result。
- **输入窗口：** 32768;131072;262144 bp。
- **最低计划规模（不是现有样本数）：** train=30000 locus pairs，validation=5000 pairs，test=5000 pairs。若真实可用数据达不到该门槛，应缩小声明或停止任务，不能用复制样本凑数。
- **最高层split与去泄漏：** locus和haplotype group隔离；群体/品种最高层bootstrap；reference bias audit。
- **主指标：** macro-F1; delta-expression Pearson; top-k locus ranking。
- **为什么能形成能力差异：** 复杂SV无法由单个512-bp窗口完整描述；长植物模型仍作为公平强基线。
- **预注册成功规则：** 对最强native长模型达到预注册MMED并在非参考单倍型子集保持。
- **禁止越界的结论：** PAV/SV预测不等于农艺性状因果。

### 9. CROP-QTL-VAR-RANK（P2_requires_curated_labels）

**要回答的问题。** 能否在QTL/GWAS区间内排序有实验或精细定位支持的候选基因/变异。

- **标签与输出：** locus-wise ranking; ref-alt delta scoring。
- **候选公开来源：** 公开作物GWAS/QTL/fine-mapping与功能验证registry；exact version and gene-ID crosswalk TBD。
- **当前数据状态：** currently blocked; missing locked labels/crosswalk。
- **输入窗口：** 8192;65536;131072 bp。
- **最低计划规模（不是现有样本数）：** train=at least 500 independent loci，validation=at least 100 loci，test=at least 100 loci。若真实可用数据达不到该门槛，应缩小声明或停止任务，不能用复制样本凑数。
- **最高层split与去泄漏：** 整个位点和study同split；按染色体/群体隔离；禁止nearest-gene伪标签。
- **主指标：** MRR; hit@1/5/10; locus-level AUCPR; bootstrap by locus。
- **为什么能形成能力差异：** 这是育种决策任务，不仅是通用序列分类；所有基础模型还必须与距离、功能注释和传统精细定位强基线比较。
- **预注册成功规则：** 对最佳序列基础模型和最佳非深度强基线均达到预注册MMED；否则只报告诊断价值。
- **禁止越界的结论：** 候选排序不能宣称发现因果基因；必须保留实验验证需求。


## 19. 如何做到“通用任务媲美、作物专属任务超过”而不作弊

### 19.1 三条并列赛道

1. **Common native track：** 512/2K/6K等所有模型都能无截断处理的通用任务。目标是对最强植物基线非劣，而不是只赢random-init。
2. **Long native track：** 32K/64K/128K/256K完整输入。只有能原生读取全部序列的模型进入；其他模型标N/E，不记0分。
3. **Compute-matched chunked track：** 对AgroNT、NT-v2 500M等短模型，把同一长序列切块，冻结统一的attention/mean aggregator，并匹配总读取碱基量与probe参数。它不能等同原生长程，但能回答“多块证据汇总是否足够”。

### 19.2 必加基线

- 植物专属：AgroNT 1B、PlantCAD2 Small、PlantCaduceus l32。
- 通用规模：NT-v2 500M、Evo2 1B。
- 通用长程：HyenaDNA 160K、Caduceus 131K。
- 短序列常见：DNABERT2、NT-v2 100M。
- 非基础模型强基线：1/3/6-mer、GC/组成、中心motif、CNN/ResNet、任务领域经典模型、QTL距离/功能注释模型。
- 消融：CropGenome-FM random-init、无RC、无region、无attention dilation、不同context。

### 19.3 预注册成功标准

- **通用任务非劣：** 本模型减最强植物基线的95% CI下界大于−0.02 AUPRC或−0.03 Pearson。
- **作物专属优效：** 对同赛道最强baseline至少+0.02 AUPRC/F1或+0.03 Pearson，且group-bootstrap 95% CI下界>0。
- **上下文因果：** 64K必须显著优于相同样本的2K/8K；远端shuffle或替换后增益应消失。
- **预训练有效：** 必须超过同架构random-init；否则不能把优势归因于预训练。
- **多seed：** 至少5个probe/fine-tune seeds；报告mean、SD、95% CI和最高单seed，不能用最高单seed冒充稳定值。
- **多任务校正：** 主要claim family内使用Holm校正；effect size和CI优先于只报p值。

## 20. 推荐的下一版预训练设计

### 20.1 先冻结benchmark，再训练

先完成P0 `CROP-LONGGENE-SEG`和`CROP-DISTAL-CIS-PAIR`的数据合同、baseline runtime与validation-only pilot；同时冻结NT-v2 500M。只有当这些任务能真实区分local shortcut和long-context能力时，再投入64K/128K/256K正式训练。

### 20.2 单一连续mixed-context run

不建议继续把每个context当成重置optimizer的新阶段。下一版建议保持单一optimizer和连续学习率，在每个训练周期按token比例混合1K、8K、32K、64K、128K、256K。一个待preflight的初始token配比是10%/25%/25%/20%/15%/5%；最终比例必须由2–3×H20 96GB真实吞吐、OOM边界和validation收益冻结，不能凭文档直接执行。

这样做的目的：短窗口维持局部motif和高吞吐，长窗口逐步提供远端结构；所有loss按有效token/label count正确加权，避免128K样本因batch小而被低估。

### 20.3 推荐新增预训练目标

- 保留MLM、RC一致性和窗口region分类。
- 新增token-level region segmentation，但只使用train assembly标签。
- 新增到TSS/TES/splice/TE边界的距离或边界分类。
- 新增同一locus多context表示一致性，且用global token保留远端增益。
- ref/alt或haplotype对比目标只在不会污染formal variant test的独立数据上使用。

### 20.4 训练预算和停止规则

369.5M参数模型当前step14000约10 token/参数。下一版可把12–20B有效训练token作为初始计算预算范围，而不是强制“1 epoch”；最终由固定validation selection loss、P0 validation task和吞吐成本共同决定。正式test不得用于早停。

### 20.5 必要Gate

1. 每个长度都做真实forward/backward/optimizer step。
2. 变长梯度与token权重和单卡参考一致。
3. checkpoint原子保存、optimizer/step/RNG精确恢复。
4. 无静默truncation；每个模型、每个任务输出eligible/excluded审计。
5. 训练数据与formal test做exact hash、MinHash/15-mer和gene-family三层去泄漏。
6. 64K/128K/256K下游embedding吞吐可完成，不能只有预训练能跑而评估跑不了。

## 21. 统计分析和test使用规则

- 分类按独立species/locus/study group bootstrap，而不是把相邻窗口当独立重复。
- 表达任务同时报告macro Pearson及per-output分布；用Fisher-z或group bootstrap构造CI。
- 配对context比较使用同一sample ID和相同split；对每个sample做paired bootstrap。
- 多倍体任务以homoeolog group为抽样单元；pan-genome任务以locus/haplotype group为抽样单元。
- 模型/超参数在validation冻结后，formal test只运行一次主分析；必要的错误修复必须版本化新release，不能覆盖旧结果。
- 失败任务也进入结果表；不能只发表本模型获胜的子集。

## 22. 执行优先级与停止条件

### Phase 0：文档和基线补齐

- 冻结本报告、source data和图。
- 下载并hash锁定NT-v2 500M；先跑512/2K common track。
- 修正README状态只能在重新生成final manifest后进行；本次不改已锁定README。

### Phase 1：P0数据与验证面板

- 从现有FASTA/GFF构建`CROP-LONGGENE-SEG`。
- 用现有木薯PRO-seq＋可冻结结构label构建`CROP-DISTAL-CIS-PAIR`pilot。
- 先跑传统baseline和现有step14000/PlantCAD2/PlantCaduceus，确认任务不是GC/motif捷径。

### Phase 2：64K训练决策

只有当P0 validation证明：a) 远端context真正有信息；b) 当前模型对长任务有可改善空间；c) H20吞吐可接受，才启动continuous mixed-context正式run。否则停止盲目长训练，优先修数据或架构。

### Phase 3：P1作物专属任务

逐项冻结Iso-Seq、polyploid、TE/SV、NLR、ATAC/PRO-seq release和accession；任何缺可信标签的任务转为blocked，不用推测数据填充。

### Phase 4：一次性formal test与论文

validation锁定模型、任务和统计脚本后再跑test；主文同时展示通用非劣和作物专属优效，并保留负结果。

## 23. 可复现性、source data和GitHub交付

本目录只提交报告、聚合source-data、构图/构文档脚本和图，不提交原始FASTA/GFF、checkpoint、embedding或大结果目录。聚合表包括：

- `source_data/assembly_species_summary.tsv`
- `source_data/pretraining_stage_summary.tsv`
- `source_data/model_architecture_parameters.tsv`
- `source_data/pretraining_objectives.tsv`
- `source_data/core_task_summary.tsv`
- `source_data/core_primary_metrics.tsv`
- `source_data/external_task_summary.tsv`
- `source_data/external_primary_metrics.tsv`
- `source_data/aggregate_primary_metrics.tsv`
- `source_data/core_auprc_tie_audit_summary.tsv`
- `source_data/baseline_registry.tsv`
- `source_data/future_task_registry.tsv`
- `source_data/public_resource_registry.tsv`

`core_primary_metrics.tsv`包含全部核心正式任务×模型×context主指标；`external_primary_metrics.tsv`包含全部外部正式任务×模型×context主指标。报告中的“全部性能”可由这两个表重新生成。

## 24. 公开资源与冻结版本

| 资源 | 类型 | 公开URL | 冻结revision/accession | 当前用途 | 状态 |
| --- | --- | --- | --- | --- | --- |
| NCBI_Assembly | pretraining raw genomes/annotations | [https://www.ncbi.nlm.nih.gov/datasets/genome/](https://www.ncbi.nlm.nih.gov/datasets/genome/) | 258 assembly accessions listed in local canonical manifest | FASTA/GFF source for pretraining and core benchmark | used |
| Plant_Genomic_Benchmark | external downstream dataset | [https://huggingface.co/datasets/InstaDeepAI/plant-genomic-benchmark](https://huggingface.co/datasets/InstaDeepAI/plant-genomic-benchmark) | 78ec8156c2ffb3e5475277fdb7eb603294224e53 | 7 external tasks; 33 frozen artifacts | used |
| AgroNT_1B | plant baseline model | [https://huggingface.co/InstaDeepAI/agro-nucleotide-transformer-1b](https://huggingface.co/InstaDeepAI/agro-nucleotide-transformer-1b) | b0e1ea1f53a2bf5bb29f8eab7a7e553bf06c1ab1 | formal baseline | used |
| PlantCAD2_Small | plant baseline model | [https://huggingface.co/kuleshov-group/PlantCAD2-Small-l24-d0768](https://huggingface.co/kuleshov-group/PlantCAD2-Small-l24-d0768) | f756c255cb76e9f538c3acec04acf4214ed03fb3 | formal baseline | used |
| PlantCaduceus_l32 | plant baseline model | [https://huggingface.co/kuleshov-group/PlantCaduceus_l32](https://huggingface.co/kuleshov-group/PlantCaduceus_l32) | e624c13c3d35415348b854c87a218893b23564f7 | formal baseline | used |
| NTv2_100M_multi_species | general baseline model | [https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-100m-multi-species](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-100m-multi-species) | f34324c6fde36a4f635f0f1f06cac5d25acd6798 | formal baseline | used |
| NTv2_500M_multi_species | required next-release general baseline | [https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species) | 06615c1660c892fc199840c18123f8385b3542a8 inspected; weights not locally frozen | next benchmark release | planned_not_evaluated |
| DNABERT2_117M | general baseline model | [https://huggingface.co/zhihan1996/DNABERT-2-117M](https://huggingface.co/zhihan1996/DNABERT-2-117M) | 7bce263b15377fc15361f52cfab88f8b586abda0 | formal 512-bp baseline | used |
| HyenaDNA_medium_160k | long-context baseline model | [https://huggingface.co/LongSafari/hyenadna-medium-160k-seqlen-hf](https://huggingface.co/LongSafari/hyenadna-medium-160k-seqlen-hf) | 7ebf71773d22c0ede2cc55cb2be15ee8c289e1ce | formal baseline | used |
| Caduceus_PS_131k | RC-equivariant long-context baseline | [https://huggingface.co/kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16](https://huggingface.co/kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16) | d89eeb853136ea64da7feb3d0c8e909771b17ae6 | formal baseline | used |
| Evo2_1B_base | evolution-scale general baseline | [https://huggingface.co/arcinstitute/evo2_1b_base](https://huggingface.co/arcinstitute/evo2_1b_base) | 2279e1df422c991037470302360edd40d0d2ea1e | formal baseline under frozen local runtime | used |

## 25. 结论

当前最稳健结论不是“已经得到完整超长作物基因组大模型”，而是：

1. 已得到一个真实训练的369.5M、8K作物基座；预训练相对同架构random-init有明确收益。
2. 在核心结构任务的2K和8K宏平均上，本模型超过当前已评估公开基线；512 bp略低于PlantCAD2/PlantCaduceus。
3. 在外部植物表达和lncRNA任务上，PlantCAD2/PlantCaduceus仍总体更强；这是真实差距。
4. Stage C1仅是可运行的64K架构＋569步partial run，没有64K科学结果。
5. 下一步不应先盲目延长训练，而应先冻结NT-v2 500M和P0作物长程任务；然后用预注册的common/native-long/chunked三赛道决定是否训练和是否真正超过基线。
6. “我们的模型能解决而短模型不能原生解决”的任务必须由输入信息范围客观定义；“超过PlantCAD2/PlantCaduceus”必须靠结果，而不是靠排除它们。

# 附录A：258个assembly的30物种组成

| 物种 | 属 | assembly数 | train | validation | test | GenBank | RefSeq | 压缩genome bytes | 压缩annotation bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arachis hypogaea | Arachis | 9 | 7 | 1 | 1 | 8 | 1 | 6,885,295,295 | 130,378,480 |
| Beta vulgaris | Beta | 2 | 1 | 1 | 0 | 1 | 1 | 333,376,382 | 14,467,706 |
| Brassica napus | Brassica | 5 | 4 | 1 | 0 | 4 | 1 | 1,342,153,515 | 86,582,074 |
| Brassica oleracea | Brassica | 4 | 3 | 0 | 1 | 3 | 1 | 667,451,826 | 40,503,375 |
| Brassica rapa | Brassica | 6 | 4 | 1 | 1 | 5 | 1 | 622,896,641 | 51,784,078 |
| Cicer arietinum | Cicer | 1 | 1 | 0 | 0 | 0 | 1 | 144,900,890 | 9,134,001 |
| Citrullus lanatus | Citrullus | 1 | 1 | 0 | 0 | 1 | 0 | 113,766,894 | 4,010,033 |
| Cucumis melo | Cucumis | 6 | 5 | 1 | 0 | 5 | 1 | 681,906,757 | 37,721,297 |
| Cucumis sativus | Cucumis | 1 | 0 | 0 | 1 | 0 | 1 | 71,224,682 | 7,763,605 |
| Daucus carota | Daucus | 3 | 1 | 1 | 1 | 2 | 1 | 386,240,517 | 22,860,613 |
| Glycine max | Glycine | 10 | 8 | 1 | 1 | 9 | 1 | 2,998,714,788 | 128,096,637 |
| Gossypium hirsutum | Gossypium | 3 | 1 | 1 | 1 | 2 | 1 | 2,118,018,201 | 71,872,818 |
| Helianthus annuus | Helianthus | 9 | 7 | 1 | 1 | 8 | 1 | 8,097,097,230 | 134,318,644 |
| Hordeum vulgare | Hordeum | 83 | 65 | 9 | 9 | 82 | 1 | 106,304,970,965 | 975,123,353 |
| Lactuca sativa | Lactuca | 1 | 1 | 0 | 0 | 0 | 1 | 788,072,127 | 13,692,789 |
| Malus domestica | Malus | 42 | 32 | 5 | 5 | 41 | 1 | 8,472,831,162 | 386,497,444 |
| Manihot esculenta | Manihot | 1 | 1 | 0 | 0 | 0 | 1 | 188,402,359 | 11,770,716 |
| Musa acuminata | Musa | 4 | 2 | 1 | 1 | 3 | 1 | 902,299,218 | 51,608,365 |
| Oryza sativa | Oryza | 23 | 17 | 3 | 3 | 22 | 1 | 2,723,321,219 | 180,672,119 |
| Phaseolus vulgaris | Phaseolus | 2 | 1 | 1 | 0 | 1 | 1 | 336,325,359 | 27,785,356 |
| Prunus persica | Prunus | 2 | 1 | 1 | 0 | 1 | 1 | 132,852,900 | 7,715,551 |
| Saccharum spontaneum | Saccharum | 1 | 1 | 0 | 0 | 1 | 0 | 957,727,589 | 14,066,694 |
| Setaria italica | Setaria | 1 | 1 | 0 | 0 | 0 | 1 | 127,656,729 | 8,636,775 |
| Solanum lycopersicum | Solanum | 3 | 3 | 0 | 0 | 2 | 1 | 678,622,437 | 26,897,442 |
| Solanum tuberosum | Solanum | 6 | 4 | 1 | 1 | 5 | 1 | 1,385,439,236 | 37,647,242 |
| Sorghum bicolor | Sorghum | 2 | 1 | 1 | 0 | 1 | 1 | 417,804,834 | 16,534,260 |
| Triticum aestivum | Triticum | 16 | 12 | 2 | 2 | 15 | 1 | 60,270,225,085 | 218,375,879 |
| Vigna radiata | Vigna | 1 | 1 | 0 | 0 | 0 | 1 | 137,333,161 | 9,760,965 |
| Vitis vinifera | Vitis | 3 | 1 | 1 | 1 | 2 | 1 | 564,473,863 | 35,985,572 |
| Zea mays | Zea | 7 | 5 | 1 | 1 | 6 | 1 | 4,632,238,818 | 78,654,231 |

# 附录B：当前证据文件角色

- **正式结果入口：** `PUBLICATION_V2_FINAL_RESULTS_CN.md`和两个run的`summary/full_data_metrics.tsv`。
- **Stage B冻结依据：** `early_stopping_state.json`与`stage_B_8k_final_manifest.json`。
- **Stage C1真实终态：** `train_step14000_64k_corrected.log`和`interrupted_step_00000569.pt`。
- **本报告不修改已被publication-v2 final manifest锁定的README和协议文件。** 若未来更新入口文档，必须重新生成manifest并明确新release。

# 附录C：指标怎么读

- AUPRC=1最好；恒定分数的Average Precision应等于正例比例。核心任务正负1:1，因此标准值应为0.5。旧formal evaluator对并列分数处理不正确，导致`majority_baseline=0.30734`；主学习模型排名不变，但下一release必须用tie-safe实现重算，详见11.2节。
- Pearson=1表示完全正线性，0表示无线性关系，负值表示反向。基因表达序列预测通常受组织、环境和转调控影响，0.5左右已经代表显著但不完整的cis sequence signal。
- 宏平均先对每个任务/输出维度独立计分再等权平均，避免样本多的任务完全支配结果；它也可能隐藏某个关键任务失败，所以本报告同时列出每项任务。
- N/E不是失败分数，而是该模型在冻结的zero-truncation协议下不具备完整输入能力或未被纳入。
