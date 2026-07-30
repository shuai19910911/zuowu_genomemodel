# CropGenome-FM 模型结构

本页只描述当前Stage B 8K模型，不混入历史Stage C试验方案。

## 核心结构

```text
单碱基DNA token
  → 1024维embedding
  → 32层HyenaLite局部卷积主干
  → 每4层插入一次512 bp分块注意力
  → MLM碱基预测头
  → RC反向互补一致性约束
  → 7类区域弱监督辅助头
```

按当前`training_server_transfer/configs/model_large.json`和`training_server_transfer/scripts/train.py`实构计数：**369,505,287个参数（约3.70亿）**。

## 关键配置

| 项目 | 当前值 | 含义 |
|---|---:|---|
| 输入单位 | single base | A/C/G/T/N/MASK/PAD共7个token |
| 最大上下文 | 8192 bp | 当前正式Stage B训练长度 |
| hidden size | 1024 | 每个位置的隐藏维度 |
| 层数 | 32 | HyenaLite残差块数量 |
| 卷积核 | 127 | 局部序列混合范围 |
| 注意力间隔 | 每4层 | 共8个分块注意力层 |
| attention chunk | 512 bp | 控制显存和局部精确交互 |
| dropout | 0.05 | 正则化 |
| 参数精度 | bf16训练 | 3×A100 DDP |
| gradient checkpointing | 开启 | 用额外计算换显存 |

## 训练目标

```text
训练总目标 = MLM + 0.02 × RC consistency + 0.05 × region auxiliary
checkpoint选择 = validation MLM + 0.02 × validation RC
```

- MLM（掩码碱基预测）是主学习目标。
- RC consistency（反向互补一致性）约束正反链行为，但不能仅凭该损失宣称严格数学等变。
- region auxiliary（区域辅助头）有background、coding、gene body、promoter、splice、TES、UTR七类，只是弱监督健康信号，不是独立正式基准。

## 当前训练协议

- 从step14000完整恢复模型、优化器和步数。
- 3个DDP rank，每rank micro-batch=4、梯度累积=3，全局有效batch=36。
- 同长度桶内执行全局无放回抽样；一个coverage cycle结束后才允许再次使用窗口。
- 每500步永久保存完整checkpoint，每1000步做固定validation。
- 目标step50000，早停关闭；best checkpoint仍按validation selection loss记录。

## 下游接口

- sequence embedding：启动子、lncRNA、表达预测等序列级任务。
- token embedding：exon/intron/UTR逐碱基分割和边界任务。
- frozen encoder + probe：冻结主干后训练统一轻量头，用于公平比较。
- 后续fine-tuning必须使用同数据、同split和同调参预算，不能给本模型额外优势。

## 结论边界

当前架构的创新重点是作物专用语料、区域感知预训练和作物基准，而不是宣称全新的序列算子。训练loss下降只说明优化正常；论文结论必须来自固定下游任务、强公开模型基线、跨属划分和多seed稳定性。
