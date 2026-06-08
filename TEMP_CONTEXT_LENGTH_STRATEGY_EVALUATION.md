# 临时评估: 多 context 长度混合输入 vs 当前渐进式继续训练

更新时间: 2026-06-08 20:47:56 CST

## Material Passport

- 类型: 临时技术评估文档
- 目标: 评估 CropGenome-FM 预训练时应采用多长度混合输入，还是当前 8K -> 64K -> 128K -> 256K 的渐进式继续训练
- 结论状态: 建议采用“渐进式继续训练 + 每阶段内部少量多长度混合”的折中方案
- 注意: 本文档为临时评估，不改动正式计划；用户确认后再合并到 `PROJECT_PLAN.md`

## 1. 结论

不建议把 8K、64K、128K、256K 四种长度从一开始完全混在一起等比例训练。更推荐保留当前计划的主框架: 同一个模型从 8K 训练起步，然后加载 checkpoint 继续到 64K、128K、256K。

但当前计划可以优化: 每个长 context 阶段内部加入少量短 context batch，形成“主长度为主、短长度保留”的课程学习。这样兼顾训练稳定性、吞吐、短程功能位点能力和长程调控能力。

最终推荐:

| 阶段 | 主训练长度 | 建议 batch 长度组成 | 目的 |
|---|---:|---|---|
| Stage B | 8K | 70% 8K + 20% 4K + 10% 16K warm-up | 高吞吐学习 motif、CDS、splice、TSS/TES |
| Stage C1 | 64K | 70% 64K + 15% 8K + 10% 16K/32K + 5% 4K | 扩展 gene body、promoter-gene、长 intron，同时防止短程能力退化 |
| Stage C2 | 128K | 75% 128K + 15% 64K + 10% 8K/16K | 学远端调控和结构上下文，保持 8K probe 表现 |
| Stage D | 256K | 80% 256K + 15% 128K + 5% 8K/64K | 资源允许时做长上下文 midtraining，不牺牲核心功能区能力 |

## 2. 为什么不建议一开始完全多长度混合

完全混合指 8K、64K、128K、256K 在同一训练阶段随机混入，甚至比例接近。这种方案表面上更全面，但对当前作物模型不划算。

主要问题:

- 训练吞吐下降明显。长序列 batch 会显著拉低整体 tokens/s，尤其 128K/256K 对显存、通信和激活保存压力很大。
- batch token 数难稳定。不同长度混合会导致每步有效 token、mask token、区域比例波动，优化器看到的 batch 分布更不稳定。
- 早期模型还不会基本 DNA 语法时，直接输入超长片段收益低。64K/128K 的远端关系需要建立在 motif、剪接、CDS、TSS 等局部语法已经学好的基础上。
- 作物数据强区域采样下，长片段容易被 intron/intergenic 背景主导，早期会稀释 CDS/splice/start/stop 的学习信号。
- 256K 阶段如果太早混入，会明显增加计算成本，但对早期 loss 下降贡献不一定高。

因此，完全混合更适合资源极大、数据极大、训练代码已高度工程化的情况。当前项目更需要可控、稳定、可复现。

## 3. 当前渐进式继续训练的优点

当前计划的 8K -> 64K -> 128K -> 256K 是同一个模型继续训练，不是四个独立模型。

优点:

- 先用 8K 高吞吐学习局部功能语法，计算性价比最高。
- 64K/128K 阶段从已有 checkpoint 扩展上下文，模型已经有基本 motif/CDS/splice 表征，长程训练更有效。
- 每个阶段的显存、batch size、gradient accumulation、checkpoint 策略更容易估算。
- 每个阶段都能做独立评测，判断是否值得进入下一阶段。
- 与 HyenaDNA、Evo/Evo 2 这类长上下文基因组模型的训练思想更一致: 使用支持长上下文的架构，在不同长度上逐步扩展能力，而不是一开始把所有长度无控制混合。

## 4. 当前方案的不足

如果严格只在每个阶段使用单一长度，也有风险:

- 到 64K/128K 以后，短程任务如 splice、start/stop、CDS frame 的小窗口性能可能轻微退化。
- 长窗口里功能位点占比可能下降，导致局部功能区域 loss 被长背景稀释。
- 模型可能过度适应某一个固定 context length，推理时面对不同长度窗口时表现不够平滑。
- 训练数据中不同区域的窗口自然长度不同，强制单长度会带来 padding 或截断浪费。

所以不建议“纯单长度阶段训练”，而是建议“阶段主长度 + 少量辅助长度混合”。

## 5. 推荐方案: 阶段式多长度课程学习

### 5.1 训练原则

- 一个阶段只设置一个主 context length。
- 每个阶段保留少量短 context batch，用于维持局部功能区能力。
- 长 context 阶段不等比例混入所有长度，而是偏向当前阶段目标长度。
- 按 token 数统计比例，不按 batch 数统计比例。否则短序列 batch 会被过度采样。
- 每个长度桶内部继续执行区域采样比例和 loss 权重。
- validation 每个阶段必须同时评估 8K、64K、128K 对应 probe，不能只看当前 context 的 val loss。

### 5.2 Stage B: 8K 主训练

目标: 先建立稳定的单碱基语法、局部 motif、CDS、splice、TSS/TES 能力。

建议长度组成:

- 70% tokens: 8K。
- 20% tokens: 4K，用于高质量 CDS/splice/start/stop 密集窗口。
- 10% tokens: 16K warm-up，用于让位置/SSM 状态提前接触略长窗口。

不建议此阶段混入 64K 以上长窗口，除非 GPU 利用率和 loss 稳定性已经验证。

### 5.3 Stage C1: 64K 继续训练

目标: 从 Stage B checkpoint 继续学习 gene body、promoter-gene、长 intron、近端调控。

建议长度组成:

- 70% tokens: 64K。
- 15% tokens: 8K，维持 CDS/splice/TSS/TES 能力。
- 10% tokens: 16K/32K，作为过渡长度。
- 5% tokens: 4K，给关键监督窗口保底。

这一阶段是最重要的长上下文扩展阶段。若资源不足，宁可把 C2 推迟，也要把 C1 做扎实。

### 5.4 Stage C2: 128K 继续训练

目标: 远端调控、gene cluster、长 intron、复杂结构上下文。

建议长度组成:

- 75% tokens: 128K。
- 15% tokens: 64K。
- 10% tokens: 8K/16K。

128K 阶段不要继续大量混入 4K，否则会浪费长上下文预算。短程能力主要通过 8K probe 和少量功能区 batch 维持。

### 5.5 Stage D: 256K midtraining

目标: 资源允许时扩展更长结构上下文，作为 midtraining，不作为第一版必须完成条件。

建议长度组成:

- 80% tokens: 256K。
- 15% tokens: 128K。
- 5% tokens: 8K/64K。

Stage D 的评估必须看下游任务是否真正受益。如果 256K val loss 下降但 splice/TSS/variant probe 没有收益，不应继续加大 256K token 预算。

## 6. 对模型和数据输入的具体影响

### 6.1 Sampler

需要把 sampler 改成两级抽样:

1. 先抽 `context_bucket`: 4K、8K、16K、32K、64K、128K、256K。
2. 再在该长度桶内按区域采样比例抽 `region_bucket`: CDS、splice、TSS、UTR、TES、intron、intergenic、background。

这样可以避免长 context 阶段被 intergenic 背景吞掉，也避免短 context 过量占据 token 预算。

### 6.2 Batch 组织

推荐每个 optimizer step 使用同长度 micro-batch，不建议一个 micro-batch 内混放不同长度。

原因:

- 同长度 micro-batch padding 最少。
- activation checkpointing、FlashAttention 或 SSM kernel 更稳定。
- tokens/s 和显存估算更准确。

如果要混合不同长度，应在 gradient accumulation 级别混合，而不是在同一个 micro-batch 内混合。

### 6.3 Loss 权重

长 context 阶段需要继续保留区域加权:

```text
L_total =
  1.00 * L_region_weighted_MLM
  + 0.10 * L_causal_next_token
  + 0.05 * L_reverse_complement_consistency
```

建议额外记录每个 context_bucket 的 loss:

- `loss_4k`
- `loss_8k`
- `loss_64k`
- `loss_128k`
- `loss_256k`

否则总 loss 可能掩盖短程能力退化。

## 7. 资源影响评估

相比当前“每阶段单长度”的计划，阶段式多长度课程学习会增加 sampler 和日志复杂度，但不会显著增加磁盘占用，因为仍然使用在线采样/tokenization。

预计变化:

| 项目 | 单长度阶段 | 阶段式多长度课程学习 | 影响 |
|---|---|---|---|
| 训练服务器磁盘 | 仍按 800GB-1.5TB 规划 | 基本不变 | 只增加少量 bucket index/config |
| GPU 显存 | 按主 context length 估算 | 基本按主 context length 估算 | micro-batch 同长度时可控 |
| CPU/RAM | online sampler 正常 | 略增 | 多一个 context_bucket 抽样和统计 |
| 训练时间 | 当前估算 | 增加约 3%-8% | 主要来自更多 validation bucket 和调度开销 |
| 工程复杂度 | 中等 | 中等偏高 | 需要更严格记录每桶 token 和 loss |

结论: 增加的工程复杂度值得接受，因为它降低长阶段遗忘短程功能的风险。

## 8. 评估指标

每个阶段结束后至少检查:

| 指标 | 目的 | 进入下一阶段条件 |
|---|---|---|
| `val_loss_8k` | 局部语法是否稳定 | 不能比上一阶段明显变差 |
| `val_loss_current_context` | 当前长度是否学到东西 | 持续下降或达到平台期 |
| splice donor/acceptor AUROC/AUPRC | 核心短程任务 | C1/C2 后不能下降超过 1%-2% |
| CDS frame/start-stop probe | 编码区语法 | C1/C2 后不能明显下降 |
| TSS/TES probe | 中程调控 | C1 后应有提升 |
| long intron/promoter-gene probe | 长程目标 | C1/C2 应优于 Stage B |
| RC consistency | 双链一致性 | 长阶段不能退化 |
| tokens/s 和 GPU memory | 工程可训练性 | 达到可持续训练吞吐 |

如果 C1/C2 的长上下文 loss 下降，但 8K 下游 probe 大幅退化，应提高短 context replay 比例。

## 9. 最终建议

正式训练不要改成“所有长度一起随机混合”。当前阶段式继续训练方向是正确的。

建议把当前计划优化为:

```text
Stage B: 8K 主训练，少量 4K/16K
Stage C1: 从 B 继续，64K 主训练，少量 8K/16K/32K replay
Stage C2: 从 C1 继续，128K 主训练，少量 64K/8K replay
Stage D: 从 C2 继续，256K midtraining，少量 128K/8K replay
```

一句话结论: 对 CropGenome-FM，最佳方案不是“多长度一锅混”，也不是“每阶段纯单长度”，而是“同一模型渐进式扩长 + 每阶段短长度 replay”。

## 10. 参考依据

- HyenaDNA 提出单碱基长程基因组建模，并展示可训练到很长 context；其论文和模型说明强调长上下文能力与序列长度扩展策略。参考: https://arxiv.org/abs/2306.15794
- Caduceus 强调 DNA 建模需要双向长程建模和 reverse-complement equivariance，这支持当前模型在长 context 阶段继续保留 RC consistency。参考: https://arxiv.org/abs/2403.03234
- Evo 2 使用 StripedHyena 2，在单碱基分辨率下扩展到百万 token context，并强调混合架构在短序列和长序列训练效率上的优势。参考: https://www.nature.com/articles/s41586-026-10176-5
- FlashAttention/可变长度 packed sequence 训练工程经验支持“micro-batch 内同长度、gradient accumulation 层面混合长度”的组织方式，以减少 padding 和提高吞吐。参考: https://github.com/Dao-AILab/flash-attention

## 11. 临时进展记录

- 2026-06-08 20:47:56 CST: 评估多 context 长度混合输入与当前渐进式继续训练；建议采用“渐进式继续训练 + 每阶段内部短长度 replay”的折中方案，暂不改正式计划。
