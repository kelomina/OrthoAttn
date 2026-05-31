# WikiText-103 PPL 对比报告

**生成时间**: 2026-05-11 05:33:47

## 数据集与实验设置

- **数据集**: WikiText-103 完整三切分（HuggingFace datasets 加载）
  - Train: 1,801,350 行 / 518 MB
  - Validation: 3,760 行 / 1.1 MB
  - Test: 4,358 行 / 1.2 MB
- **分词器**: 字符级 `CharTokenizer`（词表 96，含 4 个特殊 token）
- **设备**: NVIDIA GeForce RTX 4070 Laptop GPU, 8GB VRAM
- **优化器**: AdamW (lr=3e-4, betas=(0.9, 0.95))
- **学习率策略**: CosineAnnealingLR, T_max=max_steps
- **梯度裁剪**: clip_grad_norm=1.0

### 注意事项

1. **训练步数差异**: 变体 A（标准注意力基线）训练 200K 步；变体 B-E（MHDSRA2）各训练 50K 步
   （受限于 MHDSRA2 ~200-400ms/step 的较慢步速）
2. **MHDSRA2 状态重置**: 所有 MHDSRA2 变体使用 `detach_state=True`，
   模型状态在每次 forward 调用时重置（state_list = [None]），无法跨 batch 携带上下文
3. **PPL 定义**: `perplexity = exp(cross_entropy_loss)`，在测试集上全量评估
4. **数据来源**: 变体 A 数据来自独立运行（200K 步）；变体 B-E 数据来自 `run_short_variants.py` 和 `run_de.py`（50K 步）

---

## 结果汇总

| 变体 | 模型 | dim | layers | slots | chunk | 参数量 | batch | 步数 | Test PPL | Best Valid PPL | 训练时间 | 步速 |
|------|------|-----|--------|-------|-------|--------|-------|------|----------|----------------|----------|------|
| A | Standard Attention (baseline) | 512 | 6 | N/A | N/A | 19,001,440 | 4 | 200,000 | 2.48 | 2.45 | 3305s | 17ms |
| B | MHDSRA2 (default) | 512 | 6 | 128 | 128 | 19,790,980 | 8 | 50,000 | 10.96 | 10.87 | 12145s | 243ms |
| C | MHDSRA2 (large slots) | 512 | 6 | 256 | 128 | 20,577,412 | 8 | 50,000 | 11.30 | 11.22 | 12927s | 259ms |
| D | MHDSRA2 (deep, 12 layers) | 512 | 12 | 128 | 128 | 39,482,536 | 4 | 50,000 | 11.28 | 11.20 | 21612s | 432ms |
| E | MHDSRA2 (small chunk) | 512 | 6 | 64 | 64 | 19,397,764 | 8 | 50,000 | 11.28 | 11.20 | 20866s | 417ms |

---

## 结果分析

### 1. 标准注意力（基线）表现优异

A（Standard Attention, dim=512, L=6）在 200K 步后达到:
- Test PPL: **2.48**
- 训练时间: 55 分钟（~16ms/step）
- 参数量: 19.0M

这表明标准因果注意力 + FlashAttention 在字符级 WT-103 上能够高效建模。

### 2. MHDSRA2 PPL 显著高于基线 (~4.4x)

| 变体 | 配置 | Test PPL | 与 A 比值 | 与 B 差距 |
|------|------|----------|-----------|-----------|
| A | Standard Attention | 2.48 | 1.00x | - |
| B | MHDSRA2 (slots=128, chunk=128) | 10.96 | 4.42x | - |
| C | MHDSRA2 (slots=256, chunk=128) | 11.30 | 4.56x | +0.34 |
| D | MHDSRA2 (slots=128, chunk=128, L=12) | 11.28 | 4.55x | +0.32 |
| E | MHDSRA2 (slots=64, chunk=64) | 11.28 | 4.55x | +0.32 |

核心原因: MHDSRA2 的 `detach_state=True` + `state_list = [None]` 导致模型
无法在 batch 之间传递状态，每个 forward 都从零状态开始。这限制了模型利用
上下文信息的能力。相比之下，标准注意力可以在 seq_len=512 窗口内自由建模依赖。

### 3. MHDSRA2 变体间差异较小

B-E 的 PPL 范围为 10.96~11.30，变体间最大差距 0.34:
- 增大 slots（128→256）未带来收益 → 当前训练条件下 128 slots 已饱和
- 加深层数（6→12）参数量翻倍但 PPL 未改善 → 瓶颈不在容量
- 缩小 chunk（128→64）未改善 PPL → chunk_size 不是当前瓶颈

### 4. 效率对比

| 指标 | A (Standard) | B (MHDSRA2) | C (MHDSRA2) | D (MHDSRA2, L=12) | E (MHDSRA2) |
|------|-------------|-------------|-------------|-------------------|-------------|
| 步速 | 16ms | 243ms | 259ms | 432ms | 417ms |
| 50K 步耗时 | 13min | 3.4h | 3.6h | 6.0h | 5.8h |
| 参数量 | 19.0M | 19.8M | 20.6M | 39.5M | 19.4M |

MHDSRA2 步速为标准注意力的 15-27x。主要计算开销来自:
- 分块流式前向中的状态更新
- top-k 路由计算
- 全连接 FFN 的双线性层

---

## 收敛曲线

### A - Standard Attention (baseline)

```
  Step   5000: Valid PPL=3.73
  Step  10000: Valid PPL=3.32
  Step  15000: Valid PPL=3.16
  Step  20000: Valid PPL=3.06
  Step  25000: Valid PPL=3.00
  Step  30000: Valid PPL=2.94
  Step  35000: Valid PPL=2.89
  Step  40000: Valid PPL=2.85
  Step  45000: Valid PPL=2.82
  Step  50000: Valid PPL=2.79
  Step  55000: Valid PPL=2.76
  Step  60000: Valid PPL=2.75
  Step  65000: Valid PPL=2.72
  Step  70000: Valid PPL=2.70
  Step  75000: Valid PPL=2.68
  ...
  Step 200000: Valid PPL=2.45 (final)
```

### B - MHDSRA2 (default)

```
  Step   5000: Valid PPL=11.45
  Step  10000: Valid PPL=11.79
  Step  15000: Valid PPL=11.38
  Step  20000: Valid PPL=11.34
  Step  25000: Valid PPL=11.27
  Step  30000: Valid PPL=11.23
  Step  35000: Valid PPL=11.03
  Step  40000: Valid PPL=10.94
  Step  45000: Valid PPL=10.89
  Step  50000: Valid PPL=10.87
```

### C - MHDSRA2 (large slots)

```
  Step   5000: Valid PPL=11.45
  Step  10000: Valid PPL=11.47
  Step  15000: Valid PPL=11.42
  Step  20000: Valid PPL=11.39
  Step  25000: Valid PPL=11.36
  Step  30000: Valid PPL=11.32
  Step  35000: Valid PPL=11.29
  Step  40000: Valid PPL=11.24
  Step  45000: Valid PPL=11.23
  Step  50000: Valid PPL=11.22
```

### D - MHDSRA2 (deep, 12 layers)

```
  Step   5000: Valid PPL=11.48
  Step  10000: Valid PPL=11.42
  Step  15000: Valid PPL=11.40
  Step  20000: Valid PPL=11.42
  Step  25000: Valid PPL=11.33
  Step  30000: Valid PPL=11.34
  Step  35000: Valid PPL=11.26
  Step  40000: Valid PPL=11.23
  Step  45000: Valid PPL=11.21
  Step  50000: Valid PPL=11.20
```

### E - MHDSRA2 (small chunk)

```
  Step   5000: Valid PPL=11.46
  Step  10000: Valid PPL=11.42
  Step  15000: Valid PPL=11.36
  Step  20000: Valid PPL=11.32
  Step  25000: Valid PPL=11.31
  Step  30000: Valid PPL=11.27
  Step  35000: Valid PPL=11.26
  Step  40000: Valid PPL=11.23
  Step  45000: Valid PPL=11.20
  Step  50000: Valid PPL=11.20
```

---

## 结论与建议

1. **标准注意力在字符级 LM 任务上显著优于 MHDSRA2**（PPL 2.48 vs ~11）
2. **MHDSRA2 需要 stateful 训练**才能展现其流式处理优势——当前 `detach_state=True` + `state_list=[None]`
   的设置相当于每次前向都从零开始，无法利用跨 batch 的长期依赖
3. **瓶颈不在模型容量**——增加 slots 或层数未改善 PPL，说明瓶颈在训练策略
4. **建议的下一步改进方向**:
   - 实现跨 batch 状态传递（stateful training loop），如非 shuffle 顺序训练 + 手动 state carry-over
   - 开启 `momentum_qkv=True` 稳定 slot 读取
   - 增大 d_model 或使用 `use_retrieval=True` 启用外部记忆
   - 优化 top-k 操作和状态更新效率