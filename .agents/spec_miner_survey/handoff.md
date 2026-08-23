# Handoff Report — Stanford Zoology MQAR Specification Miner

- **Agent**: `spec_miner_survey`
- **Date**: 2026-08-22
- **Milestone**: Milestone 1 & Specification Mining / Benchmark Alignment

---

## 1. Observation (观察事实)

1. **核心数据生成实现** (`src/dsra/domain/mqar.py`):
   - 第 76-85 行：
     ```python
     key_pool_size = max(16, min(64, (V - 1) // 4))
     val_pool_size = max(16, min(64, (V - 1) // 4))
     key_start = 1
     key_end = key_start + key_pool_size
     val_start = key_end
     val_end = val_start + val_pool_size
     filler_start = val_end
     filler_end = V
     ```
   - 第 90 行：`X = torch.randint(filler_start, filler_end, (batch_size, L), device=device, dtype=torch.long)`
   - 当 $V=32$ 时，`key_pool_size=16`, `val_pool_size=16`, `filler_start=33, filler_end=32`，执行报错：
     `RuntimeError: random_ expects 'from' to be less than 'to', but got from=33 >= to=32`。
   - 当 $V=8192, K=128$ 时，因 `min(64, ...)` 截断报错：
     `ValueError: num_kv_pairs=128 exceeds pool capacity (key_pool=64, val_pool=64)`。
   - 当 $Q > K$（如 $K=4, Q=8$）时，第 124 行 `query_perm = torch.randperm(K, device=device)[:Q]` 返回长度为 4 的张量，第 128 行循环报错：
     `IndexError: index 4 is out of bounds for dimension 0 with size 4`。
   - 当 $Q=0$ 时，第 104 行 `(L - 2 - query_half_start) // Q` 触发：
     `ZeroDivisionError: division by zero`。
   - 当 `device="cpu"` (字符串) 传入时，第 63 行 `if device.type == "cuda":` 触发：
     `AttributeError: 'str' object has no attribute 'type'`。
   - `insert_mode: str = "uniform"` 在 `MQARConfig` 中声明，但在 `generate_mqar_batch` 中完全未被使用。

2. **自回归因果对齐与损失掩码** (`src/dsra/domain/mqar.py:128-135`, `scripts/benchmark_mqar.py:53`):
   - 在序列前缀 $0 \dots L//2$，键值对 $X[b, pos]=k_i, X[b, pos+1]=v_i$ 相邻放置。
   - 在序列后缀，第 $qpos$ 处放置 $X[b, qpos] = q\_keys[i]$，$Y[b, qpos] = q\_vals[i]$。
   - 在 $X[b, qpos+1]$ 处保持为 Distractor Filler，无未来真实 Value 泄露。
   - 损失计算使用 `F.cross_entropy(..., ignore_index=0)`，除 $qpos$ 处外其余位置全部掩码。

3. **Oracle 全知探针验证** (独立运行测试):
   - 构造精确查表模型后执行前向计算：
   - 输出结果：`Oracle Loss: 0.000000`, `Oracle Accuracy: 100.00% (64/64)`。

4. **现有单元测试与基准套件** (`tests/test_mqar_data_generation.py`, `scripts/benchmark_mqar.py`):
   - `python -m pytest tests/test_mqar_data_generation.py` 3 个用例全部通过。
   - `scripts/benchmark_mqar.py` 包含完整的 `AdamW` 训练与端到端评测闭环，但尚未集成 Standard Transformer Baseline 对照。

---

## 2. Logic Chain (推理链条)

1. **词表与因果数学等价性**:
   - 观察 1 显示：Keys、Values、Fillers 的索引区间严格无交集且非零，与 Stanford Zoology `zoology.data.associative_recall` 规范一致；
   - 观察 2 证实：自回归 next-token 预测目标 $Y[qpos] = v_k$ 正确对准 $X[qpos] = k$，且 $X[qpos+1]$ 为 Filler，未出现未来 Value 泄漏，符合标准自回归因果规范；
   - 观察 3 证实：Oracle 模型在评测流程中可直接达成 100% 准确率与 0 损失，证明数据生成与评测逻辑不存在阻碍模型学习的内在数学矛盾或偏倚。

2. **边界与鲁棒性缺陷推导**:
   - 由观察 1 的 $V=32, Q > K, Q=0, \text{device}=\text{str}$ 实测报错可知，当前实现存在 4 个高危/中危崩溃点，源于 `MQARConfig.__post_init__` 校验不严密以及 `generate_mqar_batch` 中的静态硬编码计算；
   - 由观察 1 的 `insert_mode` 可知，配置中的随机插入特性尚未落地代码；
   - 由观察 4 可知，基准脚本尚未引入与 Standard Transformer 的横向对比分支（R4）。

---

## 3. Caveats (注意事项与假设)

- 本次审计为只读调查，未对源码进行任何写操作。
- 硬件环境：系统已检测到 `cuda:0` 正常可用，测试均在 CPU 及 CUDA 双设备下进行了行为验证。

---

## 4. Conclusion (结论与行动建议)

1. **规范合规性**: `src/dsra/domain/mqar.py` 的核心生成逻辑在数学本质上与 Stanford Zoology MQAR 规范高度一致，无未来信息泄露，无占位伪造。
2. **待修复事项 (清晰交付给实现 Agent)**:
   - **Fix 1**: 修复 `key_pool_size` / `val_pool_size` 的动态计算，移除导致 $V < 64$ 崩溃的 `max(16, ...)` 与导致大词表受限的 `min(64, ...)`；
   - **Fix 2**: 在 `MQARConfig.__post_init__` 增加 `1 <= num_queries <= num_kv_pairs` 校验；
   - **Fix 3**: 规范化 `device = torch.device(device)`；
   - **Fix 4**: 补齐 `insert_mode="random"` 随机插入逻辑；
   - **Fix 5**: 在 `scripts/benchmark_mqar.py` 中引入 Standard Transformer 全注意力基线对比；
   - **Fix 6**: 补充针对极值词表、Oracle 探针、设备兼容性的全套单元测试。

---

## 5. Verification Method (独立验证方法)

1. **复现崩溃边界**:
   ```bash
   python -c "import torch; from src.dsra.domain.mqar import MQARConfig, generate_mqar_batch; cfg = MQARConfig(vocab_size=32, seq_len=64, num_kv_pairs=4); generate_mqar_batch(2, cfg, torch.device('cpu'))"
   ```
2. **运行现有单测**:
   ```bash
   python -m pytest tests/test_mqar_data_generation.py -v
   ```
3. **查看详细审计报告**:
   查看 `E:/Project/python/DSRA/.agents/spec_miner_survey/report.md`。
