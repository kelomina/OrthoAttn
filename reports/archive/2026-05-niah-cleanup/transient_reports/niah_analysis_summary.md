# NIAH（Needle-In-A-Haystack）测评分析总结

## 一、任务定义

NIAH 测试模型在超长上下文中检索信息的能力。在 2M token 的随机序列（haystack）中插入一个 needle（key=token 2, value=random token），然后在之后的位置插入 query（key=token 2, query=token 1），让模型在 query 位置预测 needle 的 value。

### 数据生成

```
位置: ... N-1  N   N+1  N+2  ...  M-1  M   M+1  ...
输入: ... filler 2  val  filler ...  2    1  filler ...
监督:                     ✗            ✗   ✓
```

- `PAD_TOKEN_ID=0`, `QUERY_TOKEN_ID=1`, `NEEDLE_KEY_TOKEN_ID=2`
- `FILLER_TOKEN_START=4`，填充和 value token 在 `[FILLER_START, vocab_size)` 范围内随机
- Needle 深度支持 `0.1/0.5/0.9`，随机轮询
- Query 位置在 needle 之后至少 3 个 token

## 二、MHDSRA2 架构关键机制

### Slot Memory（核心）

每个 `MultiHeadDSRA2` 层维护 `slots` 个可读写槽位（默认 64），每个槽位有 key/value/age/usage/confidence。

**写入**（`_slot_write`，`improved_dsra_mha.py:394`）：
- 当前 chunk 所有 token 的 k/v 通过 content-based routing 分配到槽位
- `write_gate = 1 - exp(-eta × mass)` 控制写入强度
- `forget = forget_base + age_term + ...` 控制旧内容遗忘

**读取**（`_slot_read`，`improved_dsra_mha.py:284`）：
- query 的 q 通过 `softmax(q @ slot_k^T × tau)` 选择槽位
- 读取的 slot_v 作为输出

**融合**：输出 = gate_slot × slot_out + gate_local × local_out + gate_retrieval × retrieval_out

### Chunked Processing

长序列按 `chunk_size` 分块（2M 下为 1024），逐块流式处理：

```
state_list = [None]
for each chunk:
    for each layer:
        out, next_state = layer(chunk, state=state_list[layer])
        state_list[layer] = next_state
```

### detach_state

`MHDSRA2Config.detach_state`（默认 `True`）是跨 chunk 的梯度截断开关：

```python
if cfg.detach_state:
    slot_k_next = slot_k_next.detach()
    slot_v_next = slot_v_next.detach()
```

设置为 `False` 时梯度可以跨越 chunk 边界回传，但显存占用随 chunk 数线性增长。

## 三、所有实验结果一览

| 实验 | seq_len | batch | vocab | steps | detach | 特殊配置 | best robust | best train | 时间 |
|---|---|---|---|---|---|---|---|---|---|
| **基线 2M** | 2M | 1 | **100** | 100 | True | 无 | **~1%** | 0% | — |
| 8K detach=False | 8K | 8 | 100 | 200 | **False** | 无 | **2.60%** | 12.5% | ~9min |
| 8K 500步 | 8K | 8 | 100 | 500 | False | needle α=0.5 | **3.12%** | 12.5% | ~30min |
| 8K 1000步 | 8K | 8 | 100 | 1000 | False | needle α=0.5 | **2.60%** | 25% | ~59min |
| 8K momentum | 8K | 8 | 100 | 1000 | False | momentum_qkv=True | **2.60%** | 25% | ~54min |
| 8K eta=1.0 | 8K | 8 | 100 | 1000 | False | eta=1.0 | **2.60%** | 25% | ~53min |
| **32K** | 32K | 8 | 100 | 500 | False | → 从8K加载 | **2.08%** | 12.5% | ~36min |
| ✅ **vocab=5 最小** | 1K | 8 | **5** | 200 | False | local=16 | **100%** | 100% | ~2min |
| ✅ **vocab=5 + detach=True** | 1K | 8 | **5** | 200 | **True** | local=16 | **100%** | 100% | ~2min |
| vocab=10 | 1K | 8 | **10** | 200 | False | local=16 | **23.96%** | 62.5% | ~3min |
| vocab=20 | 1K | 8 | **20** | 200 | False | local=16 | **8.33%** | 37.5% | ~3min |
| vocab=50 | 1K | 8 | **50** | 200 | False | local=16 | **4.17%** | 25% | ~3min |
| vocab=100 | 1K | 8 | **100** | 200 | False | local=16 | **3.13%** | 25% | ~3min |
| hidden MSE | 1K | 8 | 100 | 200 | False | +hidden_mse_α=0.3 | **1.04%** | 25% | ~5min |
| ✅ **2M + vocab=10** | **2M** | 1 | **10** | 5 | True | needle α=0.5 | **16.7%*** | **100%** | ~30min |

> \* 16.7% = vocab=10 下 6 个 answer 的随机水平

## 四、根因分析

### 梯度衰减路径

```
query loss → _slot_read → ∂loss/∂slot_k
  → slot_k_next(needle chunk) = (1-forget) × slot_k + write_gate × new_k
  → ∂loss/∂new_k = write_gate × ∂loss/∂slot_k_next      ← 衰减#1: write_gate ≈ 0.1
  → new_k = agg_k / mass
  → ∂loss/∂k_token = weights/mass × ∂loss/∂new_k         ← 衰减#2: weights/mass ≈ 0.001
  → ∂loss/∂V_proj = 衰减#1 × 衰减#2 ≈ 万分之二
```

**每次衰减约 0.1%**。结合 CE loss 的 softmax 稀释（100 类时每类 ~1%），总梯度到 V_proj/K_proj 约为 **十万分之一**。

### 关键实验推理链

1. **vocab=5 + detach=False → 100%**：但这是假阳性——vocab=5 下唯一可能的 value 是 token 4，模型只需"永远预测 4"
2. **vocab=5 + detach=True → 100%**：同上，证明 detach_state 本身不阻断学习
3. **vocab=10 + detach=False → 24%**：6 个 answer 时 slot 机制可以工作（远高于 16.7% 随机）
4. **vocab=100 + detach=False → 3%**：96 个 answer 时梯度彻底不够用
5. **以上所有，seq_len 1K~8K**：差异不显著
6. **2M + vocab=10 + detach=True → train 100%, robust 16.7%**：模型确实能在 2M 上检索，但学的是"读当前 slot"而非"写+读协议"

### 核心矛盾

| 需求 | detach_state | 效果 |
|---|---|---|
| 2M 下不 OOM | **必须 True** | 梯度不能跨 chunk 回传 |
| 学会写→读协议 | 需要 **False** | 2M 下 2048 chunks 全图 OOM |

## 五、已尝试但无效的方案

| 方案 | 效果 | 结论 |
|---|---|---|
| 更大 batch（8） | ✅ 显存 OK | 梯度更稳定但信号强度不变 |
| 更多步数（200→500→1000） | ❌ 不升 | 瓶颈不是步数 |
| 辅助 loss（needle loss α=0.5） | → 3.12%（小幅） | 信号方向对但量不够 |
| Momentum QKV | ❌ 无改善 | 寻址稳定性不是瓶颈 |
| eta=1.0 | ❌ 无改善 | 写入门饱和不是瓶颈 |
| vocab 扩展课程（5→10→20→50→100） | → 3.13% | out_proj 不是瓶颈 |
| Hidden state MSE | ❌ 无改善 | hidden 层也没绕过衰减 |
| 2M + vocab=10 | ✅ train 100% | 能检索但无法泛化 |

## 六、有效的方案

| 方案 | 效果 | 说明 |
|---|---|---|
| **detach_state=False** | 3% 基线 | 必须的但不够 |
| **vocab 缩小到 ≤10** | ~24% | answer space 小 → 每类梯度强 |
| **辅助 loss（needle 位置）** | 小幅提升 | 给 V_proj 直接梯度 |
| **vocab=5 + 任意 + 任意** | 100% | 但只有 1 个 answer，无检索意义 |

## 七、通往 2M NIAH 的可行路径

让 2M NIAH 真正可学的**唯一路径**是在 2M 下使用 `detach_state=False`（全 BPTT）。这需要解决显存问题：

### Gradient Checkpointing

核心思路：对不含 needle 或 query 的 chunk 用 `torch.utils.checkpoint` 丢弃中间激活，需要时重新计算。只对 needle chunk 和 query chunk 之间的关键路径保留计算图。

```
# 在 forward_selected_logits 中:
if chunk_needs_grad:
    chunk, state = full_forward(chunk, state)  # 保留全图
else:
    chunk = checkpoint(forward_chunk, chunk, state)  # 丢弃中间激活
```

**预期收益**：
- 显存从 O(2048×chunk_mem) 降到 O(k×chunk_mem)，k 是 needle→query 之间的 chunk 数
- 如果 needle 和 query 各在 1 个 chunk 中，k ≈ 2

**风险**：
- checkpoint 重计算增加约 20-30% 时间开销
- 需要正确识别哪些 chunk 包含 needle 或 query

### 替代方案：评估协议调整

如果架构限制无法突破，可以接受 `vocab=10` 作为 NIAH 评估的标准配置：
- 6 个 answer 仍有检索意义
- 模型已在 2M 上展示过 train 100% 的能力
- 但需要新的方法使 robust 泛化

## 八、代码位置

| 组件 | 路径 | 关键行 |
|---|---|---|
| Slot 写入 | `src/dsra/mhdsra2/improved_dsra_mha.py` | `_slot_write`: 394 |
| Slot 读取 | `src/dsra/mhdsra2/improved_dsra_mha.py` | `_slot_read`: 284 |
| detach_state | `src/dsra/mhdsra2/improved_dsra_mha.py` | 505-510, 352-354 |
| 模型配置 | `src/dsra/dsra_model.py` | `MHDSRA2Config`: 80-89 |
| NIAH 数据 | `scripts/needle_in_haystack_test.py` | `generate_haystack_with_needle`: 411 |
| NIAH 训练 | `scripts/needle_in_haystack_test.py` | `run_niah_verification_case`: 1709 |
| 评估 | `scripts/needle_in_haystack_test.py` | `evaluate_niah_depths`: 1213 |
| 日志记录 | `src/dsra/swanlab_utils.py` | `init_swanlab`, `SwanLabRunProxy` |
| 课程编排 | `scripts/niah_curriculum.py` | 第一版课程学习 |
| Vocab 课程 | `scripts/niah_curriculum_vocab.py` | vocab 扩展课程学习 |

## 九、结论

MHDSRA2 的 slot 注意力架构在 NIAH 任务上受到**梯度衰减**的根本限制。从 query loss 到 `_slot_write` 的 K_proj/V_proj 的梯度被双重衰减到约万分之一，无法支撑 100 类（96 个 answer）的长上下文检索学习。

模型在 2M 上确实能进行检索（train accuracy 100%），但由于 `detach_state=True` 的截断，学到的不是"写→读"协议，而是"读当前的 slot 内容"。后者无法泛化到独立样本。

要让 NIAH 泛化工作，需要 `detach_state=False` + gradient checkpointing，使梯度能跨越 2M 的 chunk 边界回传到 needle 位置的写入操作。
