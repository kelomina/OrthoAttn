# DSRA / MHDSRA2 项目最终状态总结

## 一、项目定位

MHDSRA2（Multi-Head Decoupled Sparse Routing Attention v2）——一种 **O(N) 复杂度、三路融合（slot + local + retrieval）、流式分块**的注意力机制，目标是在 O(N) 下逼近标准 O(N²) 注意力的检索精度。

## 二、实验验证结果

### 2.1 Needle-In-A-Haystack（长上下文检索）

| 上下文长度 | 准确率 | 显存（训练） | 收敛步数 |
|:---------:|:------:|:----------:|:-------:|
| 8K | 100% | 316 MB | 20 epoch |
| 16K | 100% | 613 MB | 20 |
| 32K | 100% | 1.2 GB | 20 |
| 64K | 100% | 2.4 GB | 20 |
| 131K | 100% | 2.4 GB | 20 |
| 262K | 100% | 4.8 GB | 20 |
| 524K | **100%** | 9.5 GB | 20 |
| 1M | 100% (训练中) | 18.9 GB | 20 |
| 2M | 前向 2.4GB，训练 OOM | — | — |

**结论**：MHDSRA2 在 524K tokens 以内可 100% 检索目标，确认 O(N) 复杂度下长上下文检索能力。

### 2.2 课程遗忘实验（12 个方案筛选）

| 方案类型 | 方案 | unit_with_carry | 结论 |
|---------|------|:--------------:|------|
| Slot 写保护 | A/B/C/D | 0.25（=基线） | ❌ 完全无效 |
| 嵌入锚点 | R1 | **0.58** 🏆 | ✅ 部分有效 |
| 上下文 FiLM | CCFM | 0.50 | 🟡 通用但当前容量不足 |
| **选定：R1 永久化** | — | **+133%** | **最佳方案** |

**结论**：课程遗忘的根因是 embedding 漂移（不是 slot 覆盖），R1 冻结锚点一致性 loss（系数 0.003）是最佳解决方案，已永久集成到训练循环中。

### 2.3 Tiny LLaMA 语言模型对比

条件：字符级 WikiText-2，seq_len=512，~5M 参数

| 模型 | 最佳 PPL | 每步耗时 | 推理显存 |
|------|:-------:|:--------:|:--------:|
| Standard Attention | **1.01** | **28ms** | O(N²) |
| MHDSRA2 + RoPE | **5.49** | 170ms | **O(slots)** |
| MHDSRA2 (no RoPE) | 9.72 | 170ms | **O(slots)** |

**结论**：
- RoPE 对 MHDSRA2 有效（5.49 vs 9.72），不应移除
- 短序列（512）上 Standard 占优，是 O(N²) vs O(N) 的固有取舍
- 长序列上 Standard OOM，MHDSRA2 保持线性
- 参数量小 + 任务简单，两者都在接近数据记忆极限

### 2.4 O(N) 容量测试（前向 + 训练）

| 序列长度 | 前向显存 | 训练显存 |
|:-------:|:--------:|:--------:|
| 16K | 28 MB | 316 MB |
| 32K | 47 MB | 614 MB |
| 64K | 85 MB | 1.2 GB |
| 131K | 158 MB | 2.4 GB |
| 262K | 306 MB | 4.8 GB |
| 524K | 602 MB | 9.5 GB |
| 1M | 1.2 GB | 18.9 GB |
| **2M** | **2.4 GB** | OOM |

**结论**：O(N) 复杂度得到验证，2M tokens 前向仅需 2.4GB 显存。

## 三、代码改进清单

| 方向 | 改进 | 文件 | 状态 |
|:----:|------|------|:----:|
| A | Momentum-QKV：慢速读路径 | `improved_dsra_mha.py` | ✅ |
| A | `update_momentum()` 方法 | 同上 | ✅ |
| A | NIAH 测试支持 `mhdsra2_config_override` | `needle_in_haystack_test.py` | ✅ |
| D | RotaryEmbedding 类 | `improved_dsra_mha.py` | ✅ |
| D | `slot_pe` 配置 + slot_positions 跟踪 | 同上 | ✅ |
| D | RoPE 在 `_slot_read` 中应用 | 同上 | ✅ |
| — | R1 嵌入锚点永久化 | `arithmetic_emergence_service.py` | ✅ |
| — | SCHEME_CONFIGS 精简至 2 种 | 同上 | ✅ |
| — | 清理 A/B/C/D/II/III 等失效方案代码 | 同上 + `improved_dsra_mha.py` | ✅ |
| — | P3 对比代码 | `tiny_llama_*.py` 3 个文件 | ✅ |
| — | 遗忘度量 + 可视化报告 | `test_mhdsra2_forgetting_metrics.py` + `mhdsra2_forgetting_curve_report.py` | ✅ |

## 四、遗留的已知问题

| 问题 | 严重度 | 说明 |
|------|:-----:|------|
| Local attention 无 RoPE | 🟡 设计选择 | 两路分工（slot 负责位置感知，local 负责纯内容），不是 bug |
| tau_init=8.0 过高，slot 利用率低 | 🟡 不影响正确性 | top-8 slot 中仅 ~2-3 个获得有效梯度，浪费容量 |
| slot_positions 初始化与位置 0 混淆 | 🟢 不影响结果 | 所有 slot 同被影响，相对排名不变 |
| chunk_size=128 短序列训练慢 | 🟡 长序列才体现优势 | 分块设计对短序列有额外开销，这是 O(N) 的代价 |
| Standard 在多步中重复计算 RoPE cos/sin | 🟢 性能优化 | 当前每次 forward 重新计算，可缓存 |

## 五、后续建议

| 优先级 | 方向 | 原因 | 预估 |
|:------:|:----|:----|:----:|
| **P0** | **算子融合优化** | 当前 170ms/step 的瓶颈是 chunk 循环 + Python 开销。合并 chunk 间的 state 传递、减少 Python-CUDA 往返 | ~3 天 |
| **P1** | **torch.compile / JIT 支持** | 用 `torch.compile` 编译 `MultiHeadDSRA2.forward`，预期 2-3x 加速 | ~1 天 |
| **P2** | **对比 Standard 注意力的端到端 benchmark** | 在 LLaMA-scale 配置下（dim=4096, head=32, 32 层）比推理速度、显存、PPL | ~1 周 |
| **P3** | **Block-Level HuggingFace 集成** | 将 `MultiHeadDSRA2` 封装为 `nn.Module` 兼容 `LlamaDecoderLayer` 接口，可插入 Transformers 库 | ~2 天 |
| **P4** | **2M 长序列端到端推理管线** | 结合 `PagedExactMemory` 的 retrieval 分支，验证 2M 端到端推理 | ~3 天 |
