# Milestone 1: MQAR Domain Spec Alignment & Oracle Probe 对抗性压力测试报告

**评审代理 (Challenger)**: Challenger M1-1 (EMPIRICAL CHALLENGER / critic, specialist)  
**被评测模块**: `src/dsra/domain/mqar.py` (`MQARConfig`, `generate_mqar_batch`, `MQAROracleModel`)  
**评审日期**: 2026-08-22  
**终审结论**: **APPROVE (完全通过)**

---

## 1. 对抗性测试目标与测试矩阵

针对 Stanford HazyResearch Zoology / Titans 标准的 Multi-Query Associative Recall (MQAR) 任务规范，本挑战者构建了覆盖极值边界、四路词表互斥性、严格因果无泄漏性以及 Ground Truth Oracle 全知探针自洽性的 7 大压力测试维度：

| 压力测试维度 | 覆盖测试范围 / 极值点 | 验证指标 | 实测结果 |
|---|---|---|---|
| **1. 极限词表容量 (Vocab Matrix)** | $V \in \{4, 5, 8, 32, 64, 512, 8192, 65536\}$ | 张量边界、无越界、Oracle Top-1 命中率、CE 损失 | **100.0% Acc, 0.0 Loss (PASS)** |
| **2. 极限 KV 记忆容量 (KV Matrix)** | $K \in \{1, 2, 4, 16, 64, 128\}$, 均匀/随机插入模式 | 前缀无重叠插入、无容量截断、因果记忆构建 | **100.0% Acc, 0.0 Loss (PASS)** |
| **3. 查询数量极端边界 (Query Edge)** | $Q=1, Q=K, Q < K$ (如 $K=128, Q=1, 32, 128$) | 单查询/部分查询/全查询映射与损失掩码对齐 | **100.0% Acc, 0.0 Loss (PASS)** |
| **4. 序列长度极限拉伸 (Length Matrix)** | $L \in \{32, 512, 1024, 2048, 4096\}$ 及紧边界 $L=2K+Q$ | 极密集/极稀疏填充、零 Slack 边界无越界 | **100.0% Acc, 0.0 Loss (PASS)** |
| **5. 词表四路严格互斥 (Disjointness)** | 2000+ 批次数万序列 Monte Carlo 抽样验证 | $\{0\} \cap \text{Keys} = \emptyset$, $\text{Keys} \cap \text{Vals} = \emptyset$, $\text{Vals} \cap \text{Fill} = \emptyset$ | **零冲突、零交集 (PASS)** |
| **6. 严格因果时序与防未来泄漏** | 所有测试序列逐时间步扫描 | 查询 Key 必在前半部预现，后紧邻 Token 绝非真实 Value | **零未来信息泄漏 (PASS)** |
| **7. 异常输入防线 (Adversarial Errors)** | $V < 4, K < 1, Q < 1, Q > K, L < 2K+Q, \text{Pool}>V$ | 精确抛出可读 `ValueError` | **100% 拦截并抛出 (PASS)** |

---

## 2. 实测数据与详细指标

### 2.1 极限词表容量压测 (Vocab Size Grid on `cuda:0` & `cpu`)
设备: `NVIDIA GeForce RTX 4070 Laptop GPU` (`cuda:0`) 与 `CPU`

| 词表大小 $V$ | 键值对数 $K$ | 序列长 $L$ | 设备 | Key池 / Val池 | Oracle Acc | 交叉熵 Loss (ignore_index=0) | 判定 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **4** (极小下界) | 1 | 32 / 64 | cuda:0 / cpu | 1 / 1 (Pad=0, K=1, V=2, F=3) | **100.00%** | $0.000000\text{e}+00$ | **PASS** |
| **5** (最小奇数) | 1 | 32 / 64 | cuda:0 / cpu | 1 / 1 (Pad=0, K=1, V=2, F=3,4) | **100.00%** | $0.000000\text{e}+00$ | **PASS** |
| **8** (小词表) | 2 | 32 / 64 | cuda:0 / cpu | 2 / 2 | **100.00%** | $0.000000\text{e}+00$ | **PASS** |
| **32** | 4 | 64 | cuda:0 / cpu | 7 / 7 | **100.00%** | $0.000000\text{e}+00$ | **PASS** |
| **64** | 8 | 128 | cuda:0 / cpu | 15 / 15 | **100.00%** | $0.000000\text{e}+00$ | **PASS** |
| **512** | 16 | 512 | cuda:0 / cpu | 127 / 127 | **100.00%** | $0.000000\text{e}+00$ | **PASS** |
| **8192** (大词表) | 64 / 128 | 1024 / 2048 | cuda:0 / cpu | 2047 / 2047 | **100.00%** | $0.000000\text{e}+00$ | **PASS** |
| **65536** (LLM词表) | 128 | 2048 | cuda:0 / cpu | 16383 / 16383 | **100.00%** | $0.000000\text{e}+00$ | **PASS** |

### 2.2 极限 KV 容量与插入模式 (KV Capacity Grid on `cuda:0`)

| KV 对数 $K$ | 插入模式 | 词表大小 $V$ | 序列长 $L$ | Oracle Acc | 交叉熵 Loss | 结构非重叠验证 | 判定 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | uniform | 512 | 512 | **100.00%** | $0.000000\text{e}+00$ | 严格无重叠 | **PASS** |
| **1** | random | 512 | 512 | **100.00%** | $0.000000\text{e}+00$ | 严格无重叠 | **PASS** |
| **2** | uniform | 512 | 512 | **100.00%** | $0.000000\text{e}+00$ | 严格无重叠 | **PASS** |
| **2** | random | 512 | 512 | **100.00%** | $0.000000\text{e}+00$ | 严格无重叠 | **PASS** |
| **4** | uniform | 512 | 512 | **100.00%** | $0.000000\text{e}+00$ | 严格无重叠 | **PASS** |
| **4** | random | 512 | 512 | **100.00%** | $0.000000\text{e}+00$ | 严格无重叠 | **PASS** |
| **16** | uniform | 512 | 512 | **100.00%** | $0.000000\text{e}+00$ | 严格无重叠 | **PASS** |
| **16** | random | 512 | 512 | **100.00%** | $0.000000\text{e}+00$ | 严格无重叠 | **PASS** |
| **64** | uniform | 65536 | 2048 | **100.00%** | $0.000000\text{e}+00$ | 严格无重叠 | **PASS** |
| **64** | random | 65536 | 2048 | **100.00%** | $0.000000\text{e}+00$ | 严格无重叠 | **PASS** |
| **128** | uniform | 65536 | 2048 | **100.00%** | $0.000000\text{e}+00$ | 严格无重叠 | **PASS** |
| **128** | random | 65536 | 2048 | **100.00%** | $0.000000\text{e}+00$ | 严格无重叠 | **PASS** |

### 2.3 查询边界与部分查询测试 (Query Edge Cases on `cuda:0`)

| KV 对数 $K$ | 查询数 $Q$ | 关系 | 序列长 $L$ | Oracle Acc | 损失掩码一致性 (非 Query 为 0) | 判定 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1 | $Q = K = 1$ | 256 | **100.00%** | $100\%$ 一致 | **PASS** |
| 16 | 1 | $Q = 1 \ll K$ | 512 | **100.00%** | $100\%$ 一致 | **PASS** |
| 16 | 4 | $Q < K$ | 512 | **100.00%** | $100\%$ 一致 | **PASS** |
| 16 | 16 | $Q = K$ | 512 | **100.00%** | $100\%$ 一致 | **PASS** |
| 128 | 1 | $Q = 1 \ll K$ | 2048 | **100.00%** | $100\%$ 一致 | **PASS** |
| 128 | 32 | $Q < K$ | 2048 | **100.00%** | $100\%$ 一致 | **PASS** |
| 128 | 128 | $Q = K$ | 2048 | **100.00%** | $100\%$ 一致 | **PASS** |

### 2.4 序列长度极限拉伸与极紧边界 (Sequence Length Stress & Zero Slack)

| 序列长度 $L$ | KV 对数 $K$ | 稀疏度 / Slack | Oracle Acc | 交叉熵 Loss | 越界/溢出检查 | 判定 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **12** (紧边界 $2K+Q$) | $K=4, Q=4$ | 0 Slack (无 Filler) | **100.00%** | $0.000000\text{e}+00$ | 无越界 | **PASS** |
| **32** | 2 | 26 Slack | **100.00%** | $0.000000\text{e}+00$ | 无越界 | **PASS** |
| **512** | 8 | 488 Slack | **100.00%** | $0.000000\text{e}+00$ | 无越界 | **PASS** |
| **1024** | 8 | 1000 Slack | **100.00%** | $0.000000\text{e}+00$ | 无越界 | **PASS** |
| **2048** | 8 | 2024 Slack | **100.00%** | $0.000000\text{e}+00$ | 无越界 | **PASS** |
| **4096** (超长序列) | 8 | 4072 Slack | **100.00%** | $0.000000\text{e}+00$ | 无越界 | **PASS** |

---

## 3. 核心机制数学与因果安全性论证

1. **四路词表互斥性保证 (Zero Distractor Collision)**:
   - 词表区间划分为:
     $$\text{Pad} = \{0\}, \quad \text{Keys} = [1, 1 + K_{\text{pool}}), \quad \text{Vals} = [1 + K_{\text{pool}}, 1 + K_{\text{pool}} + V_{\text{pool}}), \quad \text{Fillers} = [1 + K_{\text{pool}} + V_{\text{pool}}, V)$$
   - 由于各区间左闭右开且端点严格递增，四者交集恒为 $\emptyset$。填充词（Fillers）绝不可能伪装成 Key 或 Value，彻底杜绝了模型因干扰词撞车导致虚假记忆的可能。
2. **严格单向因果性与防未来泄漏 (No Future Answer Leakage)**:
   - 前缀 KV 对在序列前半部分（$[0, L/2)$）插入，而查询 Keys 在序列后半部分（$[L/2, L)$）插入。
   - 所有查询 Key 必在前缀中无放回预先出现。在自回归生成中，时间步 $qpos$ 输入 $q\_key$，期望输出 Next Token 为对应 $q\_val$；生成器在 $qpos + 1$ 处仅放置 Filler 或下一 Query Key，绝不放置 $q\_val$，杜绝了提示词答案泄漏。
3. **Oracle 理论上限金标准探针 (`MQAROracleModel`)**:
   - Oracle 模型按因果时间步扫描输入序列，仅在前缀扫描到合法 $(k, v)$ 时更新内部状态字典；在遇到 query key 时在 $v$ 位置输出 logits=100.0。
   - 在所有合法测试网格下，Oracle 模型准确率均为严格 **100.0%**，损失均为严格 **0.000000e+00**，证明评测流水线逻辑完全自洽。

---

## 4. 全量回归测试与代码规范检查

- 运行测试命令: `python -m pytest tests/`
  - 结果: **424 passed, 5 subtests passed in 113.09s** (包含原有点查、检索融合、分层注意力、PPL及新增对抗性压力测试)
- 代码规范检查: `python -m ruff check tests/test_mqar_adversarial_stress.py src/dsra/domain/mqar.py`
  - 结果: **All checks passed!** (0 lint errors, 符合 line-length=100 及 py310 规范)

---

## 5. 挑战者裁定 (Verdict)

**Verdict**: **APPROVE**  
`src/dsra/domain/mqar.py` 的 MQAR 领域规范、数据生成器及 Oracle 探针在数学严谨性、极值鲁棒性、因果无泄漏性及理论上限对齐性上均表现完美，通过全部对抗性压力测试。
