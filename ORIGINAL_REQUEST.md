# Original User Request

## 2026-08-22T00:54:35Z

对 DSRA 项目新引入的 Stanford HazyResearch MQAR (Multi-Query Associative Recall) 模块、测试集及评测流水线进行深度的基准对齐审计与独立对抗验证，确保其严格符合 Stanford Zoology (ICLR 2024) 标准规范而非简化占位。

Working directory: E:/Project/python/DSRA
Integrity mode: benchmark

## Requirements

### R1. Stanford Zoology MQAR 官方规范逐行比对与数学等价性证明
- 白盒审查 `src/dsra/domain/mqar.py` 的数据生成逻辑是否与 Stanford HazyResearch 官方 `zoology.data.associative_recall` 规范完全对齐：
  1. 词表划分：`[0]` 为 Pad/Ignore，`[1..K_pool]` 为 Keys，`[K_pool+1..V_pool]` 为 Values，其余为 Fillers，三者严格互斥无重叠；
  2. 键值分布：前半段随机插入 $K$ 对 $(k_i, v_i)$ 键值对，中间由随机 Distractor Tokens 填充；
  3. 查询生成：后半段随机打乱并排列 $Q$ 个 Query Keys，严格自回归因果对齐（输入 $X[qpos]=Key$，期望输出 $Y[qpos]=Value$），无未来信息泄漏；
  4. 损失掩码：除 Query 预测位置外，其余位置 $Y=0$（`ignore_index=0`）。

### R2. 评测流水线真实性与零占位代码白盒审计
- 审查 `scripts/benchmark_mqar.py` 与 `tests/test_mqar_data_generation.py` 是否存在任何逻辑占位、虚假返回值、未执行的代码分支或人为设定的伪结果；
- 验证纯端到端评估的损失计算与 Top-1 准确率统计是否严格遵循标准自回归 Cross-Entropy 范式。

### R3. Oracle 全知探针测试 (Ground Truth Oracle Verification)
- 构建独立的 Oracle 探针测试：构造一个具备完美 KV 查找能力的 Oracle 模型（例如通过查表直接返回对应 Value），验证评测流水线与准确率统计函数能否精确输出 **100.0%** 准确率，排除由于数据生成或评估代码自身 Bug 导致的虚假准确率。

### R4. 标准 Transformer 全注意力基线对照实验
- 在相同的 MQAR 任务（如 $L=512, K=4$ 与 $L=1024, K=8$）下，引入标准因果 Transformer（带 Flash-Attention / 全注意力）作为理论上限对照；
- 验证标准 Transformer 能否在 MQAR 上快速学习到 90%+ 准确率，并与 MHDSRA2 进行对比分析，以此从形式化和实验两个维度彻底证实评测流水线的真实有效性。

## Acceptance Criteria

### 规范合规性与数学严谨性 (Spec Compliance)
- [ ] `src/dsra/domain/mqar.py` 与 Stanford Zoology 官方 `MQARConfig` 形式等价，词表划分与键值对应无冲突。
- [ ] 零占位代码：所有模块均具备完整前向反向与张量计算逻辑。

### Oracle 探针验证 (Oracle Verification)
- [ ] Oracle 完美查找模型在评测脚本下获得精确的 `100.0%` 准确率（$loss = 0.0$），证明评测流水线在逻辑上无任何漏统或虚假上限。

### 基线对照与测试全绿 (Baseline & Tests)
- [ ] 标准 Transformer 基线能够正常收敛，与 MHDSRA2 形成有意义的对比。
- [ ] `pytest tests/test_mqar_data_generation.py` 与全仓库单测 100% 全部通过。
- [ ] 生成包含白盒审计、Oracle 探针与基线对照的正式 Markdown/JSON 验证报告。
