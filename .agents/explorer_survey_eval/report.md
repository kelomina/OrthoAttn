# DSRA 评测流水线、真实性与测试套件深度白盒审计报告

**报告时间**: 2026-08-22
**审计专员**: Evaluation Pipeline & Authenticity Explorer
**工作目录**: `E:/Project/python/DSRA/.agents/explorer_survey_eval`
**关联需求**: `ORIGINAL_REQUEST.md` (R1, R2, R3, R4) & `AGENTS.md`

---

## 一、执行摘要 (Executive Summary)

针对 DSRA 项目新引入的 Stanford HazyResearch MQAR (Multi-Query Associative Recall) 模块、测试集及评测流水线，本次审计开展了全面的只读白盒审查、真实性排查、数学一致性验证、Ground Truth Oracle 探针实验及标准因果 Transformer 基线对比。

### 核心审计结论：
1. **零占位与真实性 100% 确认**：`src/dsra/domain/mqar.py`、`scripts/benchmark_mqar.py` 及相关测试文件中**不存在任何** dummy 占位、硬编码虚假返回值、未执行分支或 synthetic mocks。全部张量前向、损失反传与准确率统计均为纯端到端真实计算。
2. **损失与准确率计算数学严密**：损失函数严格采用 `CrossEntropyLoss(ignore_index=0)`，准确率在 $B \times Q$ 个 Query 位置上精确提取 Next-Token Argmax，完全排除所有 Filler/Distractor 和前缀 KV 插入位置的干扰。
3. **Ground Truth Oracle 探针验证通过**：成功设计并实证了基于因果前缀 KV 查表的 Oracle 探针模型。在变长与变容量网格（$L \in [512, 1024, 2048], K \in [4, 8, 16]$）下，评测流水线均精确输出 **100.0% 准确率** 与 **0.000000 损失**，排除了评测流水线自身存在上限截断或统计漏洞的可能。
4. **标准因果 Transformer 基线理论上限确认**：标准因果 Transformer（`StandardAttentionLM`，2层/128维）在 MQAR 任务上快速收敛至 **100.0% 准确率**（$L=512, K=4$ 在第 30 步达到 100%，$L=1024, K=8$ 在第 80 步达到 100%），彻底证明了评测流水线的有效性与高区分度。
5. **测试套件状态全绿**：全仓库共计 411 个单元与回归测试用例全部通过（100% Pass，耗时约 40s-97s），设备显式绑定 `cuda:0`，完全符合 `AGENTS.md` 规范。

---

## 二、评测流水线与数据生成深度白盒审计

### 1. 词表划分与数据生成机制 (`src/dsra/domain/mqar.py`)

代码路径：`src/dsra/domain/mqar.py:71-136`

```python
# 词表空间划分:
# 0: Pad/Ignore
# 1 .. key_end: Keys 候选池
# val_start .. val_end: Values 候选池
# filler_start .. V-1: Filler/Distractor 候选池
key_pool_size = max(16, min(64, (V - 1) // 4))
val_pool_size = max(16, min(64, (V - 1) // 4))
key_start = 1
key_end = key_start + key_pool_size
val_start = key_end
val_end = val_start + val_pool_size
filler_start = val_end
filler_end = V
```

#### 规范性审查清单：
- **互斥性 (Disjointness)**：`[0]` 为 Ignore；`[1, key_end)` 为 Keys；`[val_start, val_end)` 为 Values；`[filler_start, V)` 为 Fillers。四个区间严格互斥，无任何重叠交集。
- **KV 插入 (Prefix Insertion)**：在序列前半段（$0 \dots L/2$）均匀或随机采样 $K$ 个锚点位置，按 $(k_i, v_i)$ 成对相邻写入（$X[b, pos]=k_i, X[b, pos+1]=v_i$），并保证单样本内 $K$ 个 key 不重复。
- **Query 采样与自回归因果对齐 (Autoregressive Alignment)**：在序列后半段（$L/2 \dots L$）放置 $Q$ 个 Query token（$X[b, qpos]=q\_key_i$）。在因果语言模型中，输入为 $X[qpos]$ 时，模型在该时间步的输出 logits 预测下一 token，目标值设定为 $Y[b, qpos]=q\_val_i$。
- **无未来信息泄漏**：Query 仅出现在序列后半部分，前缀 KV 对出现在前半部分；因果注意力掩码（Causal Mask）阻止了从后向前的任何信息窥视。

### 2. 评测执行流程 (`scripts/benchmark_mqar.py`)

代码路径：`scripts/benchmark_mqar.py:32-74`

```python
def evaluate_mqar(
    model: nn.Module,
    config: MQARConfig,
    device: torch.device,
    eval_batches: int = 10,
    batch_size: int = 4,
) -> Dict[str, float]:
    model.eval()
    total_queries = 0
    correct_queries = 0
    total_loss = 0.0

    with torch.no_grad():
        for b_idx in range(eval_batches):
            seed = 9999000 + b_idx
            X, Y, qpos, targets = generate_mqar_batch(batch_size, config, device, seed=seed)
            logits = model(X)  # [B, L, V]
            
            # 计算仅在 query 预测目标处的 loss
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), Y.view(-1), ignore_index=0)
            total_loss += loss.item()

            # 计算各个 Query 位置上的 Top-1 命中率
            for b in range(batch_size):
                for q_idx in range(qpos.shape[1]):
                    pos = int(qpos[b, q_idx].item())
                    expected = int(targets[b, q_idx].item())
                    pred = int(logits[b, pos].argmax(dim=-1).item())
                    if pred == expected:
                        correct_queries += 1
                    total_queries += 1
```

#### 真实性排查项：
- **无虚假返回**：未发现任何 `return 1.0`、固定常量命中率或伪造统计。
- **无未执行分支**：批次生成、前向计算、损失计算、逐位置比对均被完整执行。
- **纯端到端评估**：评估过程使用独立的测试集 seed 集合（`seed = 9999000 + b_idx`），与训练期 seed 严格隔离。

---

## 三、损失与准确率计算数学严谨性审查

### 1. CrossEntropyLoss (ignore_index=0) 的数学机理

在 PyTorch 中，`F.cross_entropy(logits.view(-1, V), Y.view(-1), ignore_index=0, reduction="mean")` 计算公式为：

$$\mathcal{L} = - \frac{1}{N_{\text{active}}} \sum_{i: Y_i \ne 0} \log \frac{\exp(\text{logits}_{i, Y_i})}{\sum_{v=0}^{V-1} \exp(\text{logits}_{i, v})}$$

其中 $N_{\text{active}} = \sum_{i} \mathbb{I}(Y_i \ne 0) = B \times Q$。
因为在数据生成中，除 $B \times Q$ 个 Query 目标位置被赋值为对应的 Value ID（$Y \ge 1$）外，其余所有位置（Filler、Prefix Key、Prefix Value）均设为 $Y=0$。因此：
- 非 Query 位置对梯度反传和损失统计的贡献**严格为 0**；
- 损失值完全反映模型在联想召回 Query 上的预测负对数似然。

### 2. Top-1 准确率统计的无偏性

在 `evaluate_mqar` 中，遍历每个 batch $b$ 及每个 query $q\_idx$：
- 提取 Query 所在位置 `pos = qpos[b, q_idx]`；
- 期望标签 `expected = targets[b, q_idx] = Y[b, pos]`；
- 模型预测 `pred = logits[b, pos].argmax(dim=-1)`；
- 统计总数 $N_{\text{total}} = B \times Q \times \text{eval\_batches}$；
- 准确率 $\text{Accuracy} = \frac{N_{\text{correct}}}{N_{\text{total}}}$。

该统计方式在每个 Query 发生的位置进行因果决策判定，完全符合学术界标准（ICLR 2024 Zoology MQAR 标准）。

---

## 四、Ground Truth Oracle 全知探针设计与实证验证 (R3)

为了从数学与工程双重维度彻底排除评测流水线存在假阳性、假阴性或统计上限截断的可能，我们设计并实测了 Ground Truth Oracle 探针模型。

### 1. Oracle 模型架构设计

Oracle 探针模拟具备完美精确 KV 检索能力的理想因果系统：
1. **因果前缀扫描**：按时间步 $t \in [0, L-1]$ 扫描输入序列 $X[b, t]$；
2. **KV 状态表构建**：当在位置 $t$ 观察到 Key token（$X[b, t] \in [\text{key\_start}, \text{key\_end})$）且在 $t+1$ 观察到 Value token（$X[b, t+1] \in [\text{val\_start}, \text{val\_end})$）时，将映射 $k \mapsto v$ 存入样本局部的 KV 查表中；
3. **Query 命中输出**：在任意位置 $t$，若当前输入的 token $X[b, t]$ 命中已存储的 Key，则在输出 logits 的对应 $v$ 维度赋予超大正值（如 $+100.0$），其余维度置 0。

### 2. Oracle 探针实测结果

在多个序列长度与 KV 容量配置下运行 `evaluate_mqar(oracle, cfg, device="cuda:0")`：

| 序列长度 ($L$) | KV 对数 ($K$) | Query 数 ($Q$) | 评估样本批次 | 准确率 (Accuracy) | 评估损失 (Loss) | 验证结论 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 512 | 4 | 4 | 5 batches (20 samples) | **100.0%** (80/80) | **0.000000** | 完美通过 |
| 1024 | 8 | 8 | 5 batches (20 samples) | **100.0%** (160/160) | **0.000000** | 完美通过 |
| 2048 | 16 | 16 | 5 batches (20 samples) | **100.0%** (320/320) | **0.000000** | 完美通过 |

**审计结论**：Oracle 模型在评测流水线下获得了精确的 100.0% 准确率与 0.000000 损失，证明评测脚本、数据生成器、损失掩码与统计函数完全自洽无误。

---

## 五、标准因果 Transformer 全注意力基线对照实验 (R4)

为了验证 MQAR 数据集在神经网络模型上的可学习性与理论上限，我们引入了标准的因果 Transformer（`StandardAttentionLM`，包含 Flash Attention / SDPA 全因果自注意力与 RoPE 旋转位置编码）。

### 1. 实验配置
- **模型规格**：2 层 TransformerBlock, 维度 $D=128$, 头数 $H=4$, FFN 隐藏层 512, 标准因果注意力掩码
- **优化器**：AdamW (lr=1e-3, weight_decay=1e-4)
- **设备**：`cuda:0` (NVIDIA GeForce RTX 4070 Laptop GPU)

### 2. 训练收敛动态实测

#### 配置 A: $L=512, K=4, Q=4$
- **Step 10**: train_loss = 5.4950, eval_acc = 0.0%, eval_loss = 5.4975
- **Step 20**: train_loss = 3.7042, eval_acc = 61.3%, eval_loss = 3.3512
- **Step 30**: train_loss = 0.7679, **eval_acc = 100.0%**, eval_loss = 0.4851
- **Step 40**: train_loss = 0.0538, **eval_acc = 100.0%**, eval_loss = 0.0519

#### 配置 B: $L=1024, K=8, Q=8$
- **Step 20**: train_loss = 5.0841, eval_acc = 0.0%, eval_loss = 5.0487
- **Step 40**: train_loss = 3.0039, eval_acc = 48.8%, eval_loss = 2.9727
- **Step 60**: train_loss = 0.7634, eval_acc = 98.8%, eval_loss = 0.5511
- **Step 80**: train_loss = 0.0526, **eval_acc = 100.0%**, eval_loss = 0.0760

### 3. 实验分析与意义
1. 标准 Transformer 依赖 $O(L^2)$ 全因果注意力矩阵，能够直接建立 Query token 与 Prefix $(k_i, v_i)$ 之间的注意力转移路径；
2. 实验证明 MQAR 任务在标准注意力机制下具备极佳的梯度反传可优化性，能够在几十步内迅速收敛至 100.0%；
3. 这与线性/循环/流式注意力（如 MHDSRA2 的固定容量槽位压缩与分页精确召回）形成了极具区分度的学术对比基准。

---

## 六、测试基础设施与工程规范审查 (Testing & Infrastructure)

### 1. 测试套件覆盖与执行状态
- 全仓库共有 12 个测试文件，通过 `python -m pytest` 收集并运行了 **411 个测试用例**；
- 测试涵盖：数学正交更新公式、梯度流反传、状态生命周期、门控策略、多层召回、PPL 计算回归、安全合规性及 MQAR 数据生成；
- **测试结果**：411 测全部通过（`411 passed in 40.06s`），无跳过、无失败、无异常。

### 2. 命令行与入口集成
- 统一入口 `python main.py mqar` 与 `python scripts/main.py mqar` 支持一键触发 MQAR 评测套件；
- 支持参数配置：`--seq-len`, `--num-kv`, `--epochs`, `--batch-size`, `--dim`, `--suite`, `--device`；
- 报告自动生成至 `docs/reports/verify_technical_report/mqar/`，图表输出至 `docs/figures/verify_technical_report/`。

### 3. GPU 与设备规范合规性 (`AGENTS.md` 对齐)
- 默认显式使用 `cuda:0` 设备；
- 批次生成直接在目标设备上分配显存，避免无谓的 CPU-GPU 往返传输；
- 提供 `torch.cuda.is_available()` 兼容回退，确保 CPU 环境不崩溃。

---

## 七、建议与后续实施规划 (Recommendations)

为使 MQAR 验证套件更加完善，建议在后续实施阶段补充以下增强：
1. **将 Oracle 探针纳入自动化单元测试**：在 `tests/test_mqar_data_generation.py` 中新增 `test_mqar_ground_truth_oracle_100_percent_accuracy` 用例，实现持续集成自动防护；
2. **将 Standard Transformer 对照写入标准评测脚本**：在 `scripts/benchmark_mqar.py` 中加入 `--baseline transformer` 选项，支持一键运行 MHDSRA2 与 Transformer 双曲线对比；
3. **扩充多容量与变长网格测试**：支持生成从 $L=512$ 到 $L=4096$、$K=4$ 到 $K=32$ 的长序列容量压测报告。
