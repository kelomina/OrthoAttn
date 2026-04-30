# DSRA (Decoupled Sparse Routing Attention)

## 当前 MHDSRA2 Carry Diagnostic 结论

截至 2026-04-30，`mhdsra2_carry_diagnostic_grid` 完整网格已按用户要求提前停止，保留当前 checkpoint：

- Checkpoint: `reports/mhdsra2_carry_diagnostic_grid.checkpoint.jsonl`
- 已完成进度：`81 / 1296` runs，约 `6.25%`
- 当前已完成部分主要覆盖 `curriculum_rule_set` 的 baseline 早期组合，尚未覆盖 `unit_with_carry_only`，因此“单独进位规则学习能力”还没有最终结论。

当前主实验默认参数：

```text
training_strategy = baseline
learning_rate = 0.01
max_steps_per_stage = 512
curriculum_eval_interval = 8
stage_patience = 3
replay_ratio = 0.75
layers = 1, 2, 4, 8, 16
device = auto
```

依据当前 checkpoint，低学习率与 `max_steps_per_stage=256` 是最强稳定因素；在多个 `curriculum_eval_interval` 和层数组合下，已出现 3 seeds 稳定达到：

```text
carry_em = 1.0
target_rate = 1.0
retained = 2
```

CUDA 三 seed 小对照使用 `cuda:0 = NVIDIA GeForce RTX 4070 Laptop GPU`，固定 `max_steps_per_stage=256`、`curriculum_eval_interval=8`、`replay_ratio=0.75`、`stage_patience=3`、`training_strategy=baseline`，比较 `learning_rate=0.003` 与 `0.01`：

```text
lr=0.003, layers=4: carry_mean=0.8333, retained_mean=1.3333, target_retention_rate=0.3333
lr=0.003, layers=8: carry_mean=0.7500, retained_mean=1.3333, target_retention_rate=0.3333
lr=0.010, layers=4: carry_mean=1.0000, retained_mean=2.0000, target_retention_rate=1.0000
lr=0.010, layers=8: carry_mean=1.0000, retained_mean=1.3333, target_retention_rate=0.6667
```

因此主实验默认学习率设为 `0.01`；`0.003` 仍可作为更保守的诊断候选。正式主实验报告入口现在通过 `--device auto` 选择设备，当前机器 CUDA 可用时会使用 GPU，否则回退 CPU。

使用上述默认参数重跑 `scripts/mhdsra2_layer_emergence_report.py --device auto --reports-dir reports` 后，正式报告确认：

```text
device = cuda
max_steps_per_stage = 512
learning_rate = 0.01
minimum_curriculum_mastery_layers = null
minimum_arithmetic_emergent_layers = null
```

阶段聚合显示 `4/8/16` 层均已稳定通过并保留 `unit_no_carry + unit_with_carry`，但 `two_digit_rules` 仍未达到阶段阈值，`100+100=200` 百位外推仍未涌现。

可选轻量配置：

```text
training_strategy = baseline
learning_rate = 0.01
max_steps_per_stage = 512
curriculum_eval_interval = 8
stage_patience = 3
replay_ratio = 0.75
layers = 4
device = auto
```

目前判断：此前 `unit_with_carry` 失败更可能来自学习率偏高或每阶段训练预算不足，而不是单纯层数不足。后续若继续诊断，应优先补跑 `unit_with_carry_only` 子网格，确认 MHDSRA2 是否能单独学会进位规则。

这是一个基于"固定容量可微状态库 + 语义路由 + 正交增量更新"的流式长序列注意力机制的可行性验证与测试套件。

## 目录结构
- `src/dsra/`: 正式源码目录，包含 DSRA 核心层、模型和报告工具。
- `src/dsra/mhdsra2/`: MHDSRA2 正式源码目录，包含多头流式注意力实现与分页精确记忆实现。
- `scripts/`: 正式脚本目录，包含统一入口、基准、消融和检索实验。
- `tests/`: 单元测试目录，`tests/fixtures/` 存放测试夹具数据。
- `reports/`: 正式报告输出目录。
- `archive/`: 历史副本与旧报告归档目录。
- 根目录同名 `.py` 文件为兼容薄封装，继续支持历史导入方式与 `python main.py ...` 调用。

## 核心模块
- `src/dsra/dsra_layer.py`: DSRA 核心网络层（含因果掩码、衰减机制、正交投影更新、分块因果掩码）。
- `src/dsra/dsra_model.py`: 支持多层堆叠的完整 LLM 架构包装（`MultiLayerDSRAModel`）。
- `src/dsra/mhdsra2/improved_dsra_mha.py`: MHDSRA2 多头分块流式注意力核心实现。
- `src/dsra/mhdsra2/paged_exact_memory.py`: MHDSRA2 的 CPU 分页精确记忆实现。

## 核心机制
- **正交增量更新（Orthogonal Update）**: 写入状态时将新信息投影到与已有状态正交的子空间，防止容量饱和。
- **指令旁路（Instruction Bypass）**: 由 Sigmoid 门控决定当前位置使用状态检索还是局部注意力，动态选择信息源。
- **位置编码模式（pe_mode）**:
  - `none`: 无位置编码
  - `rope`: Rotary Position Embedding
  - `alibi`: Attention with Linear Biases
  - `timestamps`: 基于时间戳的软衰减路由
- **MHDSRA2 三路融合（slot / local / retrieval）**:
  - `slot`: 由固定数量的全局槽位组成，负责跨 chunk 保留长期摘要记忆；读取时按相似度和置信度做 top-k，写入时结合 novelty、usage、age 和 conflict 控制更新幅度。
  - `local`: 由严格滑动窗口组成，负责保留最近上下文的精细时序依赖；窗口长度受 `local_window` 约束，不随总序列长度线性膨胀。
  - `retrieval`: 由 `PagedExactMemory` 等外部精确记忆提供，负责把远距离但仍需精确回忆的 token K/V 从 CPU 或更低成本存储召回到当前 chunk。
  - 三路输出会在 `MultiHeadDSRA2.forward()` 中通过 `fuse_gate` 生成的逐头门控进行归一化融合，从而同时兼顾长期压缩记忆、近邻细节和远距精确召回。

## 测试套件
- `tests/test_dsra_math.py`: 基础张量维度、正交性及极端边界条件单元测试。
- `tests/test_llm_compatibility.py`: 自回归生成与增量解码（KV-Cache）对齐测试。
- `tests/test_mhdsra2_core.py`: MHDSRA2 核心单元测试，直接覆盖张量 shape、流式状态更新和分页检索返回结果。
- `tests/test_mhdsra2_smoke.py`: MHDSRA2 脚本直跑与统一入口回归测试。
- `scripts/benchmark_complexity.py`: O(N) 性能与显存复杂度基准测试。
- `tests/test_state_saturation.py`: 状态库饱和度及 `lambda` 衰减机制验证。
- `scripts/toy_task_associative_recall.py`: 关联召回玩具任务，使用结构化 key/value/noise 分离词表。
- `scripts/needle_in_haystack_test.py`: 极长上下文的大海捞针召回能力测试。
- `scripts/ablation_study.py`: 核心机制（正交更新、指令旁路、位置编码）消融实验。
- `scripts/verify_mhdsra2.py`: MHDSRA2 冒烟测试、显存估算与可选微基准脚本。
- `main.py`: 兼容统一入口，继续支持 `unit`、`benchmark`、`saturation`、`recall`、`needle`、`mhdsra2`、`ablation` 及 `all`。

## 运行方式
```bash
# 推荐用法：
# - `python main.py ...`：根目录兼容入口（薄封装），适合大多数用户使用
# - `python scripts/main.py ...`：正式脚本入口（与 root main.py 等价），适合脚本开发/调试；支持 `-h` 查看 DSRA_FAST_ALL 等说明
#
# 查看帮助（包含 DSRA_FAST_ALL 等环境变量说明）
python scripts/main.py -h

# 单元测试
python main.py unit
python scripts/main.py unit

# 直接运行 pytest
pytest

# 直接运行 MHDSRA2 核心单元测试
python -m unittest tests.test_mhdsra2_core

# 关联召回玩具任务
python main.py recall
python scripts/main.py recall

# 大海捞针测试
python main.py needle
python scripts/main.py needle

# MHDSRA2 验证
python main.py mhdsra2
python scripts/main.py mhdsra2

# 直接运行 MHDSRA2 验证脚本
python scripts/verify_mhdsra2.py

# 消融实验
python main.py ablation
python scripts/main.py ablation

# 全部测试
python main.py all
python scripts/main.py all

# 快速跑 all（只跑最小套件，但仍生成 all_output.txt + run_summary.md）
# PowerShell:
$env:DSRA_FAST_ALL="1"; python main.py all
$env:DSRA_FAST_ALL="1"; python scripts/main.py all
# bash:
DSRA_FAST_ALL=1 python main.py all
DSRA_FAST_ALL=1 python scripts/main.py all

# 只生成报告索引（不执行其他套件）
python main.py report
python scripts/main.py report
```

## 报告输出
- `python main.py all` 会把整次统一运行日志写入 `reports/all_output.txt`，便于回看完整终端输出。
- 同一命令会生成 `reports/run_summary.md`，列出本次实际执行的测试套件与主要报告文件。
- `DSRA_FAST_ALL=1 python main.py all` 会以“最小套件”方式运行 `all` 分支，适合 CI 快速回归；但仍会生成 `reports/all_output.txt` 与 `reports/run_summary.md`。
- CI 解析建议：`python main.py all/report`（或 `python scripts/main.py all/report`）会在 stdout 末尾输出稳定的 key/value，便于机器解析与产物定位：
  ```text
  # report
  DSRA_REPORT_STATUS=ok
  DSRA_REPORT_RUN_SUMMARY=reports/run_summary.md
  DSRA_REPORT_EXECUTED_SUITES=0

  # all
  DSRA_ALL_STATUS=ok
  DSRA_ALL_LOG=reports/all_output.txt
  DSRA_ALL_RUN_SUMMARY=reports/run_summary.md
  DSRA_ALL_EXECUTED_SUITES=<N>
  ```
- 若对应套件被执行，还会生成以下正式报告文件：
  - `reports/ablation_summary.md`
  - `reports/ablation_summary.json`
  - `reports/needle_capacity_results.md`
  - `reports/needle_capacity_results.json`
  - `reports/json_retrieval_report.md`
  - `reports/json_retrieval_report.json`
  - `reports/json_retrieval_generalization_report.md`
  - `reports/json_retrieval_generalization_report.json`
- `python main.py report` 用于“只生成报告索引”，不执行其他测试套件；会生成或覆盖 `reports/run_summary.md`（`Executed Suites` 为空列表），适合作为 CI 的轻量报告产物检查与报告目录初始化步骤。

## 兼容说明
- 历史命令仍可继续使用，例如 `python main.py unit`。
- 历史导入仍可继续使用，例如 `from json_retrieval_test import run_json_retrieval_test`。
- 新开发建议优先从 `src/dsra/` 与 `scripts/` 引用正式实现。
