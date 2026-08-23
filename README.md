# DSRA Attention

> 中文：DSRA Attention 是一个面向研究验证的流式长序列注意力机制项目，核心实现是
> MHDSRA2。它用固定容量状态槽、局部窗口和可选的 CPU 分页精确召回，探索长序列场景下
> “既保留关键信息，又避免完整回看所有历史 token”的可行路径。
>
> English: DSRA Attention is a research-oriented verification suite for
> streaming long-sequence attention. Its active architecture is MHDSRA2, which
> combines fixed-capacity state slots, a bounded local window, and optional
> CPU-side paged exact retrieval.

- Package: `dsra-attn`
- Python: `>=3.10`
- Status: research / diagnostic / experimental
- Default active architecture: `mhdsra2`

## 项目定位 / Project Positioning

中文：这个仓库不是一个可直接替换 GPT 类模型的生产级 LLM 库，而是一个用于验证注意力机制
想法的实验套件。你可以把它理解成一个“长记忆注意力实验室”：我们在这里检查记忆是否串样本、
召回是否漏掉关键 token、softmax 是否把已经召回的证据再次摊薄，以及不同读出方式是否真的
改善验证集和测试集指标。

English: This repository is not a production LLM library. It is an experimental
suite for diagnosing and comparing attention mechanisms: memory isolation,
exact retrieval quality, post-retrieval softmax dilution, task-specific readout,
and reproducible validation/test behavior.

当前边界 / Current boundaries:

- 中文：结果主要是机制级、smoke 级和消融实验级证据，不能解读为“已经解决长上下文推理”。
- English: Results are mechanism-level, smoke-level, or ablation-level evidence;
  they should not be read as solved long-context reasoning.
- 中文：`dsra` 目前是归档别名，会归一化到 `mhdsra2`；不要把二者当成两个独立活跃架构。
- English: `dsra` is currently an archived alias normalized to `mhdsra2`; do not
  treat those labels as two independent active architectures.
- 中文：CPU 分页精确记忆是参考实现，适合验证机制，不是 FAISS/ScaNN 级生产索引。
- English: CPU-side paged memory is a reference implementation for experiments,
  not a FAISS/ScaNN-grade production index.

## 核心特性 / Key Features

- 中文：三路融合注意力：`slot` 全局压缩记忆、`local` 滑动窗口、`retrieval` 外部分页召回。
- English: Three-branch attention fusion: `slot` compressed global memory,
  `local` sliding-window context, and optional `retrieval` from external memory.
- 中文：batch 隔离的分页精确记忆，支持召回 mask、future-token cutoff、latest-wins 诊断和
  跨样本泄漏检查。
- English: Batch-isolated paged exact memory with retrieval masks, future-token
  cutoffs, latest-wins diagnostics, and cross-sample leak checks.
- 中文：slot 写入包含 overwrite-aware 诊断字段，如 novelty、overwrite gate、write drive、
  usage 和 confidence。
- English: Slot writes include overwrite-aware diagnostics such as novelty,
  overwrite gate, write drive, usage, and confidence.
- 中文：可选的 `retrieval_attention_topk` 用来处理“召回后 softmax 稀释”风险。
- English: Optional `retrieval_attention_topk` addresses post-retrieval softmax
  dilution.
- 中文：保留旧 API 和根目录 `python main.py ...` 调用方式，便于历史脚本继续运行。
- English: Legacy-compatible APIs and root-level `python main.py ...` commands
  are kept for older scripts.
- 中文：提供单元测试、smoke 测试、benchmark、消融实验、依赖审计和 Markdown/JSON 报告。
- English: Includes unit tests, smoke tests, benchmarks, ablations, dependency
  audits, and Markdown/JSON report writers.

## 研究重点 / Current Research Focus

### Retrieval Softmax Dilution / 召回后 Softmax 稀释

中文：分页召回已经把候选 token 找回来，并不代表模型最终会把注意力集中到正确证据上。如果
召回池里有 128 个候选，其中只有 1 个是精确匹配，普通 softmax 仍可能把权重摊到大量干扰项
上。`retrieval_attention_topk` 的作用是在召回保持较宽的前提下，只让分数最高的有效候选参与
retrieval attention 的 softmax 归一化。

English: Successful retrieval does not guarantee that the readout attention will
focus on the correct evidence. With 128 retrieved candidates and only one exact
match, a full softmax can still spread probability mass across many distractors.
`retrieval_attention_topk` keeps broad recall available but limits the retrieval
attention softmax to the strongest valid candidates.

重要边界 / Important boundary:

- 中文：`retrieval_attention_topk` 默认关闭，默认值是 `None`。
- English: `retrieval_attention_topk` is disabled by default; the default is
  `None`.
- 中文：它不同于降低 `retrieval_max_tokens`。前者是“召回后限制 softmax 候选”，后者会在召回
  阶段提前裁掉候选，可能丢失证据。
- English: It is different from lowering `retrieval_max_tokens`. Top-K is a
  post-recall attention mask; `retrieval_max_tokens` reduces the retrieved pool
  itself and may discard evidence.
- 中文：当前证据是机制探针和小型 smoke，不是 NIAH/JSON 任务准确率已经提升的最终结论。
- English: Current evidence is mechanism-probe and smoke-test evidence, not a
  final claim of improved NIAH/JSON task accuracy.

机制探针 / Mechanism probe:

```text
retrieval_max_tokens=128, retrieval_tau=8

topk=None  exact-match weight=0.117558
topk=32    exact-match weight=0.353073
topk=16    exact-match weight=0.530058
topk=8     exact-match weight=0.707344
```

相关报告 / Related report:

- `reports/mhdsra2_retrieval_attention_topk_smoke_cuda.md`

### Structured JSON Readout / 结构化 JSON 读出

中文：`extract_compose_readout` 是默认关闭的 JSON 任务读出 adapter。它验证的是“先定位证据
窗口，再按结构化字段抽取并拼接答案”的路径，不是通用自然语言逐字生成能力。

English: `extract_compose_readout` is a default-off JSON readout adapter. It
tests an evidence-window extraction and answer-composition path, not general
free-form language generation.

### NIAH and Arithmetic Diagnostics / NIAH 与算术诊断

中文：NIAH、JSON retrieval 和 two-digit arithmetic 都是诊断任务。它们用于定位候选召回、
读出、泛化和训练稳定性问题，不应被包装成生产能力证明。

English: NIAH, JSON retrieval, and two-digit arithmetic are diagnostic tasks.
They are useful for locating retrieval, readout, generalization, and training
stability issues, but they are not production capability claims.

## 架构 / Architecture

```text
.
|-- archive/                 # Historical snapshots and old local copies
|-- config/                  # Experiment configuration objects
|-- reports/                 # Generated Markdown/JSON reports and logs
|-- scripts/                 # CLI entrypoints, benchmarks, diagnostics, audits
|-- src/dsra/                # Formal Python package
|   |-- application/         # Use-case services and unit-of-work boundary
|   |-- domain/              # Specs, validation, and model-name normalization
|   |-- infrastructure/      # Repository implementations and report adapters
|   `-- mhdsra2/             # MHDSRA2 attention and exact memory engine
|-- tests/                   # Unit, integration, smoke, and report tests
|-- main.py                  # Legacy-compatible CLI wrapper
`-- pyproject.toml           # Package metadata and development dependencies
```

| Layer / 层 | Path / 路径 | Responsibility / 职责 |
|---|---|---|
| Domain / 领域层 | `src/dsra/domain/` | Attention specs, validation, model-name normalization / 注意力规格、校验、模型名规范化 |
| Application / 应用层 | `src/dsra/application/` | Forward-call coordination, retrieval services, factories / 前向调用协调、检索服务、模型工厂 |
| Infrastructure / 基础设施层 | `src/dsra/infrastructure/` | Paged memory repository and report adapters / 分页记忆仓储、报告适配 |
| Core / 核心实现层 | `src/dsra/mhdsra2/` | `MultiHeadDSRA2`, `MHDSRA2Config`, `MHDSRA2State`, `PagedExactMemory` |
| Compatibility / 兼容层 | `src/dsra/dsra_layer.py`, `src/dsra/dsra_model.py`, `main.py` | Older imports and CLI compatibility / 旧导入与 CLI 兼容 |

中文：新开发优先从 `src/dsra/` 和 `scripts/` 引用正式实现。根目录 `main.py` 是兼容包装器，
主要负责把旧命令转发给 `scripts/main.py`。

English: New code should usually import from `src/dsra/` or call scripts under
`scripts/`. The root `main.py` is a compatibility wrapper around
`scripts/main.py`.

## 安装 / Installation

中文：推荐使用项目内虚拟环境，避免和系统 Python 依赖混在一起。

English: Use a project-local virtual environment to keep dependencies isolated.

Windows PowerShell:

```powershell
python -m venv .env
.\.env\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

bash/zsh:

```bash
python -m venv .env
source .env/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

中文：`torch` 已在依赖中声明，但 CUDA 版本的 PyTorch 和本机显卡驱动强相关。如果要跑较重
的 GPU 实验，请先按 PyTorch 官方说明安装匹配 CUDA 的构建版本。

English: `torch` is declared as a dependency, but CUDA builds depend on your
local driver/runtime. For GPU-heavy experiments, install the PyTorch build that
matches your CUDA environment.

## 快速开始 / Quick Start

查看 CLI 帮助 / Show CLI help:

```bash
python main.py -h
```

运行完整测试 / Run tests:

```bash
pytest
```

运行项目兼容单测入口 / Run the compatibility unit entrypoint:

```bash
python main.py unit
```

运行 MHDSRA2 smoke 验证 / Run MHDSRA2 smoke verification:

```bash
python main.py mhdsra2
```

只生成报告索引 / Generate only the report index:

```bash
python main.py report
```

运行快速版 all suite / Run a fast `all` suite:

```powershell
$env:DSRA_FAST_ALL="1"; python main.py all
```

```bash
DSRA_FAST_ALL=1 python main.py all
```

## 常用命令 / Common Commands

| Goal / 目标 | Command / 命令 |
|---|---|
| Unit tests / 单元测试 | `python main.py unit` |
| Benchmark / 复杂度与性能基准 | `python main.py benchmark` |
| Saturation diagnostic / 状态饱和诊断 | `python main.py saturation` |
| Associative recall / 联想回忆玩具任务 | `python main.py recall` |
| Needle diagnostic / NIAH 诊断 | `python main.py needle` |
| Needle capacity / NIAH 容量报告 | `python main.py needle_capacity` |
| JSON retrieval / JSON 检索诊断 | `python main.py json_retrieval` |
| JSON generalization / JSON 泛化诊断 | `python main.py json_retrieval_generalization` |
| Attention family benchmark / 注意力家族基准 | `python main.py attention_family_benchmark` |
| MHDSRA2 smoke / MHDSRA2 验证 | `python main.py mhdsra2` |
| MHDSRA2 compare / MHDSRA2 对比报告 | `python main.py mhdsra2_compare` |
| Next-round benchmark / 下一轮统一基准 | `python main.py next_round_benchmark` |
| Arithmetic emergence / 算术层涌现报告 | `python main.py mhdsra2_layer_emergence` |
| Curriculum grid / 课程策略网格 | `python main.py mhdsra2_curriculum_strategy_grid` |
| Carry diagnostic grid / 进位规则诊断网格 | `python main.py mhdsra2_carry_diagnostic_grid` |
| Ablation / 消融实验 | `python main.py ablation` |
| Chat entrypoint / 交互聊天入口 | `python main.py chat` |
| Report index / 报告索引 | `python main.py report` |

快速对比 / Fast comparison:

```powershell
$env:DSRA_FAST_COMPARE="1"; python main.py mhdsra2_compare
```

```bash
DSRA_FAST_COMPARE=1 python main.py mhdsra2_compare
```

## 独立脚本 / Standalone Scripts

中文：有些工作流尚未收进统一 CLI，而是以独立脚本形式存在。

English: Some workflows are still exposed as standalone scripts instead of
unified `main.py` commands.

```bash
# Installed-package OSV audit; sends installed package names and versions to OSV.
python scripts/audit_installed_packages_osv.py --output reports/dependency_osv_audit.json --fail-on-vuln

# Exact retrieval quality smoke: batch isolation, future cutoff, latest-wins recall.
python scripts/mhdsra2_batch_retrieval_quality_smoke.py

# Batched retrieval profiling.
python scripts/mhdsra2_batched_retrieval_benchmark.py --json-out reports/mhdsra2_batched_retrieval_profile.json --markdown-out reports/mhdsra2_batched_retrieval_profile.md

# P0/P1 regression ablation for slot overwrite, page recall, and forward_step reuse.
python scripts/mhdsra2_bugfix_ablation.py

# P2 engineering regression ablation.
python scripts/mhdsra2_p2_engineering_ablation.py

# Quality-improvement ablation dry run.
python scripts/mhdsra2_quality_improvement_ablation.py --dry-run --device cpu
```

## 实验与报告 / Experiments and Reports

中文：`reports/` 保存 Markdown 和 JSON 实验产物。成熟使用方式是同时保留人能读懂的 `.md`
摘要，以及机器可解析的 `.json` 数据。临时日志、seed 级原始快照和本地探索文件不应被误认为
正式结论。

English: `reports/` stores generated Markdown and JSON artifacts. Mature report
usage keeps a human-readable `.md` summary and a machine-readable `.json`
payload when structured data is available. Temporary logs, seed-specific raw
snapshots, and local exploration files should not be mistaken for canonical
evidence.

常见输出 / Common outputs:

- `reports/run_summary.md`: summary generated by `python main.py report` or
  `python main.py all`.
- `reports/all_output.txt`: captured terminal output from the unified `all`
  command.
- `reports/mhdsra2_retrieval_attention_topk_smoke_cuda.md`: retrieval attention
  Top-K smoke report.

证据分级 / Evidence levels:

| Level / 等级 | Meaning / 含义 | How to read it / 如何解读 |
|---|---|---|
| Mechanism probe / 机制探针 | Direct tensor-level behavior / 直接张量行为 | Useful for confirming a specific mechanism, not task quality / 能确认机制，不等于任务质量 |
| Smoke test / 冒烟测试 | Small run that checks integration / 小规模接入检查 | Shows the path runs, not that it is optimal / 说明链路可跑，不说明最优 |
| Ablation / 消融 | Controlled comparison of variants / 受控对比 | Stronger evidence when validation-first and multi-seed / 多 seed 且先看验证集才更可信 |
| Held-out test / 独立测试 | Final check after selection / 选型后的最终检查 | Should not be used to choose the winning config / 不能用来挑配置 |

## 测试与质量 / Testing and Quality

推荐本地检查 / Recommended local checks:

```bash
pytest
ruff check .
```

快速 smoke / Faster smoke:

```bash
python main.py unit
python main.py mhdsra2
```

近期 retrieval 相关定向检查 / Focused checks used by recent retrieval work:

```bash
pytest tests/test_diagnostic_gate_policy_regressions.py -k "retrieval_attention_topk or retrieval_mask" -q
pytest tests/test_memory_lifecycle_regressions.py tests/test_multilayer_retrieval_regressions.py -q
pytest tests/test_mhdsra2_quality_improvement_ablation.py -q
pytest tests/test_security_regressions.py -q
```

中文：测试应可复现，不依赖真实生产服务、真实密钥或不稳定网络。CUDA 实验优先使用
`cuda:0`；单测在没有 CUDA 时应能跳过或走 CPU 回退。

English: Tests should be deterministic and should not depend on production
services, real secrets, or unstable network calls. CUDA experiments should use
`cuda:0` when available; unit tests should skip or fall back to CPU when CUDA is
not available.

## 配置说明 / Configuration Notes

- 中文：`MHDSRA2Config.retrieval_attention_topk=None` 是默认行为；显式设置为正整数才启用
  召回后 Top-K softmax mask。
- English: `MHDSRA2Config.retrieval_attention_topk=None` is the default; set a
  positive integer to enable post-retrieval Top-K softmax masking.
- 中文：`retrieval_tau` 控制 retrieval attention 的温度；它和 Top-K 是两个不同旋钮。
- English: `retrieval_tau` controls retrieval attention temperature; it is a
  different knob from Top-K masking.
- 中文：`retrieval_max_tokens` 控制召回候选池大小；调小它可能直接删掉证据。
- English: `retrieval_max_tokens` controls the retrieved candidate pool size;
  lowering it may remove evidence before attention sees it.
- 中文：`extract_compose_readout`、evidence supervision、learned retrieval gate
  等实验开关默认关闭，不能写成默认模型能力。
- English: `extract_compose_readout`, evidence supervision, learned retrieval
  gate, and related switches are default-off experiments; do not describe them
  as default model capabilities.
- 中文：SwanLab 上传默认关闭；只有明确需要云端记录时再启用。
- English: SwanLab uploads are disabled by default; enable cloud logging only
  when explicitly intended.

## 开发规范 / Development Guidelines

中文：本项目优先维护可验证、可回滚、边界清楚的研究代码。修改时尽量扩展已有模块，不新建平行
实现；涉及行为变化时同步补测试和报告。

English: The project favors verifiable, reversible, clearly scoped research
code. Prefer extending existing modules over creating parallel implementations;
behavioral changes should include tests and, when relevant, reports.

Practical rules / 实用规则:

- Keep tests under `tests/`, reports under `reports/`, configs under `config/`.
- Do not commit real API keys, tokens, passwords, private datasets, or sensitive
  logs.
- Keep public function names, argument order, return types, and exception
  behavior stable unless a migration is documented.
- If an experiment fails, record the failure instead of turning temporary tuning
  into hidden default behavior.
- For public distribution, add or keep a standalone `LICENSE` file consistent
  with the intended Apache-2.0 notice.

## 贡献 / Contributing

中文：仓库暂未提供正式 `CONTRIBUTING.md`。在此之前，建议每次贡献都保持小步、可验证、
证据优先。

English: This repository does not yet include a formal `CONTRIBUTING.md`. Until
one is added, keep contributions small, testable, and evidence-based.

Suggested contribution checklist / 建议贡献检查清单:

1. Explain the problem and expected measurable improvement.
2. Check existing failure records before repeating a known failed direction.
3. Add or update tests for changed behavior.
4. Run the relevant commands locally and report exact results.
5. Keep generated report changes only when the result or schema intentionally
   becomes part of the evidence.

## 许可 / License

中文：当前 README 沿用 Apache License 2.0 许可说明。若要公开发布或分发仓库，请确保源码树中
存在独立的 `LICENSE` 文件，并与项目元数据保持一致。

English: This README follows the Apache License 2.0 notice already used by the
project. Before public distribution, keep a standalone `LICENSE` file in the
source tree and make sure project metadata stays consistent with it.
