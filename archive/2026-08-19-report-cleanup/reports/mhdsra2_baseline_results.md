# MHDSRA2 Baseline Results - 2026-06-30

## 结论边界

本报告是本轮 MHDSRA2 改进候选分层验证 Loop 的 Step 0 基线锚定。它只说明当前工作树在一组低成本检查下的配置、机制安全门和 smoke 链路状态，不证明模型已经具备长上下文推理能力，也不把 1 step / 1 epoch 的结果包装成质量结论。

当前工作树在本轮开始时已经是脏状态：`git status --short` 显示 222 行状态，其中约 16 个已修改文件、5 个删除项、201 个未跟踪文件。本轮没有回滚、删除或覆盖这些已有改动；新增报告只记录本轮真实执行结果。

## 读取过的项目上下文

- `AGENTS.md`：确认必须中文回复、最小必要修改、直接修改原文件、不伪造结果、失败要记录。
- `README.md`：确认项目定位是研究验证套件，证据分级包含机制探针、smoke、消融和 held-out test。
- `docs/code_project_case_studies.md`：存在并已读取；重点历史记录包括 NIAH 梯度衰减、JSON validation-first、防 test 选参、batch retrieval mask、retrieval softmax Top-K、extract compose readout 默认关闭等。
- `pyproject.toml`：确认 Python `>=3.10`，核心依赖含 `torch>=2.1`、`numpy>=1.26`、`matplotlib>=3.8`、`swanlab>=0.4`、`datasets>=2.18`，测试工具为 `pytest` 和 `ruff`。
- MHDSRA2 核心实现与入口：`src/dsra/mhdsra2/improved_dsra_mha.py`、`src/dsra/mhdsra2/paged_exact_memory.py`、`src/dsra/dsra_model.py`、`src/dsra/infrastructure/paged_memory_repository.py`、`scripts/verify_mhdsra2.py`、`scripts/mhdsra2_quality_improvement_ablation.py`、`scripts/tiny_llama_compare.py`。

## 当前配置锚点

### 核心 `MHDSRA2Config(dim=128)` 默认值

| 字段 | 当前值 |
|---|---:|
| dim | 128 |
| heads | 8 |
| layers | 单层核心模块；多层由调用方堆叠 |
| slots | 128 |
| read_topk / write_topk | 8 / 4 |
| local_window / chunk_size | 512 / 核心模块不持有 chunk_size |
| use_retrieval | true |
| retrieval_max_tokens | 128 |
| retrieval_attention_topk | null |
| detach_state | true |
| retrieval_query_pooling | mean |
| retrieval_neighbor_span | 0 |
| retrieval_neighbor_direction | right |
| retrieval_quality_gate_bias | 0.0 |

### 多层 NIAH 风格默认入口

`MultiLayerMHDSRA2Model` 默认 `num_layers=2`，默认 `K=128`、`kr=16`、`chunk_size=256`，并构造 `MHDSRA2Config(local_window=chunk_size, use_retrieval=False, detach_state=True)`。这说明多层 token 模型的默认外部 retrieval 是关闭的，只有显式实验才打开。

### Tiny LM 对照默认入口

`scripts/tiny_llama_shared.LMConfig` 默认是 `dim=256`、`heads=4`、`num_layers=6`、`seq_len=512`、`batch_size=8`、`max_steps=50000`。对应 MHDSRA2 tiny LM 层默认 `slots=64`、`read_topk=8`、`write_topk=4`、`local_window=512`、`use_retrieval=False`、`detach_state=False`。

### 设备与随机种子

- 本轮环境探测：Python `3.14.4`，PyTorch `2.11.0+cu130`。
- CUDA：可用，`cuda_device_count=1`，`cuda:0` 为 `NVIDIA GeForce RTX 4070 Laptop GPU`。
- 项目设备策略：若 CUDA 可用使用 `cuda:0`，否则 CPU 回退。
- 本轮显式 seed：`scripts/verify_mhdsra2.py` 内部 `torch.manual_seed(0)`；batch retrieval smoke 使用 `seed=20260602`；JSON smoke 使用 `json-task-seed-roots=7`；tiny LM 脚本未在 CLI 层显式固定 seed，因此仅作为 smoke 链路证据。

## 本轮预算

- 最大假设数量：3。
- 最大 CUDA 时间：约 0.10 小时用于 smoke / sanity，不跑正式多 seed 训练。
- 最大 wall time：约 45 分钟。
- 最大 seed 数：Phase 2 smoke 最多 1 个 seed；未进入 Phase 3。
- 最大上下文长度：本轮 JSON smoke 的 `full_seq_len=32233`；tiny LM smoke `seq_len=32`；MHDSRA2 shape smoke `seq_len=2048`。
- 是否允许下载数据集：不允许。本轮只使用本地已有 `data/wikitext-2/wiki.train.tokens`。

## 已执行结果

| 命令 | 结果 | 证据等级 |
|---|---|---|
| `python -m py_compile src\dsra\mhdsra2\improved_dsra_mha.py src\dsra\mhdsra2\paged_exact_memory.py src\dsra\dsra_model.py src\dsra\infrastructure\paged_memory_repository.py scripts\verify_mhdsra2.py scripts\tiny_llama_baseline.py scripts\tiny_llama_mhdsra2.py scripts\tiny_llama_compare.py scripts\mhdsra2_quality_improvement_ablation.py scripts\json_retrieval_test.py scripts\needle_in_haystack_test.py` | 通过，退出码 0 | Phase 0 编译 |
| `python -m pytest tests\test_tiny_llama_ppl_regressions.py -q` | `4 passed in 13.42s` | PPL 评估口径 |
| `python -m pytest tests\test_diagnostic_gate_policy_regressions.py -k "retrieval_attention_topk or retrieval_mask or gate_quality_bias" -q` | `11 passed, 5 deselected in 13.52s` | Phase 1 机制红线 |
| `python scripts\verify_mhdsra2.py --seq-len 2048 --batch 1 --dim 128 --heads 4 --chunk 64 --slots 32 --read-topk 4 --write-topk 2 --local-window 64 --retrieval-tokens 16 --steps 1 --run-bench` | smoke 通过；估算 attention working-set `171.00 KB`；CPU micro-benchmark `13168.5 tok/s` | Smoke / shape |
| `python -m pytest tests\test_memory_lifecycle_regressions.py -q` | `42 passed in 3.86s` | Phase 1 机制红线 |
| `python -m pytest tests\test_multilayer_retrieval_regressions.py -q` | `6 passed in 3.84s` | Phase 1 调用链 |
| `python scripts\mhdsra2_batch_retrieval_quality_smoke.py --device cpu --tokens 64 --batch-sizes 1,2 --page-size 16 --top-pages 2 --max-tokens 4 --json-out reports\mhdsra2_baseline_batch_retrieval_quality_smoke_cpu.json --markdown-out reports\mhdsra2_baseline_batch_retrieval_quality_smoke_cpu.md` | `passed=True`；6/6 cases 通过；无 cross-sample leak；无 future leak；batch loop positions match | 机制 / smoke |
| `python -X utf8 scripts\tiny_llama_compare.py --seq-len 32 --batch-size 8 --max-steps 1 --dim 16 --heads 4 --layers 1 --device cuda:0` | 成功退出；standard PPL `116.46`，MHDSRA2 PPL `115.24`，ratio `0.990x` | Tiny LM smoke，不是质量结论 |
| `python -m pytest tests -q` | `382 passed, 5 subtests passed in 71.47s`；第一次运行暴露一个测试替身签名不同步问题，修复后通过 | 全量测试验收 |

## 已执行但失败的命令

| 命令 | 失败现象 | 处理 |
|---|---|---|
| PowerShell 中直接运行 `python - <<'PY' ...` 环境探测 | PowerShell 不支持 bash here-doc，报 `ParserError: Missing file specification after redirection operator` | 已用 PowerShell here-string `@' ... '@ \| python -` 重跑成功；失败记录保留在本报告 |
| `python scripts\tiny_llama_compare.py ...` 未加 `-X utf8` | 两个模型均已训练和评估，但最终打印 `✅` 时 Windows GBK 编码报 `UnicodeEncodeError`，退出码 1 | 已用 `python -X utf8` 重跑成功；第一次 PPL 不作为本轮基线数值 |

## 本轮基线指标

### 机制级证据

- Python 编译通过。
- retrieval attention top-k 默认关闭路径、padding mask、fp16 mask、rank-5 token-specific candidate 测试通过。
- paged memory reset、future cutoff、batch-isolated retrieval、neighbor span 安全门、多层 retrieval 调用链测试通过。
- CPU batch retrieval quality smoke：6/6 cases 通过，`no_cross_sample_leak=true`，`no_future_leak=true`，`all_batch_loop_positions_match=true`。

### Smoke 级证据

- MHDSRA2 核心 shape smoke 跑通，输出 shape、slot state shape、local cache 上限、retrieval 分支 finite 均通过。
- Tiny LM 极小 CUDA smoke 跑通，standard attention + RoPE 与 MHDSRA2 都能在同一字符级 WikiText-2 fallback validation 管线上评估 PPL。

### 项目代理指标证据

- 本轮只执行了极小 tiny LM smoke，不执行正式 tiny LM 50000 step 对照。
- 本轮没有执行正式 NIAH、JSON retrieval generalization 多 seed、two-digit retention 多 seed。

### 外部 LLM 证据

- 未执行 MMLU、HellaSwag、ARC、LongBench、RULER。
- 当前项目尚未在本轮验证标准 tokenizer、checkpoint 保存加载、统一生成接口和 lm-evaluation-harness 适配；这些只能作为长期目标。

## 历史参考结果

历史记录显示 `extract_compose_readout` 曾在 mixed-template JSON seed roots `7/11/19/23/29/31` 上复现 validation-first 正信号，并通过 two-digit retention 保持测试；`retrieval_attention_topk` 曾显示 exact-match attention weight 从 `0.117558` 提升到 `0.530058`（topk=16），但任务级 smoke 没有证明 NIAH/JSON 准确率提升。以上是历史参考，不是本轮新执行结果。

## 未执行项与原因

- 未执行 `ruff check .`：未改业务代码；当前目标是实验账本与低成本验证。
- 未执行 tiny LM 50000 step PPL：超出本轮 wall time / CUDA 预算。
- 未执行 NIAH / JSON / two-digit 多 seed Phase 3：本轮 Phase 2 smoke 没有得到足够强的 validation 正信号，按分层协议不越级消耗算力。
- 未下载新数据集：预算声明不允许下载。

## 当前风险

- 工作树已有大量未提交改动，本报告无法证明这些改动整体可发布。
- Tiny LM smoke 未固定 seed，PPL 数值只能证明管线可运行。
- JSON / NIAH 历史报告很多，容易误把 smoke、adapter 或 held-out test 结果当作通用能力；后续必须继续按 validation-first 和证据分级记录。
