# MHDSRA2 Completed Improvements

## 2026-06-30 - `extract_compose_readout` 结构化 JSON 读出候选有效

### 候选结论

`extract_compose_readout` 可以标记为“候选有效”，但只限于默认关闭的结构化 JSON generation readout adapter。它验证的是“先预测 evidence window，再用严格 parser 抽取字段并拼接答案”的结构化检索答案路径，不是通用自然语言逐字生成能力提升，也不应改成默认 MHDSRA2 行为。

### 证据分级

- 机制级证据：当前代码的 adapter 默认关闭、baseline 不调用 adapter、evidence target 与 mixed-template parser 回归测试通过。
- Smoke 级证据：本轮新增 `reports/mhdsra2_loop_extract_compose_json_smoke_cuda.json`，证明当前路径可运行，但这份 1 epoch / 1 case smoke 不作为质量结论。
- 项目代理指标证据：已审计当前仓库已有 CUDA multi-seed JSON 报告和 two-digit retention 报告，满足 Phase 3 的多 seed 与保持测试要求。
- 外部 LLM 证据：未执行，不作为本轮验收门槛。

### Phase 0 / 1 当前验证

本轮重新执行：

```powershell
python -m pytest tests\test_mhdsra2_quality_improvement_ablation.py -k "generation_readout_adapter or extract_compose or evidence_target or quality_ablation_json_row" -q
# 13 passed, 82 deselected

python -m pytest tests\test_diagnostic_gate_policy_regressions.py -k "retrieval_attention_topk or retrieval_mask" -q
# 8 passed, 8 deselected

python -m pytest tests\test_memory_lifecycle_regressions.py tests\test_multilayer_retrieval_regressions.py -q
# 48 passed

python -m py_compile scripts\json_retrieval_test.py scripts\mhdsra2_quality_improvement_ablation.py tests\test_mhdsra2_quality_improvement_ablation.py src\dsra\mhdsra2\improved_dsra_mha.py src\dsra\mhdsra2\paged_exact_memory.py src\dsra\dsra_model.py
# passed

python -m pytest tests\test_all.py::test_json_generalization_evaluates_test_only_after_best_selection -q
# 1 passed

python -m pytest tests -q
# 382 passed, 5 subtests passed
```

### Phase 2 / 3 项目代理指标

本轮审计的可信报告：

- `reports/mhdsra2_readout_adapter_mixed_stage1_cuda.json`
- `reports/mhdsra2_readout_adapter_mixed_extended_cuda.json`
- `reports/mhdsra2_readout_adapter_two_digit_retention_cuda.json`
- 汇总文件：`reports/mhdsra2_extract_compose_completed_summary.json`
- 消融汇总：`reports/mhdsra2_extract_compose_ablation_summary.json`

JSON mixed-template multi-seed 汇总：

| group | seeds | validation exact mean | validation exact std | test exact mean | test exact std |
|---|---|---:|---:|---:|---:|
| baseline | 7, 11, 19, 23, 29, 31 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| extract_compose_readout | 7, 11, 19, 23, 29, 31 | 0.9583 | 0.0932 | 1.0000 | 0.0000 |

逐 seed validation exact：

- baseline: `0.0 / 0.0 / 0.0 / 0.0 / 0.0 / 0.0`
- extract_compose_readout: `1.0 / 1.0 / 0.75 / 1.0 / 1.0 / 1.0`

two-digit retention：

- 报告：`reports/mhdsra2_readout_adapter_two_digit_retention_cuda.json`
- rows: `24`
- seeds: `101, 202, 303`
- min exact match: `1.0`
- mean exact match: `1.0`
- std: `0.0`
- 边界：这是 two-digit baseline holdout 保持测试，只说明当前算术保持测试没有回退；不能作为 `extract_compose_readout` 的算术泛化或正向能力证据。

### 消融证据

已审计 `reports/mhdsra2_extract_compose_ablation_summary.json`：

| report | group | seeds | validation exact mean | validation exact std |
|---|---|---:|---:|---:|
| `mhdsra2_evidence_retrieval_json_stage1_cuda` | baseline | 3 | 0.0000 | 0.0000 |
| `mhdsra2_evidence_retrieval_json_stage1_cuda` | evidence_hit_supervision | 3 | 0.0000 | 0.0000 |
| `mhdsra2_evidence_retrieval_json_stage1_cuda` | evidence_plus_gate | 3 | 0.0000 | 0.0000 |
| `mhdsra2_readout_adapter_mixed_stage1_cuda` | baseline | 3 | 0.0000 | 0.0000 |
| `mhdsra2_readout_adapter_mixed_stage1_cuda` | extract_compose_readout | 3 | 0.9167 | 0.1179 |

解释：evidence-only 与 evidence+gate 没有带来 JSON exact 提升；稳定正信号来自显式结构化 `extract_then_compose` readout 和 evidence-window 训练路径的组合。因此本候选不应被拆成“只加 evidence loss 就有效”。

### 默认行为与污染检查

- baseline 的 `generation_readout_mode` 为 `model`。
- extract group 的 `generation_readout_mode` 为 `extract_then_compose`。
- baseline 的 `validation_extract_then_compose_*` 与 `test_extract_then_compose_*` 诊断字段均为 `null`，说明 baseline 没有触发结构化 adapter。
- 报告中的 `generalization_score_mode` 为 `generation`，不是 teacher-forced。
- test 指标仅在报告中作为 held-out 确认，不作为选参依据。
- 独立复核子会话 Kepler 只读审计后未发现当前脚本内 direct cheating、程序化 test 参与选参、gold label / polish / target-case sampling 污染正式结果；但复核也指出无法证明历史上人没有看过多轮 test 后调整候选方向，因此后续继续调参应换新 held-out seed / pair split。

### 修改文件与调用链

本轮没有修改模型实现代码。为完成全量测试验收，更新了一个测试替身签名：

- `tests/test_all.py`：`test_json_generalization_evaluates_test_only_after_best_selection` 中的 `_fake_build_disjoint_case_pool` 增加 `**_kwargs`，使 monkeypatch 假函数兼容当前真实 `build_disjoint_case_pool(..., distractor_records_per_case=..., answer_template_mode=...)` 接口。该修复只影响测试替身，不改变业务逻辑。

候选所依赖的当前工作树模块包括：

- 上游调用方：`scripts/mhdsra2_quality_improvement_ablation.py` 的 JSON ablation row；`scripts/json_retrieval_test.py` 的 generation evaluation。
- 核心能力：`extract_compose_readout` 通过 evidence-window decoder 与 deterministic extract-then-compose parser 工作。
- 下游报告：`reports/mhdsra2_readout_adapter_mixed_stage1_cuda.*`、`reports/mhdsra2_readout_adapter_mixed_extended_cuda.*`、`reports/mhdsra2_readout_adapter_two_digit_retention_cuda.*`。

### 未执行项

- 未重新跑 6 个 80 epoch CUDA JSON rows：当前已有机器可读报告已审计；本轮补跑了当前代码回归测试和 1 epoch smoke，避免把历史训练伪称为本轮训练。
- 未执行外部 LLM benchmark：当前不是默认自动 Loop 门槛。
- 未改默认配置：保持 adapter 默认关闭。

### 风险与边界

- 这是任务特化结构化 readout，不是通用生成能力。
- 不能据此宣称 MHDSRA2 已解决长上下文推理。
- 不能把 adapter 后的 `generation_exact_match` 与 raw model byte-by-byte generation 混称。
- adapter 只能读取模型预测的 evidence window，不能退化为全上下文搜索、metadata 查表或 expected-answer 查表。
- 继续维护因果性边界：answer-start hidden state 不能看到后续 gold answer。
- 如果未来把 adapter 抽象到其它结构化任务，必须重新做 validation-first 多 seed 和 held-out test。
- 如果未来想改默认行为，必须另起任务并明确获得用户同意。
