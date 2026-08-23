# MHDSRA2 Improvement Loop Final Report - 2026-06-30

## 终止类型

本轮成功终止。前一版报告只依据本轮 1 epoch smoke 保守停止；用户要求继续完成文档任务后，本轮进一步审计当前仓库已有 CUDA multi-seed 报告，并重新运行当前代码的关键回归测试。审计证明 `extract_compose_readout` 在项目内扩展验证中稳定优于 baseline，因此可标记为候选有效。

## 已执行结果

- 基线报告：`reports/mhdsra2_baseline_results.md`
- 基线机器报告：`reports/mhdsra2_baseline_results.json`
- 候选 Loop 记录：`reports/mhdsra2_improvement_loop.md`
- JSON smoke 报告：`reports/mhdsra2_loop_extract_compose_json_smoke_cuda.md`
- JSON smoke 机器报告：`reports/mhdsra2_loop_extract_compose_json_smoke_cuda.json`
- batch retrieval smoke 报告：`reports/mhdsra2_baseline_batch_retrieval_quality_smoke_cpu.md`
- batch retrieval smoke 机器报告：`reports/mhdsra2_baseline_batch_retrieval_quality_smoke_cpu.json`
- 候选有效报告：`reports/mhdsra2_completed_improvements.md`
- multi-seed 汇总：`reports/mhdsra2_extract_compose_completed_summary.json`
- 测试修复：`tests/test_all.py` 中一个 monkeypatch 假函数签名同步当前 `build_disjoint_case_pool` 接口。

## 候选状态

| 候选 | 当前阶段 | 本轮结论 |
|---|---|---|
| `extract_compose_readout` | Phase 3 | 候选有效；仅限默认关闭的结构化 JSON readout adapter |
| `retrieval_attention_topk` | Phase 1 | 机制安全门通过；任务收益未验证 |
| neighbor/span retrieval | Phase 1 | 机制安全门通过；任务收益未验证 |

## 指标相对基线

本轮 1 epoch / 1 case smoke 只证明链路可运行，不能作为质量结论：

| group | validation exact | validation seq acc | test exact | test seq acc |
|---|---:|---:|---:|---:|
| baseline | 0.0000 | 0.0076 | 0.0000 | 0.0000 |
| extract_compose_readout | 0.0000 | 0.0303 | 0.0000 | 0.0308 |

候选有效依据来自已审计的 CUDA multi-seed 项目代理指标：

| group | seeds | validation exact mean | validation exact std | test exact mean | test exact std |
|---|---|---:|---:|---:|---:|
| baseline | 7, 11, 19, 23, 29, 31 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| extract_compose_readout | 7, 11, 19, 23, 29, 31 | 0.9583 | 0.0932 | 1.0000 | 0.0000 |

two-digit retention：`reports/mhdsra2_readout_adapter_two_digit_retention_cuda.json` 中 24 行完成，最小 exact match `1.0`，均值 `1.0`。
边界：two-digit 是保持测试，只说明没有发现该方向破坏当前算术保持测试；不能作为结构化 JSON adapter 的正向算术能力证据。

## 消融与复核

消融汇总见 `reports/mhdsra2_extract_compose_ablation_summary.json`。evidence-only 与 evidence+gate 在 3 seed JSON stage1 中 validation exact 均为 `0.0`，而 `extract_compose_readout` stage1 validation exact mean 为 `0.9167`，extended 3 seed 为 `1.0000`。因此当前成功原因应解释为：模型先预测 evidence window，再由显式结构化 readout parser 抽取并拼接答案；不是单独 evidence loss 或 gate adapter 的通用提升。

独立复核子会话 Kepler 只读审计后结论：没有看到 direct cheating、程序内 test 参与选参、gold label/polish/target-case sampling 污染；但成功结论必须保持窄边界，即“显式启用的结构化 JSON readout adapter 候选有效”，不是默认 MHDSRA2 或通用语言生成能力。

## 当前验收命令

```powershell
python -m pytest tests\test_all.py::test_json_generalization_evaluates_test_only_after_best_selection -q
# 1 passed

python -m pytest tests -q
# 382 passed, 5 subtests passed
```

全量测试第一次运行时发现 `tests/test_all.py` 中 monkeypatch 假函数 `_fake_build_disjoint_case_pool` 未接受当前真实接口新增的 `distractor_records_per_case` / `answer_template_mode` 关键字参数；已用 `**_kwargs` 做最小测试修复，随后失败用例和全量测试均通过。

## 未执行项

- 未重新执行 6 个 80 epoch JSON multi-seed 训练：当前已有机器可读 CUDA 报告已审计；本轮重新运行了当前代码回归测试与 smoke，不把历史训练伪称为本轮训练。
- 未执行完整 tiny LM 50000 step PPL：超出本轮预算。
- 未执行完整外部 LLM 评估：当前不是默认自动 Loop 的强制门槛。

## 风险

- 当前工作树已有大量修改和未跟踪报告，本轮没有接管或回滚这些历史现场。
- 本轮 1 epoch smoke 不能被误读为模型质量结论；候选有效结论来自 multi-seed 项目代理指标。
- `extract_compose_readout` 是任务特化 adapter，不是通用生成能力提升。
- two-digit retention 不能被误读为该 adapter 的算术泛化正向证据。
- `retrieval_attention_topk` 与 neighbor/span 目前只有机制和安全门证据，仍缺项目代理指标验证。

## 下一步

下一步不应把 `extract_compose_readout` 改成默认行为；应把它作为默认关闭的结构化 adapter 候选继续抽象，并在其它结构化任务上重新做 validation-first 多 seed。若要推进其它候选，优先给 `retrieval_attention_topk` 做 NIAH/JSON validation 对照。
