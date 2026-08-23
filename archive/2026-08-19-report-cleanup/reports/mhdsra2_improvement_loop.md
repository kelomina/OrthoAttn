# MHDSRA2 Improvement Candidate Loop - 2026-06-30

## 本轮目标和预算

本轮目标是基于当前 DSRA 工作树，按“先机制、再 smoke、再项目代理指标、最后才外部 LLM”的顺序筛选 MHDSRA2 改进候选。为了避免一次引入过多变量，本轮最多提出 3 个候选，并且只有在上一阶段出现足够强的 validation-first 正信号时才进入下一阶段。

预算：

- 最大假设数量：3。
- 最大 CUDA 时间：约 0.10 小时。
- 最大 wall time：约 45 分钟。
- 最大 seed 数：Phase 2 smoke 最多 1 个 seed；未进入 Phase 3。
- 最大上下文长度：JSON smoke `full_seq_len=32233`。
- 是否允许下载数据集：否。

本轮没有修改模型代码，只新增/更新实验报告。当前工作树已有大量非本轮改动，本报告不把那些改动声明为本轮实现。

## 历史失败检查

已读取 `docs/code_project_case_studies.md`。与本轮候选相关的历史经验如下：

- NIAH 历史失败显示，`detach_state=True` 与 slot 写入梯度衰减会让长上下文 NIAH 很难训练，不能把短 smoke 结果说成解决 2M 推理。
- JSON generalization 历史修复强调 validation-first，test 只能在配置固定后看，不能参与选参。
- batch retrieval 历史修复强调 batch 隔离、padding mask、future cutoff 和 fp16 mask finite，不能让 padding token 进入 softmax。
- `extract_compose_readout` 历史上有 mixed-template JSON 多 seed 正信号，但它是默认关闭的任务特化 answer readout adapter，不是通用生成能力。
- `retrieval_attention_topk` 历史上有机制级 exact-match 权重集中证据，但任务级收益尚未证明，不能默认打开。

## 候选 1：结构化 JSON `extract_compose_readout`

### 改进动机

默认逐字生成在结构化 JSON 检索任务上容易“找到证据但答题格式不稳”。`extract_compose_readout` 的思路像是先从证据窗口里定位正确记录，再按固定字段把答案拼出来；它可能改善 JSON 结构化问答的 validation exact / sequence accuracy，但不能说明通用自然语言生成变强。

### 预期改善

希望改善 JSON retrieval generalization 的 validation generation exact match 和 sequence accuracy。判断标准：必须先在 validation 上相对 baseline 出现稳定正信号；held-out test 只能在配置固定后确认。

### 涉及模块

- `scripts/json_retrieval_test.py`
- `scripts/mhdsra2_quality_improvement_ablation.py`
- `tests/test_mhdsra2_quality_improvement_ablation.py`

本轮未改这些文件，只使用当前工作树已有实现验证。

### 风险说明

- 最大风险是把任务特化 readout adapter 误读成通用生成能力。
- 如果 evidence window 预测错，adapter 可能只是更稳定地拼错答案。
- 如果 baseline 组意外调用 adapter，会污染对照。

### 历史失败检查

历史记录明确要求：该 adapter 默认关闭，baseline 不应产生 `extract_then_compose` 指标；不能把它写成默认 MHDSRA2 能力。本轮检查继续保留这个边界。

### Phase 0 / 1 结果

已执行：

```powershell
python -m pytest tests\test_mhdsra2_quality_improvement_ablation.py -k "generation_readout_adapter or extract_compose or evidence_target or quality_ablation_json_row" -q
```

结果：`13 passed, 82 deselected in 3.22s`。

解释：adapter 注册表默认关闭、baseline 不调用结构化 adapter、extract compose 相关 evidence target 和 JSON row 传参测试均通过。

### Phase 2 Smoke / Sanity 结果

已执行：

```powershell
python scripts\mhdsra2_quality_improvement_ablation.py --device cuda:0 --reports-dir reports --report-name mhdsra2_loop_extract_compose_json_smoke_cuda --checkpoint-path reports\mhdsra2_loop_extract_compose_json_smoke_cuda.checkpoint.jsonl --tasks json --groups baseline,extract_compose_readout --json-task-seed-roots 7 --json-epochs 1 --json-eval-interval 1 --json-dim 16 --json-slots 8 --json-read-topk 2 --json-chunk-size 64 --json-train-dataset-size 1 --json-validation-dataset-size 1 --json-test-dataset-size 1 --json-distractor-records-per-case 2 --json-answer-template-mode mixed
```

结果文件：

- `reports/mhdsra2_loop_extract_compose_json_smoke_cuda.json`
- `reports/mhdsra2_loop_extract_compose_json_smoke_cuda.md`

本轮结果：

| group | validation exact | validation seq acc | test exact | test seq acc | 证据等级 |
|---|---:|---:|---:|---:|---|
| baseline | 0.0000 | 0.0076 | 0.0000 | 0.0000 | smoke |
| extract_compose_readout | 0.0000 | 0.0303 | 0.0000 | 0.0308 | smoke |

### 决策

本轮不进入 Phase 3。原因是 1 epoch / 1 case 的 smoke 虽然链路跑通，并且 sequence accuracy 有轻微信号，但 validation exact 仍为 0，证据太弱，不足以消耗多 seed Phase 3 预算。

状态：`实验验证：本轮 smoke 可运行，但未达到候选有效标准；保留历史正结果为参考，后续需要按历史 stage1/extended 配置重跑。`

## 候选 2：`retrieval_attention_topk` 防召回后 softmax 稀释

### 改进动机

分页召回找回候选 token 后，retrieval attention 仍会在所有候选上 softmax。候选池太大时，正确证据权重可能被大量干扰项摊薄。`retrieval_attention_topk` 像是“先把所有候选拿回来，再只让分数最高的一小撮参与投票”，避免直接降低 `retrieval_max_tokens` 导致证据在召回阶段被裁掉。

### 预期改善

机制层希望 exact-match attention weight 上升；项目指标层希望 NIAH / JSON validation 不劣于 baseline，并在候选进入读出链路时改善最终 accuracy。

### 涉及模块

- `src/dsra/mhdsra2/improved_dsra_mha.py`
- `tests/test_diagnostic_gate_policy_regressions.py`

本轮未改这些文件，只验证当前工作树已有实现。

### 风险说明

- hard top-k 会让未入选候选拿不到 retrieval attention 梯度。
- 如果任务瓶颈在 span predictor 或答案读出，而不是 attention 稀释，任务指标可能没有提升。
- 不能把 top-k 和 `retrieval_max_tokens` 混为一谈。

### 历史失败检查

历史记录显示该方向已有机制级成功，但任务级 smoke 未证明 accuracy 提升；默认必须保持 `None`。

### Phase 0 / 1 结果

已执行：

```powershell
python -m pytest tests\test_diagnostic_gate_policy_regressions.py -k "retrieval_attention_topk or retrieval_mask" -q
```

结果：`8 passed, 8 deselected in 3.10s`。

解释：默认不漂移、padding mask、fp16 finite、rank-5 token-specific candidates、非法 top-k 拒绝等机制红线通过。

### 决策

本轮不进入 Phase 2 任务 smoke。原因是历史已经表明它的机制证据强、任务收益未定；在本轮低预算下优先把唯一 Phase 2 名额给 JSON readout adapter。该候选保持“机制有效、任务收益未验证”的状态。

状态：`未验证假设：需要后续比较 retrieval_attention_topk=None/32/16/8 的 NIAH/JSON validation 指标。`

## 候选 3：neighbor/span 类分页召回扩展

### 改进动机

有些任务的 key 和 value 是相邻 token。只召回最高分 key 时，答案 value 可能在旁边却没有进入候选表。neighbor/span 扩展的想法像是“找到页码后顺手把旁边一两行也复印出来”，让 key/value 对更可能同时出现在读出候选中。

### 预期改善

希望改善 NIAH key/value pair 或 span predictor 的候选覆盖率，例如 target value top-token rate、pair candidate rate 和最终 validation accuracy。

### 涉及模块

- `src/dsra/mhdsra2/paged_exact_memory.py`
- `src/dsra/infrastructure/paged_memory_repository.py`
- `src/dsra/dsra_model.py`
- `scripts/mhdsra2_quality_improvement_ablation.py`
- `tests/test_memory_lifecycle_regressions.py`
- `tests/test_mhdsra2_quality_improvement_ablation.py`

本轮未改这些文件，只验证当前工作树已有实现。

### 风险说明

- 扩展邻居可能带来更多噪声，让候选表变大，后续 softmax 或 span predictor 更难。
- 如果不严格做 future cutoff，右邻居扩展可能泄露未来 token。
- batch 内不同样本必须保持隔离，不能把相邻 token 从别的样本带进来。

### 历史失败检查

历史记录显示分页召回曾有 page mean 稀释、batch mask、future cutoff 风险；本轮只把该方向停在安全门验证，不做任务质量声明。

### Phase 0 / 1 结果

已执行：

```powershell
python -m pytest tests\test_memory_lifecycle_regressions.py -k "neighbor_span or batch_max_position or batch_gt_one or future_cutoff" -q
```

结果：`9 passed, 33 deselected in 3.01s`。

解释：neighbor span、batch max_position、batch>1 retrieval、future cutoff 等机制红线通过。

### 决策

本轮不进入 Phase 2。原因是该方向更适合 NIAH/span predictor 任务级验证，预算不足且已有 JSON readout smoke 未形成强正信号。该候选保持“机制安全门通过、任务收益未验证”的状态。

状态：`未验证假设：需要后续 NIAH validation-first 多 seed 验证 pair-aware/page-local 组合是否真正改善。`

## 2026-06-30 追加审计：候选 1 Phase 3 成立

用户要求继续完成文档中的任务并保证最终效果达到预期后，本轮对当前仓库已有的 CUDA multi-seed 报告进行了机器可读审计，而不是停在 1 epoch smoke。审计文件：

- `reports/mhdsra2_readout_adapter_mixed_stage1_cuda.json`
- `reports/mhdsra2_readout_adapter_mixed_extended_cuda.json`
- `reports/mhdsra2_readout_adapter_two_digit_retention_cuda.json`
- `reports/mhdsra2_extract_compose_completed_summary.json`

审计结果：

| group | seeds | validation exact mean | validation exact std | test exact mean | test exact std |
|---|---|---:|---:|---:|---:|
| baseline | 7, 11, 19, 23, 29, 31 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| extract_compose_readout | 7, 11, 19, 23, 29, 31 | 0.9583 | 0.0932 | 1.0000 | 0.0000 |

two-digit retention 报告 `reports/mhdsra2_readout_adapter_two_digit_retention_cuda.json` 中 24 行完成，seeds `101/202/303`，最小 exact match `1.0`，均值 `1.0`。
该 two-digit 结果只作为保持测试，不能作为 `extract_compose_readout` 的算术泛化正向证据。

消融审计：

- `reports/mhdsra2_evidence_retrieval_json_stage1_cuda.json`：baseline、evidence_hit_supervision、learned_retrieval_gate、evidence_plus_gate 的 3 seed validation exact 均为 `0.0`。
- `reports/mhdsra2_readout_adapter_mixed_stage1_cuda.json`：baseline 3 seed validation exact 为 `0.0`，`extract_compose_readout` 为 `1.0/1.0/0.75`。
- 结论：稳定正信号来自“evidence-window 训练路径 + 显式结构化 extract-then-compose readout”的组合，不是单独 evidence loss 或 gate adapter。

当前代码回归验证：

```powershell
python -m pytest tests\test_mhdsra2_quality_improvement_ablation.py -k "generation_readout_adapter or extract_compose or evidence_target or quality_ablation_json_row" -q
# 13 passed, 82 deselected

python -m pytest tests\test_diagnostic_gate_policy_regressions.py -k "retrieval_attention_topk or retrieval_mask" -q
# 8 passed, 8 deselected

python -m pytest tests\test_memory_lifecycle_regressions.py tests\test_multilayer_retrieval_regressions.py -q
# 48 passed
```

决策更新：`extract_compose_readout` 通过 Phase 0/1/2/3，可标记为“候选有效”，但仅限默认关闭的结构化 JSON generation readout adapter。已写入 `reports/mhdsra2_completed_improvements.md`。本轮没有把它改成默认行为，也没有把它描述成通用生成能力。

独立复核：子会话 Kepler 只读审计后未发现当前脚本内 direct cheating、程序化 test 参与选参、gold label / polish / target-case sampling 污染；同时要求保留边界：这是显式启用的任务特化结构化 JSON adapter，不能与 raw byte-by-byte generation 或通用能力混称。

## 本轮停止原因

本轮成功终止。Step 0 基线锚定、Step 1 三个候选与历史失败检查、Step 2 分层验证、Step 3 候选有效记录已完成。候选 1 `extract_compose_readout` 在项目内扩展验证中相对 baseline 稳定优于基线；候选 2/3 保持机制安全门通过但任务收益未验证。

## Smoke 误读风险

本轮最大的误读风险是把以下内容说成模型能力提升：

- tiny LM 1 step PPL：只能说明管线能跑，不能说明 PPL 质量。
- JSON 1 epoch / 1 case smoke：只能说明 adapter 路径能跑，不能证明 JSON retrieval 已提升。
- top-k / neighbor span 机制测试：只能说明张量和 mask 行为符合预期，不能证明任务准确率提升。

## 下一步建议

1. 若继续推进 `extract_compose_readout`，按历史 stage1 配置重跑 `json-task-seed-roots=7,11,19`，再跑 extended seeds `23,29,31`，并保留 two-digit retention。
2. 若推进 `retrieval_attention_topk`，先跑小型 NIAH/JSON validation 对照：`None/32/16/8`，同时记录 target candidate hit、span target value top-token rate 和最终 validation accuracy。
3. 若推进 neighbor/span，优先用 NIAH pair-aware 或 page-local span predictor 配置做 Phase 2，不要只看召回命中率，要看最终 validation-first accuracy。
