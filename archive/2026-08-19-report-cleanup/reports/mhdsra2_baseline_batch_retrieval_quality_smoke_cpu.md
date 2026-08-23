# MHDSRA2 Batch Retrieval Quality Smoke

- device: `cpu`
- seed: `20260602`
- passed: `True`
- cases: `6/6`
- no_cross_sample_leak: `True`
- no_future_leak: `True`
- batch_loop_positions_match: `True`

## Cases

| scenario | B | T | hit | top1 | marker | owner | counts | loop_match | ms | passed |
|---|---:|---:|---:|---:|---:|---|---|---|---:|---|
| niah_single_needle | 1 | 64 | 1.000 | 1.000 | 1.000 | True | `[4]` | True | 13.461 | True |
| json_latest_field | 1 | 64 | 1.000 | 1.000 | 1.000 | True | `[4]` | True | 2.339 | True |
| future_cutoff | 1 | 64 | 1.000 | 1.000 | 1.000 | True | `[4]` | True | 1.851 | True |
| niah_single_needle | 2 | 64 | 1.000 | 1.000 | 1.000 | True | `[4, 4]` | True | 4.060 | True |
| json_latest_field | 2 | 64 | 1.000 | 1.000 | 1.000 | True | `[4, 4]` | True | 3.410 | True |
| future_cutoff | 2 | 64 | 1.000 | 1.000 | 1.000 | True | `[4, 4]` | True | 4.036 | True |

## Model Call Chain

- passed: `True`
- retrieval_call_count: `6`
- mask_call_count: `6`
- logits_shape: `[4, 48]`

说明：本报告验证 external paged memory 的 batch 隔离、召回位置、latest-wins 和多层调用链 mask 传递；它不是训练后的 NIAH/JSON 任务准确率报告。
