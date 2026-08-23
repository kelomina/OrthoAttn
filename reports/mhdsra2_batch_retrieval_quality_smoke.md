# MHDSRA2 Batch Retrieval Quality Smoke

- device: `cuda:0`
- seed: `20260602`
- passed: `True`
- cases: `18/18`
- no_cross_sample_leak: `True`
- no_future_leak: `True`
- batch_loop_positions_match: `True`

## Cases

| scenario | B | T | hit | top1 | marker | owner | counts | loop_match | ms | passed |
|---|---:|---:|---:|---:|---:|---|---|---|---:|---|
| niah_single_needle | 1 | 256 | 1.000 | 1.000 | 1.000 | True | `[8]` | True | 72.334 | True |
| json_latest_field | 1 | 256 | 1.000 | 1.000 | 1.000 | True | `[8]` | True | 2.746 | True |
| future_cutoff | 1 | 256 | 1.000 | 1.000 | 1.000 | True | `[8]` | True | 2.309 | True |
| niah_single_needle | 1 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8]` | True | 7.521 | True |
| json_latest_field | 1 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8]` | True | 5.813 | True |
| future_cutoff | 1 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8]` | True | 5.932 | True |
| niah_single_needle | 4 | 256 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8]` | True | 8.526 | True |
| json_latest_field | 4 | 256 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8]` | True | 9.422 | True |
| future_cutoff | 4 | 256 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8]` | True | 8.319 | True |
| niah_single_needle | 4 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8]` | True | 21.783 | True |
| json_latest_field | 4 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8]` | True | 23.461 | True |
| future_cutoff | 4 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8]` | True | 18.387 | True |
| niah_single_needle | 8 | 256 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8, 8, 8, 8, 8]` | True | 16.404 | True |
| json_latest_field | 8 | 256 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8, 8, 8, 8, 8]` | True | 15.065 | True |
| future_cutoff | 8 | 256 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8, 8, 8, 8, 8]` | True | 13.976 | True |
| niah_single_needle | 8 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8, 8, 8, 8, 8]` | True | 38.061 | True |
| json_latest_field | 8 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8, 8, 8, 8, 8]` | True | 35.703 | True |
| future_cutoff | 8 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8, 8, 8, 8, 8]` | True | 41.511 | True |

## Model Call Chain

- passed: `True`
- retrieval_call_count: `6`
- mask_call_count: `6`
- logits_shape: `[4, 48]`

说明：本报告验证 external paged memory 的 batch 隔离、召回位置、latest-wins 和多层调用链 mask 传递；它不是训练后的 NIAH/JSON 任务准确率报告。
