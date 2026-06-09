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
| niah_single_needle | 1 | 256 | 1.000 | 1.000 | 1.000 | True | `[8]` | True | 261.680 | True |
| json_latest_field | 1 | 256 | 1.000 | 1.000 | 1.000 | True | `[8]` | True | 14.789 | True |
| future_cutoff | 1 | 256 | 1.000 | 1.000 | 1.000 | True | `[8]` | True | 15.235 | True |
| niah_single_needle | 1 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8]` | True | 25.257 | True |
| json_latest_field | 1 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8]` | True | 42.047 | True |
| future_cutoff | 1 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8]` | True | 27.014 | True |
| niah_single_needle | 4 | 256 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8]` | True | 37.453 | True |
| json_latest_field | 4 | 256 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8]` | True | 39.879 | True |
| future_cutoff | 4 | 256 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8]` | True | 21.489 | True |
| niah_single_needle | 4 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8]` | True | 62.602 | True |
| json_latest_field | 4 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8]` | True | 74.861 | True |
| future_cutoff | 4 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8]` | True | 92.530 | True |
| niah_single_needle | 8 | 256 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8, 8, 8, 8, 8]` | True | 84.139 | True |
| json_latest_field | 8 | 256 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8, 8, 8, 8, 8]` | True | 87.033 | True |
| future_cutoff | 8 | 256 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8, 8, 8, 8, 8]` | True | 83.401 | True |
| niah_single_needle | 8 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8, 8, 8, 8, 8]` | True | 159.770 | True |
| json_latest_field | 8 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8, 8, 8, 8, 8]` | True | 127.877 | True |
| future_cutoff | 8 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8, 8, 8, 8, 8]` | True | 85.809 | True |

## Model Call Chain

- passed: `True`
- retrieval_call_count: `6`
- mask_call_count: `6`
- logits_shape: `[4, 48]`

说明：本报告验证 external paged memory 的 batch 隔离、召回位置、latest-wins 和多层调用链 mask 传递；它不是训练后的 NIAH/JSON 任务准确率报告。
