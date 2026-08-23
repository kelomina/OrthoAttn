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
| niah_single_needle | 1 | 256 | 1.000 | 1.000 | 1.000 | True | `[8]` | True | 103.126 | True |
| json_latest_field | 1 | 256 | 1.000 | 1.000 | 1.000 | True | `[8]` | True | 6.527 | True |
| future_cutoff | 1 | 256 | 1.000 | 1.000 | 1.000 | True | `[8]` | True | 5.247 | True |
| niah_single_needle | 1 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8]` | True | 13.197 | True |
| json_latest_field | 1 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8]` | True | 10.990 | True |
| future_cutoff | 1 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8]` | True | 11.933 | True |
| niah_single_needle | 4 | 256 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8]` | True | 18.399 | True |
| json_latest_field | 4 | 256 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8]` | True | 19.572 | True |
| future_cutoff | 4 | 256 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8]` | True | 15.754 | True |
| niah_single_needle | 4 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8]` | True | 43.110 | True |
| json_latest_field | 4 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8]` | True | 42.586 | True |
| future_cutoff | 4 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8]` | True | 39.881 | True |
| niah_single_needle | 8 | 256 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8, 8, 8, 8, 8]` | True | 31.795 | True |
| json_latest_field | 8 | 256 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8, 8, 8, 8, 8]` | True | 40.495 | True |
| future_cutoff | 8 | 256 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8, 8, 8, 8, 8]` | True | 29.368 | True |
| niah_single_needle | 8 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8, 8, 8, 8, 8]` | True | 77.283 | True |
| json_latest_field | 8 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8, 8, 8, 8, 8]` | True | 74.079 | True |
| future_cutoff | 8 | 1024 | 1.000 | 1.000 | 1.000 | True | `[8, 8, 8, 8, 8, 8, 8, 8]` | True | 69.084 | True |

## Model Call Chain

- passed: `True`
- retrieval_call_count: `6`
- mask_call_count: `6`
- logits_shape: `[4, 48]`

说明：本报告验证 external paged memory 的 batch 隔离、召回位置、latest-wins 和多层调用链 mask 传递；它不是训练后的 NIAH/JSON 任务准确率报告。
