# MHDSRA2 vs DSRA Next-Round Benchmark

## Config
- seed: `7`
- niah_seq_lengths: `[8192, 16384]`
- niah_dim: `64`
- niah_num_layers: `2`
- niah_slots: `64`
- niah_read_topk: `8`
- json_epochs: `80`
- json_dim: `128`
- json_slots: `128`
- json_read_topk: `32`
- json_chunk_size: `256`
- json_lr: `0.0005`
- json_generalization_score_mode: `teacher_forced`
- json_local_context_size: `4`
- json_local_context_mode: `concat`

## Summary
- total rows: `10`
- valid rows: `10`
- MHDSRA2 wins: `3`
- DSRA wins: `1`
- ties: `6`
- average delta (MHDSRA2-DSRA): `0.042125`
- average ratio (MHDSRA2/DSRA): `1.140790`

## Suite Breakdown

| Suite | Rows | MHDSRA2 Wins | DSRA Wins | Ties | Avg Delta | Avg Ratio |
|---|---:|---:|---:|---:|---:|---:|
| json_retrieval_generalization | 8 | 3 | 1 | 4 | 0.052656 | 1.211185 |
| needle_in_haystack | 2 | 0 | 0 | 2 | 0.000000 | 1.000000 |

## Needle In Haystack

Uses the `run_single_niah_test()` best accuracy criterion from `needle_in_haystack_test.py`.

| Task | Split | Metric | DSRA | MHDSRA2 | Delta | Ratio | Winner | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| seq_len=8192 | overall | best_accuracy | 1.000000 | 1.000000 | 0.000000 | 1.000000 | tie |  |
| seq_len=16384 | overall | best_accuracy | 1.000000 | 1.000000 | 0.000000 | 1.000000 | tie |  |

## JSON Retrieval Generalization

Uses pooled validation/test accuracy metrics from `run_json_retrieval_generalization_test()` covering teacher-forced and generation evaluation.

| Task | Split | Metric | DSRA | MHDSRA2 | Delta | Ratio | Winner | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| museum_artifact_generalization | validation | teacher_forced_exact_match_rate | 0.000000 | 0.000000 | 0.000000 | NA | tie |  |
| museum_artifact_generalization | validation | teacher_forced_mean_sequence_accuracy | 0.442495 | 0.680818 | 0.238323 | 1.538590 | mhdsra2 |  |
| museum_artifact_generalization | validation | generation_exact_match_rate | 0.000000 | 0.000000 | 0.000000 | NA | tie |  |
| museum_artifact_generalization | validation | generation_mean_sequence_accuracy | 0.075499 | 0.077639 | 0.002140 | 1.028346 | mhdsra2 |  |
| museum_artifact_generalization | test | teacher_forced_exact_match_rate | 0.000000 | 0.000000 | 0.000000 | NA | tie |  |
| museum_artifact_generalization | test | teacher_forced_mean_sequence_accuracy | 0.432718 | 0.628715 | 0.195997 | 1.452944 | mhdsra2 |  |
| museum_artifact_generalization | test | generation_exact_match_rate | 0.000000 | 0.000000 | 0.000000 | NA | tie |  |
| museum_artifact_generalization | test | generation_mean_sequence_accuracy | 0.086838 | 0.071630 | -0.015209 | 0.824863 | dsra |  |
