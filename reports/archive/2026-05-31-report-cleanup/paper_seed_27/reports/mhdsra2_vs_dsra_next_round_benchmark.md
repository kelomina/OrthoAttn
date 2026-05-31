# MHDSRA2 vs DSRA Next-Round Benchmark

## Config
- seed: `27`
- niah_seq_lengths: `[8192, 16384, 32768, 65536, 131072, 262144]`
- niah_dim: `64`
- niah_num_layers: `2`
- niah_slots: `64`
- niah_read_topk: `8`
- json_epochs: `80`
- json_dim: `128`
- json_slots: `128`
- json_read_topk: `32`
- json_chunk_size: `512`
- json_lr: `0.0005`
- json_generalization_score_mode: `teacher_forced`
- json_local_context_size: `4`
- json_local_context_mode: `concat`
- diagnostic_device: `cuda`
- diagnostic_slots: `16`
- diagnostic_key_count: `64`
- diagnostic_value_count: `64`
- diagnostic_chunk_size: `256`
- diagnostic_page_size: `128`
- diagnostic_exact_seq_len: `2000000`
- diagnostic_exact_fact_spacing: `1024`
- diagnostic_override_seq_len: `16384`
- diagnostic_override_gap_grid: `[128, 1024, 4096]`
- diagnostic_fixation_seq_len: `65536`
- diagnostic_fixation_distractor_grid: `[16, 64, 256]`

## Summary
- total rows: `20`
- valid rows: `19`
- MHDSRA2 wins: `5`
- DSRA wins: `4`
- ties: `10`
- average delta (MHDSRA2-DSRA): `-0.011342`
- average ratio (MHDSRA2/DSRA): `1.137164`

## Suite Breakdown

| Suite | Rows | MHDSRA2 Wins | DSRA Wins | Ties | Avg Delta | Avg Ratio |
|---|---:|---:|---:|---:|---:|---:|
| diagnostic_a_exact_recall | 2 | 0 | 2 | 0 | -0.333333 | 0.000000 |
| diagnostic_b_error_override | 2 | 0 | 2 | 0 | -0.666667 | 0.000000 |
| diagnostic_c_anti_fixation | 2 | 0 | 0 | 2 | 0.000000 | 0.000000 |
| json_retrieval_generalization | 8 | 4 | 0 | 4 | 0.098062 | 2.411493 |
| needle_in_haystack | 5 | 1 | 0 | 4 | 0.200000 | 1.000000 |

## Needle In Haystack

Uses the `run_single_niah_test()` best accuracy criterion from `needle_in_haystack_test.py`.

| Task | Split | Metric | DSRA | MHDSRA2 | Delta | Ratio | Winner | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| seq_len=8192 | overall | best_accuracy | 1.000000 | 1.000000 | 0.000000 | 1.000000 | tie |  |
| seq_len=16384 | overall | best_accuracy | 1.000000 | 1.000000 | 0.000000 | 1.000000 | tie |  |
| seq_len=32768 | overall | best_accuracy | 1.000000 | 1.000000 | 0.000000 | 1.000000 | tie |  |
| seq_len=65536 | overall | best_accuracy | 1.000000 | 1.000000 | 0.000000 | 1.000000 | tie |  |
| seq_len=131072 | overall | best_accuracy | NA | 1.000000 | NA | NA | missing | OOM |
| seq_len=262144 | overall | best_accuracy | 0.000000 | 1.000000 | 1.000000 | NA | mhdsra2 |  |

## JSON Retrieval Generalization

Uses pooled validation/test accuracy metrics from `run_json_retrieval_generalization_test()` covering teacher-forced and generation evaluation.

| Task | Split | Metric | DSRA | MHDSRA2 | Delta | Ratio | Winner | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| museum_artifact_generalization | validation | teacher_forced_exact_match_rate | 0.000000 | 0.000000 | 0.000000 | NA | tie |  |
| museum_artifact_generalization | validation | teacher_forced_mean_sequence_accuracy | 0.500247 | 0.654008 | 0.153761 | 1.307370 | mhdsra2 |  |
| museum_artifact_generalization | validation | generation_exact_match_rate | 0.000000 | 0.000000 | 0.000000 | NA | tie |  |
| museum_artifact_generalization | validation | generation_mean_sequence_accuracy | 0.097796 | 0.308308 | 0.210512 | 3.152562 | mhdsra2 |  |
| museum_artifact_generalization | test | teacher_forced_exact_match_rate | 0.000000 | 0.000000 | 0.000000 | NA | tie |  |
| museum_artifact_generalization | test | teacher_forced_mean_sequence_accuracy | 0.494512 | 0.677012 | 0.182500 | 1.369050 | mhdsra2 |  |
| museum_artifact_generalization | test | generation_exact_match_rate | 0.000000 | 0.000000 | 0.000000 | NA | tie |  |
| museum_artifact_generalization | test | generation_mean_sequence_accuracy | 0.084389 | 0.322113 | 0.237724 | 3.816990 | mhdsra2 |  |

## Diagnostic A - Exact Recall

Streams sparse fact tokens through long filler contexts and checks whether the queried value is recovered exactly.

| Task | Split | Metric | DSRA | MHDSRA2 | Delta | Ratio | Winner | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| aggregate | overall | exact_match_rate.mhdsra2_without_paged_recall | 0.333333 | 0.000000 | -0.333333 | 0.000000 | dsra |  |
| aggregate | overall | exact_match_rate.mhdsra2_with_paged_recall | 0.333333 | 0.000000 | -0.333333 | 0.000000 | dsra |  |

### Mean metrics by model

| Model | successful_cases | oom_cases | confidence_margin | exact_match_rate |
|---|---|---|---|---|
| Original DSRA | 3.0000 | 0.0000 | 0.0281 | 0.3333 |
| MH-DSRA-v2 (no paged recall) | 3.0000 | 0.0000 | 0.0000 | 0.0000 |
| MH-DSRA-v2 (paged recall) | 3.0000 | 0.0000 | 0.0000 | 0.0000 |
| Sliding window attention | 3.0000 | 0.0000 | 0.0000 | 0.0000 |
| Linear attention | 3.0000 | 0.0000 | 0.0013 | 1.0000 |

### Per-case exact success

| Case | Original DSRA | MH-DSRA-v2 (no paged recall) | MH-DSRA-v2 (paged recall) | Sliding window attention | Linear attention |
|---|---|---|---|---|---|
| exact_recall.early | 0 | 0 | 0 | 0 | 1 |
| exact_recall.middle | 0 | 0 | 0 | 0 | 1 |
| exact_recall.late | 1 | 0 | 0 | 0 | 1 |

## Diagnostic B - Error Override

Writes an old fact, then a correction for the same key, and measures whether the latest fact overrides stale memory.

| Task | Split | Metric | DSRA | MHDSRA2 | Delta | Ratio | Winner | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| aggregate | overall | latest_fact_accuracy.mhdsra2_without_paged_recall | 0.666667 | 0.000000 | -0.666667 | 0.000000 | dsra |  |
| aggregate | overall | latest_fact_accuracy.mhdsra2_with_paged_recall | 0.666667 | 0.000000 | -0.666667 | 0.000000 | dsra |  |

### Mean metrics by model

| Model | successful_cases | oom_cases | confidence_margin | latest_fact_accuracy | stale_fact_rate | forget_gate_mean |
|---|---|---|---|---|---|---|
| Original DSRA | 3.0000 | 0.0000 | 0.0000 | 0.6667 | 0.3333 | - |
| MH-DSRA-v2 (no paged recall) | 3.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0033 |
| MH-DSRA-v2 (paged recall) | 3.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0033 |
| Sliding window attention | 3.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | - |
| Linear attention | 3.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | - |

### Per-case exact success

| Case | Original DSRA | MH-DSRA-v2 (no paged recall) | MH-DSRA-v2 (paged recall) | Sliding window attention | Linear attention |
|---|---|---|---|---|---|
| error_override.gap_128 | 1 | 0 | 0 | 0 | 0 |
| error_override.gap_1024 | 1 | 0 | 0 | 0 | 0 |
| error_override.gap_4096 | 0 | 0 | 0 | 0 | 0 |

## Diagnostic C - Anti-fixation

Builds a dominant wrong pattern plus one rare counterexample, then checks whether the model resists majority fixation.

| Task | Split | Metric | DSRA | MHDSRA2 | Delta | Ratio | Winner | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| aggregate | overall | counterexample_accuracy.mhdsra2_without_paged_recall | 0.000000 | 0.000000 | 0.000000 | NA | tie |  |
| aggregate | overall | counterexample_accuracy.mhdsra2_with_paged_recall | 0.000000 | 0.000000 | 0.000000 | NA | tie |  |

### Mean metrics by model

| Model | successful_cases | oom_cases | confidence_margin | counterexample_accuracy | majority_trap_rate |
|---|---|---|---|---|---|
| Original DSRA | 3.0000 | 0.0000 | 0.0498 | 0.0000 | 1.0000 |
| MH-DSRA-v2 (no paged recall) | 3.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| MH-DSRA-v2 (paged recall) | 3.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| Sliding window attention | 3.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| Linear attention | 3.0000 | 0.0000 | 1.0223 | 0.0000 | 1.0000 |

### Per-case exact success

| Case | Original DSRA | MH-DSRA-v2 (no paged recall) | MH-DSRA-v2 (paged recall) | Sliding window attention | Linear attention |
|---|---|---|---|---|---|
| anti_fixation.distractors_16 | 0 | 0 | 0 | 0 | 0 |
| anti_fixation.distractors_64 | 0 | 0 | 0 | 0 | 0 |
| anti_fixation.distractors_256 | 0 | 0 | 0 | 0 | 0 |
