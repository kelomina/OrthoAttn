# MHDSRA2 vs DSRA Next-Round Benchmark

## Config
- benchmark: `diagnostic_c_anti_fixation_after_fix_cuda`
- diagnostic_device: `cuda`
- diagnostic_slots: `16`
- diagnostic_key_count: `64`
- diagnostic_value_count: `64`
- diagnostic_chunk_size: `256`
- diagnostic_page_size: `128`
- diagnostic_retrieved_top_pages: `4`
- diagnostic_retrieved_max_tokens: `64`
- diagnostic_fixation_seq_len: `65536`
- diagnostic_fixation_distractor_grid: `[16, 64, 256]`

## Summary
- total rows: `2`
- valid rows: `2`
- MHDSRA2 wins: `2`
- DSRA wins: `0`
- ties: `0`
- average delta (MHDSRA2-DSRA): `0.666667`
- average ratio (MHDSRA2/DSRA): `0.000000`

## Suite Breakdown

| Suite | Rows | MHDSRA2 Wins | DSRA Wins | Ties | Avg Delta | Avg Ratio |
|---|---:|---:|---:|---:|---:|---:|
| diagnostic_c_anti_fixation | 2 | 2 | 0 | 0 | 0.666667 | 0.000000 |

## Diagnostic C - Anti-fixation

Builds a dominant wrong pattern plus one rare counterexample, then checks whether the model resists majority fixation.

| Task | Split | Metric | DSRA | MHDSRA2 | Delta | Ratio | Winner | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| aggregate | overall | counterexample_accuracy.mhdsra2_without_paged_recall | 0.000000 | 0.333333 | 0.333333 | NA | mhdsra2 |  |
| aggregate | overall | counterexample_accuracy.mhdsra2_with_paged_recall | 0.000000 | 1.000000 | 1.000000 | NA | mhdsra2 |  |

### Mean metrics by model

| Model | successful_cases | oom_cases | confidence_margin | counterexample_accuracy | majority_trap_rate |
|---|---|---|---|---|---|
| Original DSRA | 3.0000 | 0.0000 | 0.0498 | 0.0000 | 1.0000 |
| MH-DSRA-v2 (no paged recall) | 3.0000 | 0.0000 | 0.6697 | 0.3333 | 0.6667 |
| MH-DSRA-v2 (paged recall) | 3.0000 | 0.0000 | 0.8144 | 1.0000 | 0.0000 |
| Sliding window attention | 3.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| Linear attention | 3.0000 | 0.0000 | 1.0223 | 0.0000 | 1.0000 |

### Per-case exact success

| Case | Original DSRA | MH-DSRA-v2 (no paged recall) | MH-DSRA-v2 (paged recall) | Sliding window attention | Linear attention |
|---|---|---|---|---|---|
| anti_fixation.distractors_16 | 0 | 1 | 1 | 0 | 0 |
| anti_fixation.distractors_64 | 0 | 0 | 1 | 0 | 0 |
| anti_fixation.distractors_256 | 0 | 0 | 1 | 0 | 0 |
