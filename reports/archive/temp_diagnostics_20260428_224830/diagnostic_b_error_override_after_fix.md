# MHDSRA2 vs DSRA Next-Round Benchmark

## Config
- benchmark: `diagnostic_b_error_override_after_fix`
- diagnostic_device: `cpu`
- diagnostic_slots: `16`
- diagnostic_key_count: `64`
- diagnostic_value_count: `64`
- diagnostic_chunk_size: `256`
- diagnostic_page_size: `128`
- diagnostic_retrieved_top_pages: `4`
- diagnostic_retrieved_max_tokens: `64`
- diagnostic_override_seq_len: `16384`
- diagnostic_override_gap_grid: `[128, 1024, 4096]`

## Summary
- total rows: `2`
- valid rows: `2`
- MHDSRA2 wins: `2`
- DSRA wins: `0`
- ties: `0`
- average delta (MHDSRA2-DSRA): `0.666667`
- average ratio (MHDSRA2/DSRA): `3.000000`

## Suite Breakdown

| Suite | Rows | MHDSRA2 Wins | DSRA Wins | Ties | Avg Delta | Avg Ratio |
|---|---:|---:|---:|---:|---:|---:|
| diagnostic_b_error_override | 2 | 2 | 0 | 0 | 0.666667 | 3.000000 |

## Diagnostic B - Error Override

Writes an old fact, then a correction for the same key, and measures whether the latest fact overrides stale memory.

| Task | Split | Metric | DSRA | MHDSRA2 | Delta | Ratio | Winner | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| aggregate | overall | latest_fact_accuracy.mhdsra2_without_paged_recall | 0.333333 | 1.000000 | 0.666667 | 3.000000 | mhdsra2 |  |
| aggregate | overall | latest_fact_accuracy.mhdsra2_with_paged_recall | 0.333333 | 1.000000 | 0.666667 | 3.000000 | mhdsra2 |  |

### Mean metrics by model

| Model | successful_cases | oom_cases | confidence_margin | latest_fact_accuracy | stale_fact_rate | forget_gate_mean |
|---|---|---|---|---|---|---|
| Original DSRA | 3.0000 | 0.0000 | 0.0000 | 0.3333 | 0.6667 | - |
| MH-DSRA-v2 (no paged recall) | 3.0000 | 0.0000 | 0.1842 | 1.0000 | 0.0000 | 0.0166 |
| MH-DSRA-v2 (paged recall) | 3.0000 | 0.0000 | 0.0851 | 1.0000 | 0.0000 | 0.0166 |
| Sliding window attention | 3.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | - |
| Linear attention | 3.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | - |

### Per-case exact success

| Case | Original DSRA | MH-DSRA-v2 (no paged recall) | MH-DSRA-v2 (paged recall) | Sliding window attention | Linear attention |
|---|---|---|---|---|---|
| error_override.gap_128 | 0 | 1 | 1 | 0 | 0 |
| error_override.gap_1024 | 0 | 1 | 1 | 0 | 0 |
| error_override.gap_4096 | 1 | 1 | 1 | 0 | 0 |
