# MHDSRA2 vs DSRA Next-Round Benchmark

## Config
- diagnostic_device: `cuda`
- diagnostic_slots: `16`
- diagnostic_key_count: `64`
- diagnostic_value_count: `64`
- diagnostic_chunk_size: `256`
- diagnostic_page_size: `128`
- diagnostic_retrieved_top_pages: `4`
- diagnostic_retrieved_max_tokens: `64`
- diagnostic_exact_seq_len: `2000000`
- diagnostic_exact_fact_spacing: `1024`
- diagnostic_override_seq_len: `16384`
- diagnostic_override_gap_grid: `[128, 1024, 4096]`
- diagnostic_fixation_seq_len: `65536`
- diagnostic_fixation_distractor_grid: `[16, 64, 256]`

## Summary
- total rows: `2`
- valid rows: `2`
- MHDSRA2 wins: `1`
- DSRA wins: `1`
- ties: `0`
- average delta (MHDSRA2-DSRA): `0.166667`
- average ratio (MHDSRA2/DSRA): `1.500000`

## Suite Breakdown

| Suite | Rows | MHDSRA2 Wins | DSRA Wins | Ties | Avg Delta | Avg Ratio |
|---|---:|---:|---:|---:|---:|---:|
| diagnostic_a_exact_recall | 2 | 1 | 1 | 0 | 0.166667 | 1.500000 |

## Diagnostic A - Exact Recall

Streams sparse fact tokens through long filler contexts and checks whether the queried value is recovered exactly.

| Task | Split | Metric | DSRA | MHDSRA2 | Delta | Ratio | Winner | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| aggregate | overall | exact_match_rate.mhdsra2_without_paged_recall | 0.333333 | 0.000000 | -0.333333 | 0.000000 | dsra |  |
| aggregate | overall | exact_match_rate.mhdsra2_with_paged_recall | 0.333333 | 1.000000 | 0.666667 | 3.000000 | mhdsra2 |  |

### Mean metrics by model

| Model | successful_cases | oom_cases | confidence_margin | exact_match_rate |
|---|---|---|---|---|
| Original DSRA | 3.0000 | 0.0000 | 0.0281 | 0.3333 |
| MH-DSRA-v2 (no paged recall) | 3.0000 | 0.0000 | 0.3896 | 0.0000 |
| MH-DSRA-v2 (paged recall) | 3.0000 | 0.0000 | 1.2947 | 1.0000 |
| Sliding window attention | 3.0000 | 0.0000 | 0.0000 | 0.0000 |
| Linear attention | 3.0000 | 0.0000 | 0.0013 | 1.0000 |

### Per-case exact success

| Case | Original DSRA | MH-DSRA-v2 (no paged recall) | MH-DSRA-v2 (paged recall) | Sliding window attention | Linear attention |
|---|---|---|---|---|---|
| exact_recall.early | 0 | 0 | 1 | 0 | 1 |
| exact_recall.middle | 0 | 0 | 1 | 0 | 1 |
| exact_recall.late | 1 | 0 | 1 | 0 | 1 |
