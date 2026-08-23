# MHDSRA2 vs DSRA Next-Round Benchmark

## Config
- seed: `7`

## Summary
- total rows: `2`
- valid rows: `2`
- MHDSRA2 wins: `1`
- DSRA wins: `1`
- ties: `0`
- average delta (MHDSRA2-DSRA): `0.000000`
- average ratio (MHDSRA2/DSRA): `0.933333`

## Suite Breakdown

| Suite | Rows | MHDSRA2 Wins | DSRA Wins | Ties | Avg Delta | Avg Ratio |
|---|---:|---:|---:|---:|---:|---:|
| json_retrieval_generalization | 1 | 0 | 1 | 0 | -0.050000 | 0.800000 |
| needle_in_haystack | 1 | 1 | 0 | 0 | 0.050000 | 1.066667 |

## Synthetic

| Task | Split | Metric | DSRA | MHDSRA2 | Delta | Ratio | Winner | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| seq_len=8192 | overall | final_eval_mean_accuracy | 0.750000 | 0.800000 | 0.050000 | 1.066667 | mhdsra2 |  |
| museum_artifact_generalization | test | generation_exact_match_rate | 0.250000 | 0.200000 | -0.050000 | 0.800000 | dsra |  |

### Five-model diagnostic summary

| Model | exact_match_rate |
|---|---|
| Archived DSRA alias / MHDSRA2 | 0.2500 |
| MH-DSRA-v2 (paged recall) | 1.0000 |
