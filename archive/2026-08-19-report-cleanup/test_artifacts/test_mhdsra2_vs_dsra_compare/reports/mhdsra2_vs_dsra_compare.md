# MHDSRA2 vs DSRA Comparison

## Config
- device: `cpu`
- batch_size: `[1]`
- dim: `[32]`
- warmup_runs: `1`
- repeat_runs: `2`
- use_retrieval: `False`
- retrieval_tokens: `0`
- slots grid: `[8]`
- read_topk grid: `[4]`
- chunk_size grid: `[8]`

## Automatic Summary
- total cases: `2`
- MHDSRA2 faster cases: `2` / `2`
- MHDSRA2 faster ratio: `1.000`
- average speedup: `1.211`
- median speedup: `1.211`
- average state-bytes ratio (MHDSRA2/DSRA): `0.707`
- best speedup case: `batch=1, dim=32, seq=16, slots=8, topk=4, chunk=8` => `1.281x`
- weakest speedup case: `batch=1, dim=32, seq=8, slots=8, topk=4, chunk=8` => `1.140x`
- MHDSRA2 min state-overhead case: `batch=1, dim=32, seq=8, slots=8, topk=4, chunk=8` => state ratio `0.707`
- MHDSRA2 max state-overhead case: `batch=1, dim=32, seq=8, slots=8, topk=4, chunk=8` => state ratio `0.707`

## Grouped Conclusions

### By Batch Size

| Value | Cases | Avg DSRA ms | Avg DSRA std ms | Avg MHDSRA2 ms | Avg MHDSRA2 std ms | Avg Speedup | Avg State Ratio (MHDSRA2/DSRA) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 7.036 | 0.346 | 5.680 | 0.424 | 1.211 | 0.707 |

### By Dim

| Value | Cases | Avg DSRA ms | Avg DSRA std ms | Avg MHDSRA2 ms | Avg MHDSRA2 std ms | Avg Speedup | Avg State Ratio (MHDSRA2/DSRA) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 2 | 7.036 | 0.346 | 5.680 | 0.424 | 1.211 | 0.707 |

### By Seq Len

| Value | Cases | Avg DSRA ms | Avg DSRA std ms | Avg MHDSRA2 ms | Avg MHDSRA2 std ms | Avg Speedup | Avg State Ratio (MHDSRA2/DSRA) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 3.899 | 0.191 | 3.421 | 0.045 | 1.140 | 0.707 |
| 16 | 1 | 10.173 | 0.501 | 7.939 | 0.803 | 1.281 | 0.707 |

### By Slots

| Value | Cases | Avg DSRA ms | Avg DSRA std ms | Avg MHDSRA2 ms | Avg MHDSRA2 std ms | Avg Speedup | Avg State Ratio (MHDSRA2/DSRA) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 2 | 7.036 | 0.346 | 5.680 | 0.424 | 1.211 | 0.707 |

### By Read TopK

| Value | Cases | Avg DSRA ms | Avg DSRA std ms | Avg MHDSRA2 ms | Avg MHDSRA2 std ms | Avg Speedup | Avg State Ratio (MHDSRA2/DSRA) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2 | 7.036 | 0.346 | 5.680 | 0.424 | 1.211 | 0.707 |

### By Chunk Size

| Value | Cases | Avg DSRA ms | Avg DSRA std ms | Avg MHDSRA2 ms | Avg MHDSRA2 std ms | Avg Speedup | Avg State Ratio (MHDSRA2/DSRA) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 2 | 7.036 | 0.346 | 5.680 | 0.424 | 1.211 | 0.707 |

## Top Speedup Cases

| Rank | Case | Speedup | State Ratio (MHDSRA2/DSRA) |
|---:|---|---:|---:|
| 1 | batch=1, dim=32, seq=16, slots=8, topk=4, chunk=8 | 1.281 | 0.707 |
| 2 | batch=1, dim=32, seq=8, slots=8, topk=4, chunk=8 | 1.140 | 0.707 |

## Raw Cases

| Batch | Dim | Seq Len | Slots | Read TopK | Chunk | DSRA ms | DSRA std ms | MHDSRA2 ms | MHDSRA2 std ms | Speedup (DSRA/MHDSRA2) | State Ratio (MHDSRA2/DSRA) | DSRA State Bytes | MHDSRA2 State Bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 8 | 8 | 4 | 8 | 3.899 | 0.191 | 3.421 | 0.045 | 1.140 | 0.707 | 6336 | 4480 |
| 1 | 32 | 16 | 8 | 4 | 8 | 10.173 | 0.501 | 7.939 | 0.803 | 1.281 | 0.707 | 6336 | 4480 |
