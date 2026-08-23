# mhdsra2_score_margin_retrieval_smoke_cuda

- device: `cuda:0`
- groups: `baseline`
- tasks: `smoke`
- dry_run: `False`

## Groups

- `baseline`: Current default MHDSRA2 configuration. override=`{}`

## Rows

| task | group | seed | status | config | validation | test | elapsed_sec |
|---|---|---:|---|---|---|---|---:|
| smoke | shared | 20260602 | passed | `{"batch_sizes": [1, 2], "max_tokens": 4, "page_size": 8, "tokens": [32], "top_pages": 2}` | `{"all_batch_loop_positions_match": true, "no_cross_sample_leak": true, "no_future_leak": true, "passed": true}` | `{}` | 1.51 |

## Diagnostics

| task | group | seed | diagnostics |
|---|---|---:|---|
| smoke | shared | 20260602 | `{}` |

## Success Summary

- niah_rows_completed: `0`
- json_rows_completed: `0`
- two_digit_rows_completed: `0`
- two_digit_min_exact_match: `None`
- selection_policy: `Select candidates by validation metrics only; inspect held-out test metrics only after a candidate improves validation.`
