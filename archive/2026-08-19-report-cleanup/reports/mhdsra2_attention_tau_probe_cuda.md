# mhdsra2_attention_tau_probe_cuda

- device: `cuda:0`
- taus: `8, 16, 24`
- seeds: `101, 202, 303`
- retrieval_max_tokens: `128`
- seq_len/chunk_size/epochs: `256 / 64 / 10`
- rows completed: `9/9`
- rows failed: `0`

## Summary

| tau | final_acc_mean | best_acc_mean | target_hit_mean | value_top_mean | span_top1_mean | span_rank_mean | train_span_loss_mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.5000 | 0.6111 | 0.8333 | 0.6111 | 0.3333 | 1.0000 | 1.6072 |
| 16 | 0.5000 | 0.6111 | 0.8333 | 0.6111 | 0.3333 | 1.0000 | 1.6152 |
| 24 | 0.5000 | 0.6111 | 0.8333 | 0.6111 | 0.3333 | 1.0000 | 1.6212 |

## Rows

| tau | seed | final_acc | best_acc | target_hit | value_top | span_top1 | span_rank | elapsed_sec |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 101 | 0.6667 | 0.6667 | 1.0000 | 0.8333 | 1.0000 | 1.0000 | 50.68 |
| 8 | 202 | 0.5000 | 0.5000 | 0.8333 | 0.5000 | 0.0000 | 1.0000 | 30.88 |
| 8 | 303 | 0.3333 | 0.6667 | 0.6667 | 0.5000 | 0.0000 | 1.0000 | 26.60 |
| 16 | 101 | 0.6667 | 0.6667 | 1.0000 | 0.8333 | 1.0000 | 1.0000 | 20.37 |
| 16 | 202 | 0.5000 | 0.5000 | 0.8333 | 0.5000 | 0.0000 | 1.0000 | 36.75 |
| 16 | 303 | 0.3333 | 0.6667 | 0.6667 | 0.5000 | 0.0000 | 1.0000 | 80.68 |
| 24 | 101 | 0.6667 | 0.6667 | 1.0000 | 0.8333 | 1.0000 | 1.0000 | 85.38 |
| 24 | 202 | 0.5000 | 0.5000 | 0.8333 | 0.5000 | 0.0000 | 1.0000 | 83.69 |
| 24 | 303 | 0.3333 | 0.6667 | 0.6667 | 0.5000 | 0.0000 | 1.0000 | 81.92 |

## Notes

- This is a small diagnostic run, not a final benchmark.
- `retrieval_max_tokens` is fixed at 128 to isolate the effect of `retrieval_tau`.
- Test metrics are disabled; results are validation/diagnostic only.
