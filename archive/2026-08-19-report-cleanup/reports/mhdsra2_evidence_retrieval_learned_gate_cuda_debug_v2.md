# mhdsra2_evidence_retrieval_learned_gate_cuda_debug_v2

- device: `cuda:0`
- groups: `learned_retrieval_gate`
- tasks: `niah`
- dry_run: `False`

## Groups

- `learned_retrieval_gate`: Enable a zero-initialized learned retrieval gate adapter. override=`{"retrieval_quality_gate_adapter": true}`

## Rows

| task | group | seed | status | config | validation | test | elapsed_sec |
|---|---|---:|---|---|---|---|---:|
| niah | learned_retrieval_gate | 101 | completed | `{"dim": 64, "epochs": 5, "eval_batches_per_depth": 1, "mhdsra2_config_override": {"retrieval_quality_gate_adapter": true}, "num_layers": 2, "read_topk": 8, "retrieval_evidence_loss_alpha": 0.0, "seq_len": 8192, "slots": 64, "use_retrieval": true}` | `{"best_eval_mean_accuracy": 0.0, "final_eval_loss": 4.945390860239665, "final_eval_mean_accuracy": 0.0, "final_eval_min_depth_accuracy": 0.0}` | `{}` | 25.93 |

## Diagnostics

| task | group | seed | diagnostics |
|---|---|---:|---|
| niah | learned_retrieval_gate | 101 | `{"final_train_loss": 6.992099285125732, "peak_memory_allocated_mb": 405.97314453125, "peak_memory_reserved_mb": 462.0, "retrieval_evidence": {"available": false, "evidence_weight_mean": null, "gate_loss": null, "gate_mean": null, "hit_rate": null, "positive_count": 0, "ranking_loss": null, "unavailable_reason": null}, "retrieval_evidence_available": false, "retrieval_evidence_hit_rate": null, "retrieval_evidence_unavailable_reason": null, "retrieval_evidence_weight_mean": null, "slot_collision": {"available": true, "collision_risk": "low", "effective_slot_count": 241.57475859213997, "slot_confidence_mean": 0.9811035990715027, "slot_count": 256, "slot_usage_sum": 50300.375, "top1_usage_share": 0.006856009364128113}}` |

## Success Summary

- niah_rows_completed: `1`
- json_rows_completed: `0`
- two_digit_rows_completed: `0`
- two_digit_min_exact_match: `None`
- selection_policy: `Select candidates by validation metrics only; inspect held-out test metrics only after a candidate improves validation.`
