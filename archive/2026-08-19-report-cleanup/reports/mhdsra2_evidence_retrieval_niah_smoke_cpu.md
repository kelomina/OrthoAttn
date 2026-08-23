# mhdsra2_evidence_retrieval_niah_smoke_cpu

- device: `cpu`
- groups: `evidence_hit_supervision`
- tasks: `niah`
- dry_run: `False`

## Groups

- `evidence_hit_supervision`: Train-only auxiliary evidence-hit supervision for NIAH/JSON. override=`{}`

## Rows

| task | group | seed | status | config | validation | test | elapsed_sec |
|---|---|---:|---|---|---|---|---:|
| niah | evidence_hit_supervision | 101 | completed | `{"dim": 16, "epochs": 1, "eval_batches_per_depth": 1, "mhdsra2_config_override": {}, "num_layers": 1, "read_topk": 2, "retrieval_evidence_loss_alpha": 0.25, "seq_len": 128, "slots": 8}` | `{"best_eval_mean_accuracy": 0.0, "final_eval_loss": 3.7353416283925376, "final_eval_mean_accuracy": 0.0, "final_eval_min_depth_accuracy": 0.0}` | `{}` | 6.30 |

## Diagnostics

| task | group | seed | diagnostics |
|---|---|---:|---|
| niah | evidence_hit_supervision | 101 | `{"final_train_loss": 5.470801830291748, "peak_memory_allocated_mb": 0.0, "peak_memory_reserved_mb": 0.0, "retrieval_evidence": {"available": false, "evidence_weight_mean": 0.0, "gate_loss": 0.0, "gate_mean": 0.0, "hit_rate": 0.0, "positive_count": 0, "ranking_loss": 0.0, "unavailable_reason": "missing_selected_metadata_or_gate"}, "retrieval_evidence_available": false, "retrieval_evidence_hit_rate": 0.0, "retrieval_evidence_unavailable_reason": "missing_selected_metadata_or_gate", "retrieval_evidence_weight_mean": 0.0, "slot_collision": {"available": true, "collision_risk": "low", "effective_slot_count": 7.534153104082085, "slot_confidence_mean": 0.8218017816543579, "slot_count": 8, "slot_usage_sum": 191.13348388671875, "top1_usage_share": 0.17230214178562164}}` |

## Success Summary

- niah_rows_completed: `1`
- json_rows_completed: `0`
- two_digit_rows_completed: `0`
- two_digit_min_exact_match: `None`
- selection_policy: `Select candidates by validation metrics only; inspect held-out test metrics only after a candidate improves validation.`
