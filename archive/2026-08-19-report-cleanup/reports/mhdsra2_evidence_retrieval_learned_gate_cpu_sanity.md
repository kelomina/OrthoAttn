# mhdsra2_evidence_retrieval_learned_gate_cpu_sanity

- device: `cpu`
- groups: `learned_retrieval_gate`
- tasks: `niah`
- dry_run: `False`

## Groups

- `learned_retrieval_gate`: Enable a zero-initialized learned retrieval gate adapter. override=`{"retrieval_quality_gate_adapter": true}`

## Rows

| task | group | seed | status | config | validation | test | elapsed_sec |
|---|---|---:|---|---|---|---|---:|
| niah | learned_retrieval_gate | 101 | completed | `{"dim": 16, "epochs": 1, "eval_batches_per_depth": 1, "mhdsra2_config_override": {"retrieval_quality_gate_adapter": true}, "num_layers": 1, "read_topk": 4, "retrieval_evidence_loss_alpha": 0.0, "seq_len": 256, "slots": 8, "use_retrieval": true}` | `{"best_eval_mean_accuracy": 0.0, "final_eval_loss": 3.228862682978312, "final_eval_mean_accuracy": 0.0, "final_eval_min_depth_accuracy": 0.0}` | `{}` | 3.86 |

## Diagnostics

| task | group | seed | diagnostics |
|---|---|---:|---|
| niah | learned_retrieval_gate | 101 | `{"final_train_loss": 5.246702194213867, "peak_memory_allocated_mb": 0.0, "peak_memory_reserved_mb": 0.0, "retrieval_evidence": {"available": false, "evidence_weight_mean": null, "gate_loss": null, "gate_mean": null, "hit_rate": null, "positive_count": 0, "ranking_loss": null, "unavailable_reason": null}, "retrieval_evidence_available": false, "retrieval_evidence_hit_rate": null, "retrieval_evidence_unavailable_reason": null, "retrieval_evidence_weight_mean": null, "slot_collision": {"available": true, "collision_risk": "low", "effective_slot_count": 7.847024055493933, "slot_confidence_mean": 0.8534027338027954, "slot_count": 8, "slot_usage_sum": 368.6432189941406, "top1_usage_share": 0.16608251631259918}}` |

## Success Summary

- niah_rows_completed: `1`
- json_rows_completed: `0`
- two_digit_rows_completed: `0`
- two_digit_min_exact_match: `None`
- selection_policy: `Select candidates by validation metrics only; inspect held-out test metrics only after a candidate improves validation.`
