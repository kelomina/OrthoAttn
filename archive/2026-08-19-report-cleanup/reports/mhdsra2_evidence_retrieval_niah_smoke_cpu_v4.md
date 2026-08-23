# mhdsra2_evidence_retrieval_niah_smoke_cpu_v4

- device: `cpu`
- groups: `evidence_hit_supervision`
- tasks: `niah`
- dry_run: `False`

## Groups

- `evidence_hit_supervision`: Train-only auxiliary evidence-hit supervision for NIAH/JSON. override=`{}`

## Rows

| task | group | seed | status | config | validation | test | elapsed_sec |
|---|---|---:|---|---|---|---|---:|
| niah | evidence_hit_supervision | 101 | completed | `{"dim": 16, "epochs": 2, "eval_batches_per_depth": 1, "mhdsra2_config_override": {}, "num_layers": 1, "read_topk": 4, "retrieval_evidence_loss_alpha": 0.25, "seq_len": 256, "slots": 8, "use_retrieval": true}` | `{"best_eval_mean_accuracy": 0.3333333333333333, "final_eval_loss": 4.2197028795878095, "final_eval_mean_accuracy": 0.0, "final_eval_min_depth_accuracy": 0.0}` | `{}` | 4.69 |

## Diagnostics

| task | group | seed | diagnostics |
|---|---|---:|---|
| niah | evidence_hit_supervision | 101 | `{"final_train_loss": 6.068317890167236, "peak_memory_allocated_mb": 0.0, "peak_memory_reserved_mb": 0.0, "retrieval_evidence": {"available": true, "evidence_weight_mean": 0.008872120641171932, "gate_loss": 1.228470802307129, "gate_mean": 0.2927398979663849, "hit_rate": 1.0, "positive_count": 1, "ranking_loss": 4.724841594696045, "unavailable_reason": null}, "retrieval_evidence_available": true, "retrieval_evidence_hit_rate": 1.0, "retrieval_evidence_unavailable_reason": null, "retrieval_evidence_weight_mean": 0.008872120641171932, "slot_collision": {"available": true, "collision_risk": "low", "effective_slot_count": 7.63976579640243, "slot_confidence_mean": 0.8724732398986816, "slot_count": 8, "slot_usage_sum": 346.3200988769531, "top1_usage_share": 0.16514834761619568}}` |

## Success Summary

- niah_rows_completed: `1`
- json_rows_completed: `0`
- two_digit_rows_completed: `0`
- two_digit_min_exact_match: `None`
- selection_policy: `Select candidates by validation metrics only; inspect held-out test metrics only after a candidate improves validation.`
