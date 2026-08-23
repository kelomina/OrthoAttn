# mhdsra2_evidence_retrieval_niah_cuda_sanity

- device: `cuda:0`
- groups: `evidence_hit_supervision`
- tasks: `niah`
- dry_run: `False`

## Groups

- `evidence_hit_supervision`: Train-only auxiliary evidence-hit supervision for NIAH/JSON. override=`{}`

## Rows

| task | group | seed | status | config | validation | test | elapsed_sec |
|---|---|---:|---|---|---|---|---:|
| niah | evidence_hit_supervision | 101 | completed | `{"dim": 16, "epochs": 1, "eval_batches_per_depth": 1, "mhdsra2_config_override": {}, "num_layers": 1, "read_topk": 4, "retrieval_evidence_loss_alpha": 0.25, "seq_len": 512, "slots": 8, "use_retrieval": true}` | `{"best_eval_mean_accuracy": 0.0, "final_eval_loss": 4.253831624984741, "final_eval_mean_accuracy": 0.0, "final_eval_min_depth_accuracy": 0.0}` | `{}` | 8.06 |

## Diagnostics

| task | group | seed | diagnostics |
|---|---|---:|---|
| niah | evidence_hit_supervision | 101 | `{"final_train_loss": 9.111913681030273, "peak_memory_allocated_mb": 18.3388671875, "peak_memory_reserved_mb": 24.0, "retrieval_evidence": {"available": true, "evidence_weight_mean": 0.0002082317223539576, "gate_loss": 1.2356699705123901, "gate_mean": 0.2906399667263031, "hit_rate": 1.0, "positive_count": 1, "ranking_loss": 8.476859092712402, "unavailable_reason": null}, "retrieval_evidence_available": true, "retrieval_evidence_hit_rate": 1.0, "retrieval_evidence_unavailable_reason": null, "retrieval_evidence_weight_mean": 0.0002082317223539576, "slot_collision": {"available": true, "collision_risk": "low", "effective_slot_count": 7.945093984203545, "slot_confidence_mean": 0.9283581972122192, "slot_count": 8, "slot_usage_sum": 736.2921752929688, "top1_usage_share": 0.14300282299518585}}` |

## Success Summary

- niah_rows_completed: `1`
- json_rows_completed: `0`
- two_digit_rows_completed: `0`
- two_digit_min_exact_match: `None`
- selection_policy: `Select candidates by validation metrics only; inspect held-out test metrics only after a candidate improves validation.`
