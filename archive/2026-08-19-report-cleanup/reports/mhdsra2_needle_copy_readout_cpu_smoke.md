# mhdsra2_needle_copy_readout_cpu_smoke

- device: `cpu`
- groups: `baseline, needle_copy_readout`
- tasks: `niah`
- dry_run: `False`

## Groups

- `baseline`: Current default MHDSRA2 configuration. override=`{}`
- `needle_copy_readout`: Evaluate NIAH with an explicit retrieval-candidate needle copy readout. override=`{}`

## Rows

| task | group | seed | status | config | validation | test | elapsed_sec |
|---|---|---:|---|---|---|---|---:|
| niah | baseline | 101 | completed | `{"dim": 16, "epochs": 2, "eval_batches_per_depth": 1, "mhdsra2_config_override": {}, "niah_readout_mode": "model", "num_layers": 1, "read_topk": 4, "retrieval_evidence_loss_alpha": 0.0, "seq_len": 256, "slots": 8, "use_retrieval": false}` | `{"best_eval_mean_accuracy": 0.0, "final_eval_loss": 3.1126839319864907, "final_eval_mean_accuracy": 0.0, "final_eval_min_depth_accuracy": 0.0, "final_readout_available_rate": null}` | `{}` | 6.21 |
| niah | needle_copy_readout | 101 | completed | `{"dim": 16, "epochs": 2, "eval_batches_per_depth": 1, "mhdsra2_config_override": {}, "niah_readout_mode": "needle_copy", "num_layers": 1, "read_topk": 4, "retrieval_evidence_loss_alpha": 0.0, "seq_len": 256, "slots": 8, "use_retrieval": true}` | `{"best_eval_mean_accuracy": 0.0, "final_eval_loss": 1.0, "final_eval_mean_accuracy": 0.0, "final_eval_min_depth_accuracy": 0.0, "final_readout_available_rate": 1.0}` | `{}` | 5.59 |

## Diagnostics

| task | group | seed | diagnostics |
|---|---|---:|---|
| niah | baseline | 101 | `{"final_readout_available_rate": null, "final_readout_mode": "model", "final_train_loss": 4.583413124084473, "peak_memory_allocated_mb": 0.0, "peak_memory_reserved_mb": 0.0, "retrieval_evidence": {"available": false, "evidence_weight_mean": null, "gate_loss": null, "gate_mean": null, "hit_rate": null, "positive_count": 0, "ranking_loss": null, "unavailable_reason": null}, "retrieval_evidence_available": false, "retrieval_evidence_hit_rate": null, "retrieval_evidence_unavailable_reason": null, "retrieval_evidence_weight_mean": null, "slot_collision": {"available": true, "collision_risk": "low", "effective_slot_count": 7.91750553414808, "slot_confidence_mean": 0.8460391163825989, "slot_count": 8, "slot_usage_sum": 346.65203857421875, "top1_usage_share": 0.1395936757326126}}` |
| niah | needle_copy_readout | 101 | `{"final_readout_available_rate": 1.0, "final_readout_mode": "needle_copy", "final_train_loss": 4.580101013183594, "peak_memory_allocated_mb": 0.0, "peak_memory_reserved_mb": 0.0, "retrieval_evidence": {"available": false, "evidence_weight_mean": null, "gate_loss": null, "gate_mean": null, "hit_rate": null, "positive_count": 0, "ranking_loss": null, "unavailable_reason": null}, "retrieval_evidence_available": false, "retrieval_evidence_hit_rate": null, "retrieval_evidence_unavailable_reason": null, "retrieval_evidence_weight_mean": null, "slot_collision": {"available": true, "collision_risk": "low", "effective_slot_count": 7.9383835682260075, "slot_confidence_mean": 0.8442502021789551, "slot_count": 8, "slot_usage_sum": 347.0931701660156, "top1_usage_share": 0.13863791525363922}}` |

## Success Summary

- niah_rows_completed: `2`
- json_rows_completed: `0`
- two_digit_rows_completed: `0`
- two_digit_min_exact_match: `None`
- selection_policy: `Select candidates by validation metrics only; inspect held-out test metrics only after a candidate improves validation.`
