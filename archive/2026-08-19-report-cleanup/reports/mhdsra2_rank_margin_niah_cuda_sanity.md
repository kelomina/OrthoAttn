# mhdsra2_rank_margin_niah_cuda_sanity

- device: `cuda:0`
- groups: `baseline, evidence_rank_margin, evidence_rank_margin_needle_copy`
- tasks: `niah`
- dry_run: `False`

## Groups

- `baseline`: Current default MHDSRA2 configuration. override=`{}`
- `evidence_rank_margin`: Train NIAH evidence retrieval with an extra margin objective so the known evidence candidate must outrank the strongest negative candidate. override=`{}`
- `evidence_rank_margin_needle_copy`: Train the NIAH evidence rank-margin objective, then evaluate with retrieval-candidate needle copy readout. override=`{}`

## Rows

| task | group | seed | status | config | validation | test | elapsed_sec |
|---|---|---:|---|---|---|---|---:|
| niah | baseline | 101 | completed | `{"dim": 32, "epochs": 3, "eval_batches_per_depth": 1, "mhdsra2_config_override": {}, "niah_readout_mode": "model", "num_layers": 1, "read_topk": 4, "retrieval_evidence_loss_alpha": 0.0, "retrieval_evidence_rank_margin": 0.0, "seq_len": 512, "slots": 16, "use_retrieval": false}` | `{"best_eval_mean_accuracy": 0.0, "final_eval_loss": 4.243153651555379, "final_eval_mean_accuracy": 0.0, "final_eval_min_depth_accuracy": 0.0, "final_readout_available_rate": null, "final_target_candidate_hit_rate": null}` | `{}` | 7.41 |
| niah | evidence_rank_margin | 101 | completed | `{"dim": 32, "epochs": 3, "eval_batches_per_depth": 1, "mhdsra2_config_override": {}, "niah_readout_mode": "model", "num_layers": 1, "read_topk": 4, "retrieval_evidence_loss_alpha": 0.25, "retrieval_evidence_rank_margin": 0.15, "seq_len": 512, "slots": 16, "use_retrieval": true}` | `{"best_eval_mean_accuracy": 0.0, "final_eval_loss": 4.268115679423015, "final_eval_mean_accuracy": 0.0, "final_eval_min_depth_accuracy": 0.0, "final_readout_available_rate": null, "final_target_candidate_hit_rate": null}` | `{}` | 6.71 |
| niah | evidence_rank_margin_needle_copy | 101 | completed | `{"dim": 32, "epochs": 3, "eval_batches_per_depth": 1, "mhdsra2_config_override": {}, "niah_readout_mode": "needle_copy", "num_layers": 1, "read_topk": 4, "retrieval_evidence_loss_alpha": 0.25, "retrieval_evidence_rank_margin": 0.15, "seq_len": 512, "slots": 16, "use_retrieval": true}` | `{"best_eval_mean_accuracy": 0.0, "final_eval_loss": 1.0, "final_eval_mean_accuracy": 0.0, "final_eval_min_depth_accuracy": 0.0, "final_readout_available_rate": 1.0, "final_target_candidate_hit_rate": 0.3333333333333333}` | `{}` | 7.51 |

## Diagnostics

| task | group | seed | diagnostics |
|---|---|---:|---|
| niah | baseline | 101 | `{"final_mean_target_candidate_rank": null, "final_readout_available_rate": null, "final_readout_mode": "model", "final_target_candidate_hit_rate": null, "final_train_loss": 6.137888431549072, "peak_memory_allocated_mb": 20.42529296875, "peak_memory_reserved_mb": 26.0, "retrieval_evidence": {"available": false, "best_negative_weight_mean": null, "evidence_margin_mean": null, "evidence_weight_mean": null, "gate_loss": null, "gate_mean": null, "hit_rate": null, "margin_loss": null, "positive_count": 0, "ranking_loss": null, "target_rank_mean": null, "top1_rate": null, "unavailable_reason": null}, "retrieval_evidence_available": false, "retrieval_evidence_best_negative_weight_mean": null, "retrieval_evidence_hit_rate": null, "retrieval_evidence_margin_mean": null, "retrieval_evidence_target_rank_mean": null, "retrieval_evidence_top1_rate": null, "retrieval_evidence_unavailable_reason": null, "retrieval_evidence_weight_mean": null, "slot_collision": {"available": true, "collision_risk": "low", "effective_slot_count": 28.6227531067686, "slot_confidence_mean": 0.9157773852348328, "slot_count": 32, "slot_usage_sum": 1454.3035888671875, "top1_usage_share": 0.05412313714623451}}` |
| niah | evidence_rank_margin | 101 | `{"final_mean_target_candidate_rank": null, "final_readout_available_rate": null, "final_readout_mode": "model", "final_target_candidate_hit_rate": null, "final_train_loss": 6.156713008880615, "peak_memory_allocated_mb": 21.244140625, "peak_memory_reserved_mb": 26.0, "retrieval_evidence": {"available": true, "best_negative_weight_mean": 0.0, "evidence_margin_mean": 0.0, "evidence_weight_mean": 0.0, "gate_loss": 0.3609599173069, "gate_mean": 0.3029930591583252, "hit_rate": 0.0, "margin_loss": 0.0, "positive_count": 0, "ranking_loss": 0.0, "target_rank_mean": null, "top1_rate": null, "unavailable_reason": null}, "retrieval_evidence_available": true, "retrieval_evidence_best_negative_weight_mean": 0.0, "retrieval_evidence_hit_rate": 0.0, "retrieval_evidence_margin_mean": 0.0, "retrieval_evidence_target_rank_mean": null, "retrieval_evidence_top1_rate": null, "retrieval_evidence_unavailable_reason": null, "retrieval_evidence_weight_mean": 0.0, "slot_collision": {"available": true, "collision_risk": "low", "effective_slot_count": 28.61036129559254, "slot_confidence_mean": 0.9212955236434937, "slot_count": 32, "slot_usage_sum": 1454.8572998046875, "top1_usage_share": 0.055680546909570694}}` |
| niah | evidence_rank_margin_needle_copy | 101 | `{"final_mean_target_candidate_rank": 40.0, "final_readout_available_rate": 1.0, "final_readout_mode": "needle_copy", "final_target_candidate_hit_rate": 0.3333333333333333, "final_train_loss": 6.156713008880615, "peak_memory_allocated_mb": 21.244140625, "peak_memory_reserved_mb": 26.0, "retrieval_evidence": {"available": true, "best_negative_weight_mean": 0.0, "evidence_margin_mean": 0.0, "evidence_weight_mean": 0.0, "gate_loss": 0.3609599173069, "gate_mean": 0.3029930591583252, "hit_rate": 0.0, "margin_loss": 0.0, "positive_count": 0, "ranking_loss": 0.0, "target_rank_mean": null, "top1_rate": null, "unavailable_reason": null}, "retrieval_evidence_available": true, "retrieval_evidence_best_negative_weight_mean": 0.0, "retrieval_evidence_hit_rate": 0.0, "retrieval_evidence_margin_mean": 0.0, "retrieval_evidence_target_rank_mean": null, "retrieval_evidence_top1_rate": null, "retrieval_evidence_unavailable_reason": null, "retrieval_evidence_weight_mean": 0.0, "slot_collision": {"available": true, "collision_risk": "low", "effective_slot_count": 28.61036129559254, "slot_confidence_mean": 0.9212955236434937, "slot_count": 32, "slot_usage_sum": 1454.8572998046875, "top1_usage_share": 0.055680546909570694}}` |

## Success Summary

- niah_rows_completed: `3`
- json_rows_completed: `0`
- two_digit_rows_completed: `0`
- two_digit_min_exact_match: `None`
- selection_policy: `Select candidates by validation metrics only; inspect held-out test metrics only after a candidate improves validation.`
