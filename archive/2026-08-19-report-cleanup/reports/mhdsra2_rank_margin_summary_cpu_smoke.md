# mhdsra2_rank_margin_summary_cpu_smoke

- device: `cpu`
- groups: `evidence_rank_margin_needle_copy`
- tasks: `niah`
- dry_run: `False`

## Groups

- `evidence_rank_margin_needle_copy`: Train the NIAH evidence rank-margin objective, then evaluate with retrieval-candidate needle copy readout. override=`{}`

## Rows

| task | group | seed | status | config | validation | test | elapsed_sec |
|---|---|---:|---|---|---|---|---:|
| niah | evidence_rank_margin_needle_copy | 101 | completed | `{"dim": 16, "epochs": 2, "eval_batches_per_depth": 1, "mhdsra2_config_override": {}, "niah_readout_mode": "needle_copy", "num_layers": 1, "read_topk": 4, "retrieval_evidence_loss_alpha": 0.25, "retrieval_evidence_rank_margin": 0.15, "seq_len": 128, "slots": 8, "use_retrieval": true}` | `{"best_eval_mean_accuracy": 0.0, "final_eval_loss": 0.6666666666666666, "final_eval_mean_accuracy": 0.3333333333333333, "final_eval_min_depth_accuracy": 0.0, "final_readout_available_rate": 1.0, "final_target_candidate_hit_rate": 1.0}` | `{}` | 5.06 |

## Diagnostics

| task | group | seed | diagnostics |
|---|---|---:|---|
| niah | evidence_rank_margin_needle_copy | 101 | `{"final_mean_target_candidate_rank": 2.0, "final_readout_available_rate": 1.0, "final_readout_mode": "needle_copy", "final_target_candidate_hit_rate": 1.0, "final_train_loss": 6.837496757507324, "peak_memory_allocated_mb": 0.0, "peak_memory_reserved_mb": 0.0, "retrieval_evidence": {"available": true, "best_negative_weight_mean": 0.11499479413032532, "evidence_margin_mean": -0.11494404077529907, "evidence_weight_mean": 5.075264198239893e-05, "gate_loss": 1.0958079099655151, "gate_mean": 0.3342694342136383, "hit_rate": 1.0, "margin_loss": 0.26494404673576355, "positive_count": 1, "ranking_loss": 9.88854694366455, "target_rank_mean": 61.0, "top1_rate": 0.0, "unavailable_reason": null}, "retrieval_evidence_available": true, "retrieval_evidence_best_negative_weight_mean": 0.11499479413032532, "retrieval_evidence_hit_rate": 1.0, "retrieval_evidence_margin_mean": -0.11494404077529907, "retrieval_evidence_target_rank_mean": 61.0, "retrieval_evidence_top1_rate": 0.0, "retrieval_evidence_unavailable_reason": null, "retrieval_evidence_weight_mean": 5.075264198239893e-05, "slot_collision": {"available": true, "collision_risk": "low", "effective_slot_count": 7.614129231126715, "slot_confidence_mean": 0.838097095489502, "slot_count": 8, "slot_usage_sum": 194.55535888671875, "top1_usage_share": 0.1793477088212967}, "train_retrieval_evidence_summary": {"available_steps": 2, "mean_best_negative_weight": 0.11831267923116684, "mean_evidence_margin": -0.11826984211802483, "mean_evidence_weight": 4.2835119529627264e-05, "mean_gate_loss": 1.097787857055664, "mean_hit_rate": 1.0, "mean_margin_loss": 0.2682698518037796, "mean_ranking_loss": 10.075533390045166, "mean_target_rank": 61.5, "mean_top1_rate": 0.0, "positive_steps": 2, "steps": 2}}` |

## Success Summary

- niah_rows_completed: `1`
- json_rows_completed: `0`
- two_digit_rows_completed: `0`
- two_digit_min_exact_match: `None`
- selection_policy: `Select candidates by validation metrics only; inspect held-out test metrics only after a candidate improves validation.`
