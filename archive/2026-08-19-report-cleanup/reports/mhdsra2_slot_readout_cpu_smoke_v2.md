# mhdsra2_slot_readout_cpu_smoke_v2

- device: `cpu`
- groups: `slot_readout_bias`
- tasks: `json`
- dry_run: `False`

## Groups

- `slot_readout_bias`: Train the existing JSON slot decoder and apply its generation-time byte-level readout bias. override=`{}`

## Rows

| task | group | seed | status | config | validation | test | elapsed_sec |
|---|---|---:|---|---|---|---|---:|
| json | slot_readout_bias | 7 | completed | `{"chunk_size": 64, "dim": 16, "epochs": 1, "evidence_hint_weight": 0.0, "evidence_loss_weight": 0.0, "generalization_score_mode": "generation", "mhdsra2_config_override": {}, "read_topk": 2, "seed_bundle": {"model_seed": 411, "pair_split_seed": 310, "seed_root": 7, "test_dataset_seed": 209, "train_dataset_seed": 7, "validation_dataset_seed": 108}, "slot_decoder_logit_bias": 4.0, "slot_decoder_loss_weight": 0.35, "slots": 8, "test_dataset_size": 1, "train_dataset_size": 1, "validation_dataset_size": 1}` | `{"validation_generation_exact_match_rate": 0.0, "validation_generation_mean_sequence_accuracy": 0.023076923076923078}` | `{"test_generation_exact_match_rate": 0.0, "test_generation_mean_sequence_accuracy": 0.05}` | 38.29 |

## Diagnostics

| task | group | seed | diagnostics |
|---|---|---:|---|
| json | slot_readout_bias | 7 | `{"test_evidence_window_accuracy": null, "test_evidence_window_mean_distance": null, "test_slot_decoder_artifact_accuracy": 0.0, "test_slot_decoder_artist_accuracy": 1.0, "test_slot_decoder_dynasty_accuracy": 0.0, "test_slot_decoder_full_answer_accuracy": 0.0, "test_slot_decoder_museum_accuracy": 0.0, "test_teacher_forced_exact_match_rate": 0.0, "test_teacher_forced_mean_sequence_accuracy": 0.041666666666666664, "validation_evidence_window_accuracy": null, "validation_evidence_window_mean_distance": null, "validation_slot_decoder_artifact_accuracy": 0.0, "validation_slot_decoder_artist_accuracy": 0.0, "validation_slot_decoder_dynasty_accuracy": 0.0, "validation_slot_decoder_full_answer_accuracy": 0.0, "validation_slot_decoder_museum_accuracy": 0.0, "validation_teacher_forced_exact_match_rate": 0.0, "validation_teacher_forced_mean_sequence_accuracy": 0.023076923076923078}` |

## Success Summary

- niah_rows_completed: `0`
- json_rows_completed: `1`
- two_digit_rows_completed: `0`
- two_digit_min_exact_match: `None`
- selection_policy: `Select candidates by validation metrics only; inspect held-out test metrics only after a candidate improves validation.`
