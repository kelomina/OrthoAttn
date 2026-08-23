# mhdsra2_slot_readout_smoke_cpu

- device: `cpu`
- groups: `baseline, slot_readout_bias, evidence_slot_readout`
- tasks: `json`
- dry_run: `False`

## Groups

- `baseline`: Current default MHDSRA2 configuration. override=`{}`
- `slot_readout_bias`: Train the existing JSON slot decoder and apply its generation-time byte-level readout bias. override=`{}`
- `evidence_slot_readout`: Train evidence-window supervision plus the existing JSON slot decoder, then use the slot readout bias during generation. override=`{}`

## Rows

| task | group | seed | status | config | validation | test | elapsed_sec |
|---|---|---:|---|---|---|---|---:|
| json | baseline | 7 | completed | `{"chunk_size": 64, "dim": 32, "epochs": 3, "evidence_hint_weight": 0.0, "evidence_loss_weight": 0.0, "generalization_score_mode": "generation", "mhdsra2_config_override": {}, "read_topk": 4, "seed_bundle": {"model_seed": 411, "pair_split_seed": 310, "seed_root": 7, "test_dataset_seed": 209, "train_dataset_seed": 7, "validation_dataset_seed": 108}, "slot_decoder_logit_bias": 0.0, "slot_decoder_loss_weight": 0.0, "slots": 16, "test_dataset_size": 1, "train_dataset_size": 2, "validation_dataset_size": 1}` | `{"validation_generation_exact_match_rate": 0.0, "validation_generation_mean_sequence_accuracy": 0.0}` | `{"test_generation_exact_match_rate": 0.0, "test_generation_mean_sequence_accuracy": 0.0}` | 48.46 |
| json | slot_readout_bias | 7 | completed | `{"chunk_size": 64, "dim": 32, "epochs": 3, "evidence_hint_weight": 0.0, "evidence_loss_weight": 0.0, "generalization_score_mode": "generation", "mhdsra2_config_override": {}, "read_topk": 4, "seed_bundle": {"model_seed": 411, "pair_split_seed": 310, "seed_root": 7, "test_dataset_seed": 209, "train_dataset_seed": 7, "validation_dataset_seed": 108}, "slot_decoder_logit_bias": 4.0, "slot_decoder_loss_weight": 0.35, "slots": 16, "test_dataset_size": 1, "train_dataset_size": 2, "validation_dataset_size": 1}` | `{"validation_generation_exact_match_rate": 0.0, "validation_generation_mean_sequence_accuracy": 0.06153846153846154}` | `{"test_generation_exact_match_rate": 0.0, "test_generation_mean_sequence_accuracy": 0.008333333333333333}` | 39.21 |
| json | evidence_slot_readout | 7 | completed | `{"chunk_size": 64, "dim": 32, "epochs": 3, "evidence_hint_weight": 0.0, "evidence_loss_weight": 0.2, "generalization_score_mode": "generation", "mhdsra2_config_override": {}, "read_topk": 4, "seed_bundle": {"model_seed": 411, "pair_split_seed": 310, "seed_root": 7, "test_dataset_seed": 209, "train_dataset_seed": 7, "validation_dataset_seed": 108}, "slot_decoder_logit_bias": 4.0, "slot_decoder_loss_weight": 0.35, "slots": 16, "test_dataset_size": 1, "train_dataset_size": 2, "validation_dataset_size": 1}` | `{"validation_generation_exact_match_rate": 0.0, "validation_generation_mean_sequence_accuracy": 0.08461538461538462}` | `{"test_generation_exact_match_rate": 0.0, "test_generation_mean_sequence_accuracy": 0.008333333333333333}` | 43.69 |

## Diagnostics

| task | group | seed | diagnostics |
|---|---|---:|---|
| json | baseline | 7 | `{"test_evidence_window_accuracy": null, "test_evidence_window_mean_distance": null, "test_slot_decoder_artifact_accuracy": null, "test_slot_decoder_artist_accuracy": null, "test_slot_decoder_dynasty_accuracy": null, "test_slot_decoder_full_answer_accuracy": null, "test_slot_decoder_museum_accuracy": null, "test_teacher_forced_exact_match_rate": 0.0, "test_teacher_forced_mean_sequence_accuracy": 0.03333333333333333, "validation_evidence_window_accuracy": null, "validation_evidence_window_mean_distance": null, "validation_slot_decoder_artifact_accuracy": null, "validation_slot_decoder_artist_accuracy": null, "validation_slot_decoder_dynasty_accuracy": null, "validation_slot_decoder_full_answer_accuracy": null, "validation_slot_decoder_museum_accuracy": null, "validation_teacher_forced_exact_match_rate": 0.0, "validation_teacher_forced_mean_sequence_accuracy": 0.03076923076923077}` |
| json | slot_readout_bias | 7 | `{"test_evidence_window_accuracy": null, "test_evidence_window_mean_distance": null, "test_slot_decoder_artifact_accuracy": 0.0, "test_slot_decoder_artist_accuracy": 0.0, "test_slot_decoder_dynasty_accuracy": 0.0, "test_slot_decoder_full_answer_accuracy": 0.0, "test_slot_decoder_museum_accuracy": 0.0, "test_teacher_forced_exact_match_rate": 0.0, "test_teacher_forced_mean_sequence_accuracy": 0.03333333333333333, "validation_evidence_window_accuracy": null, "validation_evidence_window_mean_distance": null, "validation_slot_decoder_artifact_accuracy": 0.0, "validation_slot_decoder_artist_accuracy": 1.0, "validation_slot_decoder_dynasty_accuracy": 0.0, "validation_slot_decoder_full_answer_accuracy": 0.0, "validation_slot_decoder_museum_accuracy": 0.0, "validation_teacher_forced_exact_match_rate": 0.0, "validation_teacher_forced_mean_sequence_accuracy": 0.038461538461538464}` |
| json | evidence_slot_readout | 7 | `{"test_evidence_window_accuracy": 0.0, "test_evidence_window_mean_distance": 5.0, "test_slot_decoder_artifact_accuracy": 0.0, "test_slot_decoder_artist_accuracy": 0.0, "test_slot_decoder_dynasty_accuracy": 0.0, "test_slot_decoder_full_answer_accuracy": 0.0, "test_slot_decoder_museum_accuracy": 0.0, "test_teacher_forced_exact_match_rate": 0.0, "test_teacher_forced_mean_sequence_accuracy": 0.03333333333333333, "validation_evidence_window_accuracy": 0.0, "validation_evidence_window_mean_distance": 6.0, "validation_slot_decoder_artifact_accuracy": 1.0, "validation_slot_decoder_artist_accuracy": 0.0, "validation_slot_decoder_dynasty_accuracy": 0.0, "validation_slot_decoder_full_answer_accuracy": 0.0, "validation_slot_decoder_museum_accuracy": 0.0, "validation_teacher_forced_exact_match_rate": 0.0, "validation_teacher_forced_mean_sequence_accuracy": 0.046153846153846156}` |

## Success Summary

- niah_rows_completed: `0`
- json_rows_completed: `3`
- two_digit_rows_completed: `0`
- two_digit_min_exact_match: `None`
- selection_policy: `Select candidates by validation metrics only; inspect held-out test metrics only after a candidate improves validation.`
