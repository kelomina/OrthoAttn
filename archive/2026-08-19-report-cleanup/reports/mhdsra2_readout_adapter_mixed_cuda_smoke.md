# mhdsra2_readout_adapter_mixed_cuda_smoke

- device: `cuda:0`
- groups: `baseline, extract_compose_readout`
- tasks: `json`
- dry_run: `False`

## Groups

- `baseline`: Current default MHDSRA2 configuration. override=`{}`
- `extract_compose_readout`: Use the existing evidence-window decoder plus deterministic extract-then-compose answer readout for JSON generation metrics. override=`{}`

## Rows

| task | group | seed | status | config | validation | test | elapsed_sec |
|---|---|---:|---|---|---|---|---:|
| json | baseline | 7 | completed | `{"answer_template_mode": "mixed", "chunk_size": 64, "dim": 16, "distractor_records_per_case": 2, "epochs": 1, "evidence_hint_weight": 0.0, "evidence_loss_weight": 0.0, "generalization_score_mode": "generation", "generation_readout_mode": "model", "mhdsra2_config_override": {}, "read_topk": 2, "seed_bundle": {"model_seed": 411, "pair_split_seed": 310, "seed_root": 7, "test_dataset_seed": 209, "train_dataset_seed": 7, "validation_dataset_seed": 108}, "slot_decoder_logit_bias": 0.0, "slot_decoder_loss_weight": 0.0, "slots": 8, "test_dataset_size": 1, "train_dataset_size": 1, "validation_dataset_size": 1}` | `{"validation_generation_exact_match_rate": 0.0, "validation_generation_mean_sequence_accuracy": 0.007575757575757576}` | `{"test_generation_exact_match_rate": 0.0, "test_generation_mean_sequence_accuracy": 0.0}` | 24.25 |
| json | extract_compose_readout | 7 | completed | `{"answer_template_mode": "mixed", "chunk_size": 64, "dim": 16, "distractor_records_per_case": 2, "epochs": 1, "evidence_hint_weight": 0.0, "evidence_loss_weight": 0.2, "generalization_score_mode": "generation", "generation_readout_mode": "extract_then_compose", "mhdsra2_config_override": {}, "read_topk": 2, "seed_bundle": {"model_seed": 411, "pair_split_seed": 310, "seed_root": 7, "test_dataset_seed": 209, "train_dataset_seed": 7, "validation_dataset_seed": 108}, "slot_decoder_logit_bias": 0.0, "slot_decoder_loss_weight": 0.0, "slots": 8, "test_dataset_size": 1, "train_dataset_size": 1, "validation_dataset_size": 1}` | `{"validation_generation_exact_match_rate": 0.0, "validation_generation_mean_sequence_accuracy": 0.030303030303030304}` | `{"test_generation_exact_match_rate": 0.0, "test_generation_mean_sequence_accuracy": 0.03076923076923077}` | 18.23 |

## Diagnostics

| task | group | seed | diagnostics |
|---|---|---:|---|
| json | baseline | 7 | `{"test_evidence_window_accuracy": null, "test_evidence_window_mean_distance": null, "test_extract_then_compose_exact_match_rate": null, "test_extract_then_compose_mean_sequence_accuracy": null, "test_slot_decoder_artifact_accuracy": null, "test_slot_decoder_artist_accuracy": null, "test_slot_decoder_dynasty_accuracy": null, "test_slot_decoder_full_answer_accuracy": null, "test_slot_decoder_museum_accuracy": null, "test_teacher_forced_exact_match_rate": 0.0, "test_teacher_forced_mean_sequence_accuracy": 0.007692307692307693, "validation_evidence_window_accuracy": null, "validation_evidence_window_mean_distance": null, "validation_extract_then_compose_exact_match_rate": null, "validation_extract_then_compose_mean_sequence_accuracy": null, "validation_slot_decoder_artifact_accuracy": null, "validation_slot_decoder_artist_accuracy": null, "validation_slot_decoder_dynasty_accuracy": null, "validation_slot_decoder_full_answer_accuracy": null, "validation_slot_decoder_museum_accuracy": null, "validation_teacher_forced_exact_match_rate": 0.0, "validation_teacher_forced_mean_sequence_accuracy": 0.007575757575757576}` |
| json | extract_compose_readout | 7 | `{"test_evidence_window_accuracy": 0.0, "test_evidence_window_mean_distance": 5.0, "test_extract_then_compose_exact_match_rate": 0.0, "test_extract_then_compose_mean_sequence_accuracy": 0.03076923076923077, "test_slot_decoder_artifact_accuracy": null, "test_slot_decoder_artist_accuracy": null, "test_slot_decoder_dynasty_accuracy": null, "test_slot_decoder_full_answer_accuracy": null, "test_slot_decoder_museum_accuracy": null, "test_teacher_forced_exact_match_rate": 0.0, "test_teacher_forced_mean_sequence_accuracy": 0.007692307692307693, "validation_evidence_window_accuracy": 0.0, "validation_evidence_window_mean_distance": 1.0, "validation_extract_then_compose_exact_match_rate": 0.0, "validation_extract_then_compose_mean_sequence_accuracy": 0.030303030303030304, "validation_slot_decoder_artifact_accuracy": null, "validation_slot_decoder_artist_accuracy": null, "validation_slot_decoder_dynasty_accuracy": null, "validation_slot_decoder_full_answer_accuracy": null, "validation_slot_decoder_museum_accuracy": null, "validation_teacher_forced_exact_match_rate": 0.0, "validation_teacher_forced_mean_sequence_accuracy": 0.007575757575757576}` |

## Success Summary

- niah_rows_completed: `0`
- json_rows_completed: `2`
- two_digit_rows_completed: `0`
- two_digit_min_exact_match: `None`
- selection_policy: `Select candidates by validation metrics only; inspect held-out test metrics only after a candidate improves validation.`
