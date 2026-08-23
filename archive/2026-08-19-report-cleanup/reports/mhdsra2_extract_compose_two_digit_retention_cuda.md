# mhdsra2_extract_compose_two_digit_retention_cuda

- device: `cuda:0`
- groups: `baseline, extract_compose_readout`
- tasks: `two_digit`
- dry_run: `False`

## Groups

- `baseline`: Current default MHDSRA2 configuration. override=`{}`
- `extract_compose_readout`: Use the existing evidence-window decoder plus deterministic extract-then-compose answer readout for JSON generation metrics. override=`{}`

## Rows

| task | group | seed | status | config | validation | test | elapsed_sec |
|---|---|---:|---|---|---|---|---:|
| two_digit | baseline_holdout | 101 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 4, "training_strategy": "baseline"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 30.32 |
| two_digit | baseline_holdout | 101 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 4, "training_strategy": "two_digit_replay"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 36.60 |
| two_digit | baseline_holdout | 101 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 4, "training_strategy": "two_digit_weighted_loss"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 36.30 |
| two_digit | baseline_holdout | 101 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 4, "training_strategy": "combined"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 36.19 |
| two_digit | baseline_holdout | 101 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 8, "training_strategy": "baseline"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 70.42 |
| two_digit | baseline_holdout | 101 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 8, "training_strategy": "two_digit_replay"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 69.30 |
| two_digit | baseline_holdout | 101 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 8, "training_strategy": "two_digit_weighted_loss"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 69.98 |
| two_digit | baseline_holdout | 101 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 8, "training_strategy": "combined"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 70.04 |
| two_digit | baseline_holdout | 202 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 4, "training_strategy": "baseline"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 36.10 |
| two_digit | baseline_holdout | 202 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 4, "training_strategy": "two_digit_replay"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 35.67 |
| two_digit | baseline_holdout | 202 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 4, "training_strategy": "two_digit_weighted_loss"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 36.20 |
| two_digit | baseline_holdout | 202 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 4, "training_strategy": "combined"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 35.72 |
| two_digit | baseline_holdout | 202 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 8, "training_strategy": "baseline"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 69.91 |
| two_digit | baseline_holdout | 202 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 8, "training_strategy": "two_digit_replay"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 71.47 |
| two_digit | baseline_holdout | 202 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 8, "training_strategy": "two_digit_weighted_loss"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 70.68 |
| two_digit | baseline_holdout | 202 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 8, "training_strategy": "combined"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 69.93 |
| two_digit | baseline_holdout | 303 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 4, "training_strategy": "baseline"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 35.89 |
| two_digit | baseline_holdout | 303 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 4, "training_strategy": "two_digit_replay"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 36.09 |
| two_digit | baseline_holdout | 303 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 4, "training_strategy": "two_digit_weighted_loss"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 35.86 |
| two_digit | baseline_holdout | 303 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 4, "training_strategy": "combined"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 36.35 |
| two_digit | baseline_holdout | 303 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 8, "training_strategy": "baseline"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 69.50 |
| two_digit | baseline_holdout | 303 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 8, "training_strategy": "two_digit_replay"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 69.48 |
| two_digit | baseline_holdout | 303 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 8, "training_strategy": "two_digit_weighted_loss"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 69.72 |
| two_digit | baseline_holdout | 303 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 8, "training_strategy": "combined"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 70.38 |

## Diagnostics

| task | group | seed | diagnostics |
|---|---|---:|---|
| two_digit | baseline_holdout | 101 | `{"slot_collision": {"available": false, "reason": "arithmetic_factory_does_not_expose_mhdsra2_aux"}}` |
| two_digit | baseline_holdout | 101 | `{"slot_collision": {"available": false, "reason": "arithmetic_factory_does_not_expose_mhdsra2_aux"}}` |
| two_digit | baseline_holdout | 101 | `{"slot_collision": {"available": false, "reason": "arithmetic_factory_does_not_expose_mhdsra2_aux"}}` |
| two_digit | baseline_holdout | 101 | `{"slot_collision": {"available": false, "reason": "arithmetic_factory_does_not_expose_mhdsra2_aux"}}` |
| two_digit | baseline_holdout | 101 | `{"slot_collision": {"available": false, "reason": "arithmetic_factory_does_not_expose_mhdsra2_aux"}}` |
| two_digit | baseline_holdout | 101 | `{"slot_collision": {"available": false, "reason": "arithmetic_factory_does_not_expose_mhdsra2_aux"}}` |
| two_digit | baseline_holdout | 101 | `{"slot_collision": {"available": false, "reason": "arithmetic_factory_does_not_expose_mhdsra2_aux"}}` |
| two_digit | baseline_holdout | 101 | `{"slot_collision": {"available": false, "reason": "arithmetic_factory_does_not_expose_mhdsra2_aux"}}` |
| two_digit | baseline_holdout | 202 | `{"slot_collision": {"available": false, "reason": "arithmetic_factory_does_not_expose_mhdsra2_aux"}}` |
| two_digit | baseline_holdout | 202 | `{"slot_collision": {"available": false, "reason": "arithmetic_factory_does_not_expose_mhdsra2_aux"}}` |
| two_digit | baseline_holdout | 202 | `{"slot_collision": {"available": false, "reason": "arithmetic_factory_does_not_expose_mhdsra2_aux"}}` |
| two_digit | baseline_holdout | 202 | `{"slot_collision": {"available": false, "reason": "arithmetic_factory_does_not_expose_mhdsra2_aux"}}` |
| two_digit | baseline_holdout | 202 | `{"slot_collision": {"available": false, "reason": "arithmetic_factory_does_not_expose_mhdsra2_aux"}}` |
| two_digit | baseline_holdout | 202 | `{"slot_collision": {"available": false, "reason": "arithmetic_factory_does_not_expose_mhdsra2_aux"}}` |
| two_digit | baseline_holdout | 202 | `{"slot_collision": {"available": false, "reason": "arithmetic_factory_does_not_expose_mhdsra2_aux"}}` |
| two_digit | baseline_holdout | 202 | `{"slot_collision": {"available": false, "reason": "arithmetic_factory_does_not_expose_mhdsra2_aux"}}` |
| two_digit | baseline_holdout | 303 | `{"slot_collision": {"available": false, "reason": "arithmetic_factory_does_not_expose_mhdsra2_aux"}}` |
| two_digit | baseline_holdout | 303 | `{"slot_collision": {"available": false, "reason": "arithmetic_factory_does_not_expose_mhdsra2_aux"}}` |
| two_digit | baseline_holdout | 303 | `{"slot_collision": {"available": false, "reason": "arithmetic_factory_does_not_expose_mhdsra2_aux"}}` |
| two_digit | baseline_holdout | 303 | `{"slot_collision": {"available": false, "reason": "arithmetic_factory_does_not_expose_mhdsra2_aux"}}` |
| two_digit | baseline_holdout | 303 | `{"slot_collision": {"available": false, "reason": "arithmetic_factory_does_not_expose_mhdsra2_aux"}}` |
| two_digit | baseline_holdout | 303 | `{"slot_collision": {"available": false, "reason": "arithmetic_factory_does_not_expose_mhdsra2_aux"}}` |
| two_digit | baseline_holdout | 303 | `{"slot_collision": {"available": false, "reason": "arithmetic_factory_does_not_expose_mhdsra2_aux"}}` |
| two_digit | baseline_holdout | 303 | `{"slot_collision": {"available": false, "reason": "arithmetic_factory_does_not_expose_mhdsra2_aux"}}` |

## Success Summary

- niah_rows_completed: `0`
- json_rows_completed: `0`
- two_digit_rows_completed: `24`
- two_digit_min_exact_match: `1.0`
- selection_policy: `Select candidates by validation metrics only; inspect held-out test metrics only after a candidate improves validation.`
