# mhdsra2_readout_adapter_two_digit_retention_cuda

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
| two_digit | baseline_holdout | 101 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 4, "training_strategy": "baseline"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 32.30 |
| two_digit | baseline_holdout | 101 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 4, "training_strategy": "two_digit_replay"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 34.22 |
| two_digit | baseline_holdout | 101 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 4, "training_strategy": "two_digit_weighted_loss"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 37.07 |
| two_digit | baseline_holdout | 101 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 4, "training_strategy": "combined"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 36.91 |
| two_digit | baseline_holdout | 101 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 8, "training_strategy": "baseline"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 71.13 |
| two_digit | baseline_holdout | 101 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 8, "training_strategy": "two_digit_replay"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 71.33 |
| two_digit | baseline_holdout | 101 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 8, "training_strategy": "two_digit_weighted_loss"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 72.23 |
| two_digit | baseline_holdout | 101 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 8, "training_strategy": "combined"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 71.16 |
| two_digit | baseline_holdout | 202 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 4, "training_strategy": "baseline"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 36.77 |
| two_digit | baseline_holdout | 202 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 4, "training_strategy": "two_digit_replay"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 36.70 |
| two_digit | baseline_holdout | 202 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 4, "training_strategy": "two_digit_weighted_loss"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 36.97 |
| two_digit | baseline_holdout | 202 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 4, "training_strategy": "combined"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 36.72 |
| two_digit | baseline_holdout | 202 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 8, "training_strategy": "baseline"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 71.08 |
| two_digit | baseline_holdout | 202 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 8, "training_strategy": "two_digit_replay"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 71.03 |
| two_digit | baseline_holdout | 202 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 8, "training_strategy": "two_digit_weighted_loss"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 70.88 |
| two_digit | baseline_holdout | 202 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 8, "training_strategy": "combined"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 71.13 |
| two_digit | baseline_holdout | 303 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 4, "training_strategy": "baseline"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 36.71 |
| two_digit | baseline_holdout | 303 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 4, "training_strategy": "two_digit_replay"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 36.94 |
| two_digit | baseline_holdout | 303 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 4, "training_strategy": "two_digit_weighted_loss"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 37.11 |
| two_digit | baseline_holdout | 303 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 4, "training_strategy": "combined"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 36.86 |
| two_digit | baseline_holdout | 303 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 8, "training_strategy": "baseline"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 71.25 |
| two_digit | baseline_holdout | 303 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 8, "training_strategy": "two_digit_replay"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 71.66 |
| two_digit | baseline_holdout | 303 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 8, "training_strategy": "two_digit_weighted_loss"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 71.62 |
| two_digit | baseline_holdout | 303 | completed | `{"dataset": "two_digit_only", "learning_rate": 0.01, "max_steps_per_stage": 512, "num_layers": 8, "training_strategy": "combined"}` | `{}` | `{"two_digit_exact_match": 1.0}` | 71.61 |

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
