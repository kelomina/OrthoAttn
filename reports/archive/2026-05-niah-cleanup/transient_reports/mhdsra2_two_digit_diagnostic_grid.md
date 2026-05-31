# MHDSRA2 Two-Digit Diagnostic Grid

## Configuration

- Device: cpu
- Datasets: curriculum_rule_set, two_digit_only, prereq_plus_two_digit
- Layers: 1
- Max steps per stage values: 2
- Learning rates: 0.01
- Training strategies: baseline, two_digit_replay
- Seeds: 101
- Replay ratio: 0.75
- Stage patience: 1
- Two-digit replay ratios: 0.5
- Stage loss weights: {'two_digit_rules': 2.0}
- Checkpoint path: reports/test_mhdsra2_two_digit_diagnostic_grid.checkpoint.jsonl
- Resume supported: True

## Summary

- Stable target strategy count: 0
- Has stable target strategy: False
- Two-digit-only success count: 0

## Aggregates

| Dataset | Strategy | Two-Digit Replay Ratio | LR | Max Steps | Layers | Runs | Two-Digit EM Mean | Target Retention Rate | Retained Mean | Train EM Mean | Final Loss Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| curriculum_rule_set | baseline | 0.50 | 0.0100 | 2 | 1 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.6322 |
| curriculum_rule_set | two_digit_replay | 0.50 | 0.0100 | 2 | 1 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.6322 |
| prereq_plus_two_digit | baseline | 0.50 | 0.0100 | 2 | 1 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.6322 |
| prereq_plus_two_digit | two_digit_replay | 0.50 | 0.0100 | 2 | 1 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.6322 |
| two_digit_only | baseline | 0.50 | 0.0100 | 2 | 1 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.8272 |
| two_digit_only | two_digit_replay | 0.50 | 0.0100 | 2 | 1 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.8272 |
