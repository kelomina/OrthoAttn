# MHDSRA2 Two-Digit Diagnostic Grid

## Configuration

- Device: cuda
- Datasets: two_digit_only
- Layers: 4, 8
- Max steps per stage values: 512
- Learning rates: 0.01
- Training strategies: baseline, two_digit_replay, two_digit_weighted_loss, combined
- Seeds: 101, 202, 303
- Replay ratio: 0.75
- Stage patience: 3
- Two-digit replay ratio: 0.75
- Stage loss weights: {'two_digit_rules': 2.0}
- Checkpoint path: reports\mhdsra2_two_digit_medium_grid.checkpoint.jsonl
- Resume supported: True

## Summary

- Stable target strategy count: 0
- Has stable target strategy: False
- Two-digit-only success count: 8

## Aggregates

| Dataset | Strategy | LR | Max Steps | Layers | Runs | Two-Digit EM Mean | Target Retention Rate | Retained Mean | Train EM Mean | Final Loss Mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| two_digit_only | baseline | 0.0100 | 512 | 4 | 3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.1279 |
| two_digit_only | baseline | 0.0100 | 512 | 8 | 3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.1276 |
| two_digit_only | combined | 0.0100 | 512 | 4 | 3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.2653 |
| two_digit_only | combined | 0.0100 | 512 | 8 | 3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.2616 |
| two_digit_only | two_digit_replay | 0.0100 | 512 | 4 | 3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.1279 |
| two_digit_only | two_digit_replay | 0.0100 | 512 | 8 | 3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.1276 |
| two_digit_only | two_digit_weighted_loss | 0.0100 | 512 | 4 | 3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.2653 |
| two_digit_only | two_digit_weighted_loss | 0.0100 | 512 | 8 | 3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.2616 |
