# MHDSRA2 Carry Diagnostic Grid

## Configuration

- Layers: 1
- Max steps per stage values: 2
- Curriculum eval intervals: 1
- Learning rates: 0.01
- Training strategies: baseline, carry_replay
- Seeds: 101
- Replay ratio: 0.75
- Stage patience: 1
- Carry replay ratio: 0.5
- Stage loss weights: {'unit_with_carry': 2.0}
- Checkpoint path: reports/test_mhdsra2_carry_diagnostic_grid.checkpoint.jsonl
- Resume supported: True

## Summary

- Stable target strategy count: 0
- Has stable target strategy: False
- Carry-only success count: 0

## Aggregates

| Dataset | Strategy | LR | Eval Interval | Max Steps | Layers | Runs | Carry EM Mean | Target Retention Rate | Retained Mean | Train EM Mean | Final Loss Mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| curriculum_rule_set | baseline | 0.0100 | 1 | 2 | 1 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.6322 |
| curriculum_rule_set | carry_replay | 0.0100 | 1 | 2 | 1 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.6322 |
| unit_with_carry_only | baseline | 0.0100 | 1 | 2 | 1 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.6693 |
| unit_with_carry_only | carry_replay | 0.0100 | 1 | 2 | 1 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.6693 |
