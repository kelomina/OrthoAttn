# MHDSRA2 Curriculum Strategy Grid

## Objective

Find whether replay ratio and stage patience can stably retain the target curriculum stages: unit_no_carry, unit_with_carry.

## Configuration

- Replay ratios: 0.25
- Stage patiences: 1
- Layers: 1
- Seeds: 101
- Max steps per stage values: 2
- Curriculum eval interval: 1
- Stage threshold: 0.95
- Target stage count: 2

## Summary

- Stable target strategy count: 0
- Has stable target strategy: False
- Best strategy: replay_ratio=0.25, stage_patience=1, max_steps_per_stage=2, layers=1, target_retention_rate=0.0000

## Grid Results

| Replay Ratio | Stage Patience | Max Steps | Layers | Runs | Target Retention Rate | Stable | Retained Mean | Ever Passed Mean | Train EM Mean | Final Loss Mean |
|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| 0.25 | 1 | 2 | 1 | 1 | 0.0000 | no | 0.0000 | 0.0000 | 0.0000 | 2.6482 |
