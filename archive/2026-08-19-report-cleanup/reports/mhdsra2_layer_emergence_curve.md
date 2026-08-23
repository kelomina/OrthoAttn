# MHDSRA2 Decimal Arithmetic Emergence

## Proxy Definition

The headline probe asks whether a model trained only on low-value decimal addition curriculum stages can greedily generate `200<eos>` for `100+100=`.

## Configuration

- Layers: 1
- Seeds: 101
- Max steps per curriculum stage: 2
- Curriculum stage EM threshold: 0.95
- Curriculum eval interval: 1
- Replay ratio: 0.25
- Stage patience: 1
- Learning rate: 0.01
- Device: cpu
- Minimum curriculum mastery layers: null
- Minimum arithmetic emergent layers: null

## Dataset Specs

### curriculum_rule_set
- Training: 0+0=0, 1+1=2, 2+3=5, 4+5=9, 5+5=10, 8+2=10, 9+1=10, 9+9=18, 10+10=20, 11+11=22, 12+12=24, 20+20=40, 30+40=70, 55+44=99
- Headline: 100+100=200
- OOD: 101+101=202, 110+110=220, 99+1=100, 20+80=100
- Curriculum:
  - unit_no_carry: 0+0=0, 1+1=2, 2+3=5, 4+5=9
  - unit_with_carry: 5+5=10, 8+2=10, 9+1=10, 9+9=18
  - two_digit_rules: 10+10=20, 11+11=22, 12+12=24, 20+20=40, 30+40=70, 55+44=99

### single_fact_only
- Training: 1+1=2
- Headline: 100+100=200
- OOD: 101+101=202, 110+110=220, 99+1=100, 20+80=100

## Multi-Seed Summary

| Dataset | Model | Layers | Seeds | Train EM Mean | Headline EM Mean | OOD EM Mean | Final Loss Mean | Meets Criteria |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| curriculum_rule_set | mhdsra2 | 1 | 1 | 0.0000 | 0.0000 | 0.0000 | 2.6322 | no |
| single_fact_only | mhdsra2 | 1 | 1 | 1.0000 | 0.0000 | 0.0000 | 2.3746 | no |

## Training Stop Summary

| Dataset | Model | Layers | Seed | Steps Executed | Ever Passed | Retained | Stop Reason |
|---|---|---:|---:|---:|---:|---:|---|
| curriculum_rule_set | mhdsra2 | 1 | 101 | 2 | 0 | 0 | stage_max_steps_exhausted:unit_no_carry |
| single_fact_only | mhdsra2 | 1 | 101 | 2 | 0 | 0 | max_steps_exhausted |

## Curriculum Stage Aggregate

| Dataset | Model | Layers | Stage | Runs | Pass Rate | Mean Pass Step | Final EM Mean |
|---|---|---:|---|---:|---:|---:|---:|
| curriculum_rule_set | mhdsra2 | 1 | two_digit_rules | 1 | 0.0000 | - | 0.0000 |
| curriculum_rule_set | mhdsra2 | 1 | unit_no_carry | 1 | 0.0000 | - | 0.0000 |
| curriculum_rule_set | mhdsra2 | 1 | unit_with_carry | 1 | 0.0000 | - | 0.0000 |
