# mhdsra2_slot_readout_dry_run

- device: `cpu`
- groups: `baseline, slot_readout_bias, evidence_slot_readout`
- tasks: `json`
- dry_run: `True`

## Groups

- `baseline`: Current default MHDSRA2 configuration. override=`{}`
- `slot_readout_bias`: Train the existing JSON slot decoder and apply its generation-time byte-level readout bias. override=`{}`
- `evidence_slot_readout`: Train evidence-window supervision plus the existing JSON slot decoder, then use the slot readout bias during generation. override=`{}`

## Rows

| task | group | seed | status | config | validation | test | elapsed_sec |
|---|---|---:|---|---|---|---|---:|

## Diagnostics

| task | group | seed | diagnostics |
|---|---|---:|---|

## Success Summary

- niah_rows_completed: `0`
- json_rows_completed: `0`
- two_digit_rows_completed: `0`
- two_digit_min_exact_match: `None`
- selection_policy: `Select candidates by validation metrics only; inspect held-out test metrics only after a candidate improves validation.`
