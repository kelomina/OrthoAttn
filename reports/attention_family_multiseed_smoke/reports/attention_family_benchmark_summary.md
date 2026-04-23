# Attention Family Benchmark Summary

## Test Design
- Complexity benchmark: random token forward pass with shared `dim`, `chunk_size`, and local-context setup.
- Task benchmark: held-out `museum/artifact` JSON retrieval generalization under identical training settings.
- Task seed roots: `[7, 11]`
- Task variants:
  baseline: plain end-to-end token decoding
  evidence_extract: add evidence window supervision and report extract-then-compose recovery

## Complexity Benchmark
- Device: `cuda:0`
- Sequence Lengths: `[64]`
- Batch Size: `1`

### DSRA
- seq_len=64: time=`282.04 ms +/- 0.00` | peak_mem=`9.25 MB`

### Standard Attention
- seq_len=64: time=`2.31 ms +/- 0.00` | peak_mem=`9.30 MB`

## JSON Retrieval Comparison

### baseline
- Variant Config: `{'evidence_loss_weight': 0.0, 'evidence_hint_weight': 0.0}`
- DSRA: val_tf_seq=`0.00% +/- 0.00` | test_tf_seq=`0.00% +/- 0.00` | val_gen_seq=`0.00% +/- 0.00` | test_gen_seq=`0.00% +/- 0.00`
  report root: `reports\attention_family_multiseed_smoke\reports\attention_family_json_retrieval\reports\baseline\dsra`
  seed=7: test_tf_seq=`0.00%` | report=`reports\attention_family_multiseed_smoke\reports\attention_family_json_retrieval\reports\baseline\dsra\seed_7\reports`
  seed=11: test_tf_seq=`0.00%` | report=`reports\attention_family_multiseed_smoke\reports\attention_family_json_retrieval\reports\baseline\dsra\seed_11\reports`
- Standard Attention: val_tf_seq=`0.38% +/- 0.54` | test_tf_seq=`0.00% +/- 0.00` | val_gen_seq=`0.38% +/- 0.54` | test_gen_seq=`0.00% +/- 0.00`
  report root: `reports\attention_family_multiseed_smoke\reports\attention_family_json_retrieval\reports\baseline\standard_attention`
  seed=7: test_tf_seq=`0.00%` | report=`reports\attention_family_multiseed_smoke\reports\attention_family_json_retrieval\reports\baseline\standard_attention\seed_7\reports`
  seed=11: test_tf_seq=`0.00%` | report=`reports\attention_family_multiseed_smoke\reports\attention_family_json_retrieval\reports\baseline\standard_attention\seed_11\reports`

