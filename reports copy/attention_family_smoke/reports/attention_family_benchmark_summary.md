# Attention Family Benchmark Summary

## Test Design
- Complexity benchmark: random token forward pass with shared `dim`, `chunk_size`, and local-context setup.
- Task benchmark: held-out `museum/artifact` JSON retrieval generalization under identical training settings.
- Task variants:
  baseline: plain end-to-end token decoding
  evidence_extract: add evidence window supervision and report extract-then-compose recovery

## Complexity Benchmark
- Device: `cuda:0`
- Sequence Lengths: `[64]`
- Batch Size: `1`

### DSRA
- seq_len=64: time=`916.10 ms +/- 0.00` | peak_mem=`9.25 MB`

### Sparse Attention
- Family Config: `{'sparse_local_window': 32, 'sparse_global_stride': 64}`
- seq_len=64: time=`122.47 ms +/- 0.00` | peak_mem=`9.30 MB`

### Sliding Window Attention
- Family Config: `{'window_size': 64}`
- seq_len=64: time=`2.30 ms +/- 0.00` | peak_mem=`9.30 MB`

### Linear Attention
- seq_len=64: time=`16.63 ms +/- 0.00` | peak_mem=`9.30 MB`

### Standard Attention
- seq_len=64: time=`2.53 ms +/- 0.00` | peak_mem=`9.30 MB`

## JSON Retrieval Comparison

### baseline
- DSRA: val_tf_seq=`0.71%` | test_tf_seq=`0.00%` | val_gen_seq=`0.00%` | test_gen_seq=`0.00%`
  report: `reports\attention_family_smoke\reports\attention_family_json_retrieval\reports\baseline\dsra\reports`
- Sparse Attention: val_tf_seq=`0.00%` | test_tf_seq=`0.00%` | val_gen_seq=`0.00%` | test_gen_seq=`0.00%`
  report: `reports\attention_family_smoke\reports\attention_family_json_retrieval\reports\baseline\sparse_attention\reports`
- Sliding Window Attention: val_tf_seq=`0.00%` | test_tf_seq=`0.00%` | val_gen_seq=`0.00%` | test_gen_seq=`0.00%`
  report: `reports\attention_family_smoke\reports\attention_family_json_retrieval\reports\baseline\sliding_window_attention\reports`
- Linear Attention: val_tf_seq=`0.00%` | test_tf_seq=`0.00%` | val_gen_seq=`0.00%` | test_gen_seq=`0.00%`
  report: `reports\attention_family_smoke\reports\attention_family_json_retrieval\reports\baseline\linear_attention\reports`
- Standard Attention: val_tf_seq=`0.00%` | test_tf_seq=`0.00%` | val_gen_seq=`0.00%` | test_gen_seq=`0.00%`
  report: `reports\attention_family_smoke\reports\attention_family_json_retrieval\reports\baseline\standard_attention\reports`

