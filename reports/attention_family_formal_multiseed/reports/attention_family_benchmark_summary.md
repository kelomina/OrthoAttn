# Attention Family Benchmark Summary

## Test Design
- Complexity benchmark: random token forward pass with shared `dim`, `chunk_size`, and local-context setup.
- Task benchmark: held-out `museum/artifact` JSON retrieval generalization under identical training settings.
- Task seed roots: `[7, 11, 19, 29, 37]`
- Task variants:
  baseline: plain end-to-end token decoding
  evidence_extract: add evidence window supervision and report extract-then-compose recovery

## Complexity Benchmark
- Device: `cuda:0`
- Sequence Lengths: `[1024, 4096, 16384, 32768]`
- Batch Size: `1`

### DSRA
- seq_len=1024: time=`13.23 ms +/- 0.46` | peak_mem=`63.58 MB`
- seq_len=4096: time=`49.88 ms +/- 8.01` | peak_mem=`68.65 MB`
- seq_len=16384: time=`151.87 ms +/- 5.25` | peak_mem=`98.26 MB`
- seq_len=32768: time=`300.96 ms +/- 23.69` | peak_mem=`186.57 MB`

### Sparse Attention
- Family Config: `{'sparse_local_window': 512, 'sparse_global_stride': 1024}`
- seq_len=1024: time=`3.23 ms +/- 0.33` | peak_mem=`46.77 MB`
- seq_len=4096: time=`17.51 ms +/- 9.11` | peak_mem=`31.96 MB`
- seq_len=16384: time=`45.31 ms +/- 2.80` | peak_mem=`98.20 MB`
- seq_len=32768: time=`85.52 ms +/- 0.48` | peak_mem=`186.51 MB`

### Sliding Window Attention
- Family Config: `{'window_size': 1024}`
- seq_len=1024: time=`2.56 ms +/- 0.38` | peak_mem=`47.65 MB`
- seq_len=4096: time=`7.53 ms +/- 0.21` | peak_mem=`31.96 MB`
- seq_len=16384: time=`41.12 ms +/- 0.64` | peak_mem=`98.20 MB`
- seq_len=32768: time=`68.72 ms +/- 12.80` | peak_mem=`186.51 MB`

### Linear Attention
- seq_len=1024: time=`188.85 ms +/- 6.47` | peak_mem=`46.77 MB`
- seq_len=4096: time=`809.48 ms +/- 26.51` | peak_mem=`31.96 MB`
- seq_len=16384: time=`4007.86 ms +/- 9.28` | peak_mem=`98.20 MB`
- seq_len=32768: time=`7163.96 ms +/- 743.78` | peak_mem=`186.51 MB`

### Standard Attention
- seq_len=1024: time=`4.33 ms +/- 0.01` | peak_mem=`47.65 MB`
- seq_len=4096: time=`8.95 ms +/- 0.92` | peak_mem=`35.45 MB`
- seq_len=16384: time=`45.82 ms +/- 6.99` | peak_mem=`110.57 MB`
- seq_len=32768: time=`132.08 ms +/- 0.74` | peak_mem=`211.01 MB`

## JSON Retrieval Comparison

### baseline
- Variant Config: `{'evidence_loss_weight': 0.0, 'evidence_hint_weight': 0.0}`
- DSRA: val_tf_seq=`50.83% +/- 3.07` | test_tf_seq=`50.52% +/- 3.30` | val_gen_seq=`10.54% +/- 2.20` | test_gen_seq=`11.24% +/- 2.36`
  report root: `reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\dsra`
  seed=7: test_tf_seq=`47.40%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\dsra\seed_7\reports`
  seed=11: test_tf_seq=`47.63%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\dsra\seed_11\reports`
  seed=19: test_tf_seq=`49.54%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\dsra\seed_19\reports`
  seed=29: test_tf_seq=`54.00%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\dsra\seed_29\reports`
  seed=37: test_tf_seq=`54.03%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\dsra\seed_37\reports`
- Sparse Attention: val_tf_seq=`42.80% +/- 2.81` | test_tf_seq=`42.61% +/- 2.54` | val_gen_seq=`9.68% +/- 2.14` | test_gen_seq=`10.53% +/- 2.67`
  report root: `reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\sparse_attention`
  seed=7: test_tf_seq=`44.53%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\sparse_attention\seed_7\reports`
  seed=11: test_tf_seq=`45.81%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\sparse_attention\seed_11\reports`
  seed=19: test_tf_seq=`40.14%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\sparse_attention\seed_19\reports`
  seed=29: test_tf_seq=`42.38%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\sparse_attention\seed_29\reports`
  seed=37: test_tf_seq=`40.21%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\sparse_attention\seed_37\reports`
- Sliding Window Attention: val_tf_seq=`43.89% +/- 1.80` | test_tf_seq=`43.07% +/- 2.15` | val_gen_seq=`9.18% +/- 2.20` | test_gen_seq=`8.87% +/- 2.50`
  report root: `reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\sliding_window_attention`
  seed=7: test_tf_seq=`44.32%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\sliding_window_attention\seed_7\reports`
  seed=11: test_tf_seq=`46.05%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\sliding_window_attention\seed_11\reports`
  seed=19: test_tf_seq=`40.56%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\sliding_window_attention\seed_19\reports`
  seed=29: test_tf_seq=`42.57%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\sliding_window_attention\seed_29\reports`
  seed=37: test_tf_seq=`41.86%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\sliding_window_attention\seed_37\reports`
- Linear Attention: val_tf_seq=`16.29% +/- 0.17` | test_tf_seq=`16.36% +/- 0.10` | val_gen_seq=`16.29% +/- 0.17` | test_gen_seq=`16.36% +/- 0.10`
  report root: `reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\linear_attention`
  seed=7: test_tf_seq=`16.32%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\linear_attention\seed_7\reports`
  seed=11: test_tf_seq=`16.51%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\linear_attention\seed_11\reports`
  seed=19: test_tf_seq=`16.29%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\linear_attention\seed_19\reports`
  seed=29: test_tf_seq=`16.42%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\linear_attention\seed_29\reports`
  seed=37: test_tf_seq=`16.29%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\linear_attention\seed_37\reports`
- Standard Attention: val_tf_seq=`44.01% +/- 1.33` | test_tf_seq=`42.92% +/- 1.97` | val_gen_seq=`10.38% +/- 0.92` | test_gen_seq=`10.10% +/- 1.78`
  report root: `reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\standard_attention`
  seed=7: test_tf_seq=`43.39%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\standard_attention\seed_7\reports`
  seed=11: test_tf_seq=`45.62%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\standard_attention\seed_11\reports`
  seed=19: test_tf_seq=`40.75%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\standard_attention\seed_19\reports`
  seed=29: test_tf_seq=`43.63%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\standard_attention\seed_29\reports`
  seed=37: test_tf_seq=`41.24%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\baseline\standard_attention\seed_37\reports`

### evidence_extract
- Variant Config: `{'evidence_window_count': 16, 'evidence_loss_weight': 0.5, 'evidence_hint_weight': 0.0, 'evidence_min_context_bytes': 16384}`
- DSRA: val_tf_seq=`50.38% +/- 3.04` | test_tf_seq=`50.03% +/- 2.78` | val_gen_seq=`10.66% +/- 2.26` | test_gen_seq=`9.48% +/- 2.44`
  evidence: val_window=`35.00% +/- 28.50` | test_window=`15.00% +/- 22.36` | val_extract_exact=`35.00% +/- 28.50` | test_extract_exact=`15.00% +/- 22.36`
  report root: `reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\dsra`
  seed=7: test_tf_seq=`47.56%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\dsra\seed_7\reports`
  seed=11: test_tf_seq=`47.63%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\dsra\seed_11\reports`
  seed=19: test_tf_seq=`48.93%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\dsra\seed_19\reports`
  seed=29: test_tf_seq=`53.40%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\dsra\seed_29\reports`
  seed=37: test_tf_seq=`52.60%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\dsra\seed_37\reports`
- Sparse Attention: val_tf_seq=`42.75% +/- 2.91` | test_tf_seq=`42.44% +/- 2.73` | val_gen_seq=`9.39% +/- 1.75` | test_gen_seq=`10.18% +/- 2.25`
  evidence: val_window=`30.00% +/- 20.92` | test_window=`20.00% +/- 32.60` | val_extract_exact=`30.00% +/- 20.92` | test_extract_exact=`20.00% +/- 32.60`
  report root: `reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\sparse_attention`
  seed=7: test_tf_seq=`44.32%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\sparse_attention\seed_7\reports`
  seed=11: test_tf_seq=`46.02%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\sparse_attention\seed_11\reports`
  seed=19: test_tf_seq=`39.55%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\sparse_attention\seed_19\reports`
  seed=29: test_tf_seq=`42.13%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\sparse_attention\seed_29\reports`
  seed=37: test_tf_seq=`40.21%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\sparse_attention\seed_37\reports`
- Sliding Window Attention: val_tf_seq=`43.70% +/- 2.02` | test_tf_seq=`42.91% +/- 2.28` | val_gen_seq=`10.10% +/- 2.19` | test_gen_seq=`9.03% +/- 2.87`
  evidence: val_window=`25.00% +/- 30.62` | test_window=`25.00% +/- 35.36` | val_extract_exact=`25.00% +/- 30.62` | test_extract_exact=`25.00% +/- 35.36`
  report root: `reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\sliding_window_attention`
  seed=7: test_tf_seq=`44.32%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\sliding_window_attention\seed_7\reports`
  seed=11: test_tf_seq=`46.05%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\sliding_window_attention\seed_11\reports`
  seed=19: test_tf_seq=`40.37%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\sliding_window_attention\seed_19\reports`
  seed=29: test_tf_seq=`42.38%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\sliding_window_attention\seed_29\reports`
  seed=37: test_tf_seq=`41.45%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\sliding_window_attention\seed_37\reports`
- Linear Attention: val_tf_seq=`16.29% +/- 0.17` | test_tf_seq=`16.36% +/- 0.10` | val_gen_seq=`16.29% +/- 0.17` | test_gen_seq=`16.36% +/- 0.10`
  evidence: val_window=`0.00% +/- 0.00` | test_window=`15.00% +/- 13.69` | val_extract_exact=`0.00% +/- 0.00` | test_extract_exact=`15.00% +/- 13.69`
  report root: `reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\linear_attention`
  seed=7: test_tf_seq=`16.32%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\linear_attention\seed_7\reports`
  seed=11: test_tf_seq=`16.51%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\linear_attention\seed_11\reports`
  seed=19: test_tf_seq=`16.29%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\linear_attention\seed_19\reports`
  seed=29: test_tf_seq=`16.42%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\linear_attention\seed_29\reports`
  seed=37: test_tf_seq=`16.29%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\linear_attention\seed_37\reports`
- Standard Attention: val_tf_seq=`43.95% +/- 1.15` | test_tf_seq=`43.02% +/- 1.95` | val_gen_seq=`8.85% +/- 1.84` | test_gen_seq=`9.99% +/- 1.85`
  evidence: val_window=`25.00% +/- 17.68` | test_window=`25.00% +/- 17.68` | val_extract_exact=`25.00% +/- 17.68` | test_extract_exact=`15.00% +/- 22.36`
  report root: `reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\standard_attention`
  seed=7: test_tf_seq=`43.64%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\standard_attention\seed_7\reports`
  seed=11: test_tf_seq=`45.62%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\standard_attention\seed_11\reports`
  seed=19: test_tf_seq=`40.76%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\standard_attention\seed_19\reports`
  seed=29: test_tf_seq=`43.65%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\standard_attention\seed_29\reports`
  seed=37: test_tf_seq=`41.44%` | report=`reports\attention_family_formal_multiseed\reports\attention_family_json_retrieval\reports\evidence_extract\standard_attention\seed_37\reports`

