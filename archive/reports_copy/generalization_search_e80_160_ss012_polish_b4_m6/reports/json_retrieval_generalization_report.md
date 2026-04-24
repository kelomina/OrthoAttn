# JSON Retrieval Generalization Report

## Validation Pool
- Cases: `6`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `30.95%`
- Generation Mean Prefix Match Length: `33.00`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `73.37%`
- Teacher-Forced Mean Prefix Match Length: `33.00`

## Test Pool
- Cases: `6`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `32.68%`
- Generation Mean Prefix Match Length: `33.00`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `70.82%`
- Teacher-Forced Mean Prefix Match Length: `33.00`

## Training Config
- Device: `cuda:0`
- Epochs: `160`
- Dim: `128`
- K: `128`
- kr: `32`
- Chunk Size: `256`
- Learning Rate: `0.0005`
- Warmup Ratio: `0.2`
- Local Context Mode: `concat`
- Local Context Size: `4`
- Train Dataset Size: `24`
- Train Dataset Seed: `7`
- Pool Split Mode: `museum_artifact_held_out`
- Pair Split Seed: `29`
- Train Pair Count: `21`
- Validation Dataset Size: `6`
- Validation Dataset Seed: `17`
- Validation Pair Count: `7`
- Test Dataset Size: `6`
- Test Dataset Seed: `23`
- Test Pair Count: `8`
- Scheduled Sampling Max Ratio: `0.0`
- Final Generation Polish Epochs: `20`
- Final Generation Polish LR: `5e-05`
- Generation Polish Batch Size: `4`
- Generation Polish Monitor Case Count: `6`

## Search Summary
- Trial 1: epochs=160, kr=32, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.0, val_gen_exact=0.00%, val_tf_exact=0.00%, test_gen_exact=0.00%, test_tf_exact=0.00%
- Trial 2: epochs=160, kr=16, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.0, val_gen_exact=0.00%, val_tf_exact=0.00%, test_gen_exact=0.00%, test_tf_exact=0.00%
- Trial 3: epochs=160, kr=32, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.2, val_gen_exact=0.00%, val_tf_exact=0.00%, test_gen_exact=0.00%, test_tf_exact=0.00%
- Trial 4: epochs=160, kr=16, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.1, val_gen_exact=0.00%, val_tf_exact=0.00%, test_gen_exact=0.00%, test_tf_exact=0.00%
- Trial 5: epochs=160, kr=32, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.1, val_gen_exact=0.00%, val_tf_exact=0.00%, test_gen_exact=0.00%, test_tf_exact=0.00%
- Trial 6: epochs=160, kr=16, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.2, val_gen_exact=0.00%, val_tf_exact=0.00%, test_gen_exact=0.00%, test_tf_exact=0.00%
- Trial 7: epochs=80, kr=32, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.1, val_gen_exact=0.00%, val_tf_exact=0.00%, test_gen_exact=0.00%, test_tf_exact=0.00%
- Trial 8: epochs=80, kr=32, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.2, val_gen_exact=0.00%, val_tf_exact=0.00%, test_gen_exact=0.00%, test_tf_exact=0.00%
- Trial 9: epochs=80, kr=16, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.2, val_gen_exact=0.00%, val_tf_exact=0.00%, test_gen_exact=0.00%, test_tf_exact=0.00%
- Trial 10: epochs=80, kr=16, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.1, val_gen_exact=0.00%, val_tf_exact=0.00%, test_gen_exact=0.00%, test_tf_exact=0.00%
- Trial 11: epochs=80, kr=16, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.0, val_gen_exact=0.00%, val_tf_exact=0.00%, test_gen_exact=0.00%, test_tf_exact=0.00%
- Trial 12: epochs=80, kr=32, chunk_size=256, lr=0.0005, warmup_ratio=0.2, scheduled_sampling_max_ratio=0.0, val_gen_exact=0.00%, val_tf_exact=0.00%, test_gen_exact=0.00%, test_tf_exact=0.00%
