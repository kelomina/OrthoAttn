# JSON Retrieval Generalization Report

## Validation Pool
- Cases: `4`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `16.61%`
- Generation Mean Prefix Match Length: `0.00`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `16.61%`
- Teacher-Forced Mean Prefix Match Length: `0.00`

## Test Pool
- Cases: `4`
- Generation Exact Match Rate: `0.00%`
- Generation Mean Sequence Accuracy: `16.28%`
- Generation Mean Prefix Match Length: `0.00`
- Teacher-Forced Exact Match Rate: `0.00%`
- Teacher-Forced Mean Sequence Accuracy: `16.28%`
- Teacher-Forced Mean Prefix Match Length: `0.00`

## Training Config
- Device: `cuda:0`
- Epochs: `40`
- Dim: `64`
- K: `64`
- kr: `16`
- Chunk Size: `256`
- Learning Rate: `0.0005`
- Warmup Ratio: `0.2`
- Local Context Mode: `concat`
- Local Context Size: `4`
- Train Dataset Size: `12`
- Train Dataset Seed: `7`
- Validation Dataset Size: `4`
- Validation Dataset Seed: `17`
- Test Dataset Size: `4`
- Test Dataset Seed: `23`

## Search Summary
- Trial 1: kr=16, chunk_size=256, lr=0.0005, warmup_ratio=0.2, val_gen_exact=0.00%, val_tf_exact=0.00%, test_gen_exact=0.00%, test_tf_exact=0.00%
