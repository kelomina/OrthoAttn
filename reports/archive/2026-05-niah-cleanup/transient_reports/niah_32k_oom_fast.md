# MHDSRA2 2M NIAH Verification

## Summary

- status: `completed`
- sequence length: `32768`
- final eval source: `light`
- best eval source: `light`
- final eval mean accuracy: `0.00%`
- final eval min-depth accuracy: `0.00%`
- best eval mean accuracy: `0.00%`
- best eval min-depth accuracy: `0.00%`
- best accuracy step: `0`
- best accuracy loss: `4.720329`
- final step train accuracy: `0.00%`
- best step train accuracy: `0.00%`
- passed target accuracy: `False`
- passed success criteria: `False`
- re-eval mean accuracy: `0.00%`
- re-eval min-depth accuracy: `0.00%`
- final loss: `4.788864`
- elapsed seconds: `12.371`
- peak allocated memory: `4518.74 MB`
- peak reserved memory: `4702.00 MB`
- parameter count: `62576`
- device: `cuda:0`
- CUDA device: `NVIDIA GeForce RTX 4070 Laptop GPU`
- torch: `2.11.0+cu130`
- torch CUDA: `13.0`

## Config

| Field | Value |
|---|---:|
| vocab_size | `100` |
| dim | `64` |
| num_layers | `2` |
| slots | `64` |
| read_topk | `8` |
| chunk_size | `1024` |
| batch_size | `8` |
| epochs | `1` |
| optimizer_steps | `1` |
| learning_rate | `0.001` |
| seed | `20260506` |
| target_accuracy | `1.0` |
| stop_loss | `0.1` |
| eval_interval | `5` |
| eval_batches_per_depth | `1` |
| light_eval_batches_per_depth | `1` |
| robust_eval_interval | `5` |
| robust_eval_batches_per_depth | `1` |
| eval_depths | `[0.1, 0.5, 0.9]` |
| cudnn_benchmark | `False` |

## Observed Steps

| Step | Train Depth | Train Loss | Train Accuracy | Light Mean Accuracy | Light Min-Depth Accuracy | Light Loss | Top-3 | Target Rank | Entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1 | 4.391278 | 0.00% | 0.00% | 0.00% | 4.720329 | 0.00% | 49.58 | 4.4769 |

## Latest Light Per-Depth Metrics

| Depth | Accuracy | Loss | Top-3 | Target Rank | Target Prob | Logit Margin | Entropy | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.00% | 4.737947 | 0.00% | 51.62 | 0.0092 | -1.0149 | 4.4768 | 8 |
| 0.5 | 0.00% | 4.559589 | 0.00% | 40.38 | 0.0114 | -0.8345 | 4.4769 | 8 |
| 0.9 | 0.00% | 4.863450 | 0.00% | 56.75 | 0.0091 | -1.1475 | 4.4769 | 8 |
