# MHDSRA2 2M NIAH Verification

## Summary

- status: `completed`
- sequence length: `8192`
- final eval source: `robust`
- best eval source: `robust`
- final eval mean accuracy: `1.56%`
- final eval min-depth accuracy: `1.56%`
- best eval mean accuracy: `1.56%`
- best eval min-depth accuracy: `0.00%`
- best accuracy step: `1`
- best accuracy loss: `4.749357`
- final step train accuracy: `0.00%`
- best step train accuracy: `0.00%`
- passed target accuracy: `False`
- passed success criteria: `False`
- re-eval mean accuracy: `1.56%`
- re-eval min-depth accuracy: `1.56%`
- final loss: `4.733769`
- elapsed seconds: `38.099`
- peak allocated memory: `1138.28 MB`
- peak reserved memory: `1180.00 MB`
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
| batch_size | `4` |
| epochs | `2` |
| optimizer_steps | `2` |
| learning_rate | `0.001` |
| seed | `20260506` |
| target_accuracy | `1.0` |
| stop_loss | `0.1` |
| eval_interval | `2` |
| eval_batches_per_depth | `1` |
| light_eval_batches_per_depth | `1` |
| robust_eval_interval | `2` |
| robust_eval_batches_per_depth | `32` |
| eval_depths | `[0.1, 0.5, 0.9]` |
| cudnn_benchmark | `False` |

## Observed Steps

| Step | Train Depth | Train Loss | Train Accuracy | Light Mean Accuracy | Light Min-Depth Accuracy | Light Loss | Top-3 | Target Rank | Entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1 | 7.132210 | 0.00% | 0.00% | 0.00% | 4.678371 | 0.00% | 47.08 | 4.4825 |
| 1 | 0.5 | 7.038217 | 0.00% | 8.33% | 0.00% | 4.472642 | 16.67% | 38.42 | 4.4826 |

## Robust Evaluations

| Step | Mean Accuracy | Min-Depth Accuracy | Mean Loss | Top-3 | Target Rank | Entropy | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.56% | 0.00% | 4.749357 | 3.91% | 51.23 | 4.4823 | 384 |

## Latest Light Per-Depth Metrics

| Depth | Accuracy | Loss | Top-3 | Target Rank | Target Prob | Logit Margin | Entropy | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 25.00% | 4.553814 | 25.00% | 47.75 | 0.0124 | -0.9606 | 4.4834 | 4 |
| 0.5 | 0.00% | 4.332902 | 25.00% | 28.50 | 0.0145 | -0.8281 | 4.4818 | 4 |
| 0.9 | 0.00% | 4.531210 | 0.00% | 39.00 | 0.0111 | -1.0127 | 4.4826 | 4 |
