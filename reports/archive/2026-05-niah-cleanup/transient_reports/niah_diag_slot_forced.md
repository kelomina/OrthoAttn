# MHDSRA2 2M NIAH Verification

## Summary

- status: `success`
- sequence length: `1024`
- final eval source: `robust`
- best eval source: `robust`
- final eval mean accuracy: `100.00%`
- final eval min-depth accuracy: `100.00%`
- best eval mean accuracy: `100.00%`
- best eval min-depth accuracy: `100.00%`
- best accuracy step: `19`
- best accuracy loss: `0.012067`
- final step train accuracy: `100.00%`
- best step train accuracy: `100.00%`
- passed target accuracy: `True`
- passed success criteria: `True`
- final loss: `0.012067`
- elapsed seconds: `10.092`
- peak allocated memory: `350.85 MB`
- peak reserved memory: `402.00 MB`
- parameter count: `50321`
- device: `cuda:0`
- CUDA device: `NVIDIA GeForce RTX 4070 Laptop GPU`
- torch: `2.11.0+cu130`
- torch CUDA: `13.0`

## Config

| Field | Value |
|---|---:|
| vocab_size | `5` |
| dim | `64` |
| num_layers | `2` |
| slots | `64` |
| read_topk | `8` |
| chunk_size | `256` |
| batch_size | `8` |
| epochs | `200` |
| optimizer_steps | `200` |
| learning_rate | `0.001` |
| seed | `20260506` |
| target_accuracy | `1.0` |
| stop_loss | `0.1` |
| eval_interval | `20` |
| eval_batches_per_depth | `1` |
| light_eval_batches_per_depth | `1` |
| robust_eval_interval | `20` |
| robust_eval_batches_per_depth | `4` |
| eval_depths | `[0.1, 0.5, 0.9]` |
| cudnn_benchmark | `False` |

## Observed Steps

| Step | Train Depth | Train Loss | Train Accuracy | Light Mean Accuracy | Light Min-Depth Accuracy | Light Loss | Top-3 | Target Rank | Entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1 | 2.876474 | 0.00% | 0.00% | 0.00% | 1.691318 | 100.00% | 3.00 | 1.5797 |
| 1 | 0.5 | 2.385643 | 0.00% | 100.00% | 100.00% | 1.358962 | 100.00% | 1.00 | 1.5755 |
| 2 | 0.9 | 1.917123 | 100.00% | 100.00% | 100.00% | 1.056518 | 100.00% | 1.00 | 1.5295 |
| 3 | 0.1 | 1.528520 | 100.00% | 100.00% | 100.00% | 0.811404 | 100.00% | 1.00 | 1.4391 |
| 4 | 0.5 | 1.165795 | 100.00% | 100.00% | 100.00% | 0.601582 | 100.00% | 1.00 | 1.2992 |
| 5 | 0.9 | 0.861253 | 100.00% | 100.00% | 100.00% | 0.433222 | 100.00% | 1.00 | 1.1218 |
| 6 | 0.1 | 0.695041 | 100.00% | 100.00% | 100.00% | 0.310781 | 100.00% | 1.00 | 0.9391 |
| 7 | 0.5 | 0.462877 | 100.00% | 100.00% | 100.00% | 0.221100 | 100.00% | 1.00 | 0.7637 |
| 8 | 0.9 | 0.318987 | 100.00% | 100.00% | 100.00% | 0.158888 | 100.00% | 1.00 | 0.6133 |
| 9 | 0.1 | 0.269404 | 100.00% | 100.00% | 100.00% | 0.114138 | 100.00% | 1.00 | 0.4859 |
| 10 | 0.5 | 0.172828 | 100.00% | 100.00% | 100.00% | 0.085297 | 100.00% | 1.00 | 0.3918 |
| 11 | 0.9 | 0.123156 | 100.00% | 100.00% | 100.00% | 0.060827 | 100.00% | 1.00 | 0.3029 |
| 12 | 0.1 | 0.114256 | 100.00% | 100.00% | 100.00% | 0.046091 | 100.00% | 1.00 | 0.2436 |
| 13 | 0.5 | 0.073331 | 100.00% | 100.00% | 100.00% | 0.035507 | 100.00% | 1.00 | 0.1978 |
| 14 | 0.9 | 0.055471 | 100.00% | 100.00% | 100.00% | 0.028382 | 100.00% | 1.00 | 0.1650 |
| 15 | 0.1 | 0.054589 | 100.00% | 100.00% | 100.00% | 0.022695 | 100.00% | 1.00 | 0.1374 |
| 16 | 0.5 | 0.038060 | 100.00% | 100.00% | 100.00% | 0.018957 | 100.00% | 1.00 | 0.1184 |
| 17 | 0.9 | 0.031305 | 100.00% | 100.00% | 100.00% | 0.015992 | 100.00% | 1.00 | 0.1027 |
| 18 | 0.1 | 0.030258 | 100.00% | 100.00% | 100.00% | 0.013904 | 100.00% | 1.00 | 0.0914 |
| 19 | 0.5 | 0.023735 | 100.00% | 100.00% | 100.00% | 0.012061 | 100.00% | 1.00 | 0.0811 |

## Robust Evaluations

| Step | Mean Accuracy | Min-Depth Accuracy | Mean Loss | Top-3 | Target Rank | Entropy | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 19 | 100.00% | 100.00% | 0.012067 | 100.00% | 1.00 | 0.0811 | 96 |

## Latest Light Per-Depth Metrics

| Depth | Accuracy | Loss | Top-3 | Target Rank | Target Prob | Logit Margin | Entropy | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 100.00% | 0.012159 | 100.00% | 1.00 | 0.9879 | 5.6066 | 0.0816 | 8 |
| 0.5 | 100.00% | 0.012036 | 100.00% | 1.00 | 0.9880 | 5.6103 | 0.0809 | 8 |
| 0.9 | 100.00% | 0.011988 | 100.00% | 1.00 | 0.9881 | 5.6139 | 0.0806 | 8 |
