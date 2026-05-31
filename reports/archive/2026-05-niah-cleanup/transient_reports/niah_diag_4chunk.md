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
- best accuracy loss: `0.012126`
- final step train accuracy: `100.00%`
- best step train accuracy: `100.00%`
- passed target accuracy: `True`
- passed success criteria: `True`
- final loss: `0.012126`
- elapsed seconds: `10.695`
- peak allocated memory: `367.72 MB`
- peak reserved memory: `418.00 MB`
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
| 0 | 0.1 | 2.875351 | 0.00% | 0.00% | 0.00% | 1.689521 | 100.00% | 3.00 | 1.5798 |
| 1 | 0.5 | 2.382005 | 0.00% | 100.00% | 100.00% | 1.358352 | 100.00% | 1.00 | 1.5756 |
| 2 | 0.9 | 1.918418 | 100.00% | 100.00% | 100.00% | 1.063955 | 100.00% | 1.00 | 1.5314 |
| 3 | 0.1 | 1.528921 | 100.00% | 100.00% | 100.00% | 0.805195 | 100.00% | 1.00 | 1.4360 |
| 4 | 0.5 | 1.157037 | 100.00% | 100.00% | 100.00% | 0.600082 | 100.00% | 1.00 | 1.2979 |
| 5 | 0.9 | 0.859317 | 100.00% | 100.00% | 100.00% | 0.434390 | 100.00% | 1.00 | 1.1232 |
| 6 | 0.1 | 0.694282 | 100.00% | 100.00% | 100.00% | 0.308861 | 100.00% | 1.00 | 0.9358 |
| 7 | 0.5 | 0.459769 | 100.00% | 100.00% | 100.00% | 0.218573 | 100.00% | 1.00 | 0.7581 |
| 8 | 0.9 | 0.319711 | 100.00% | 100.00% | 100.00% | 0.158576 | 100.00% | 1.00 | 0.6125 |
| 9 | 0.1 | 0.267639 | 100.00% | 100.00% | 100.00% | 0.113611 | 100.00% | 1.00 | 0.4843 |
| 10 | 0.5 | 0.171557 | 100.00% | 100.00% | 100.00% | 0.085446 | 100.00% | 1.00 | 0.3924 |
| 11 | 0.9 | 0.123710 | 100.00% | 100.00% | 100.00% | 0.061567 | 100.00% | 1.00 | 0.3059 |
| 12 | 0.1 | 0.114322 | 100.00% | 100.00% | 100.00% | 0.046981 | 100.00% | 1.00 | 0.2475 |
| 13 | 0.5 | 0.073617 | 100.00% | 100.00% | 100.00% | 0.036316 | 100.00% | 1.00 | 0.2016 |
| 14 | 0.9 | 0.056407 | 100.00% | 100.00% | 100.00% | 0.028864 | 100.00% | 1.00 | 0.1674 |
| 15 | 0.1 | 0.054720 | 100.00% | 100.00% | 100.00% | 0.023119 | 100.00% | 1.00 | 0.1396 |
| 16 | 0.5 | 0.038178 | 100.00% | 100.00% | 100.00% | 0.019209 | 100.00% | 1.00 | 0.1198 |
| 17 | 0.9 | 0.031581 | 100.00% | 100.00% | 100.00% | 0.016157 | 100.00% | 1.00 | 0.1037 |
| 18 | 0.1 | 0.030331 | 100.00% | 100.00% | 100.00% | 0.013956 | 100.00% | 1.00 | 0.0917 |
| 19 | 0.5 | 0.023692 | 100.00% | 100.00% | 100.00% | 0.012115 | 100.00% | 1.00 | 0.0814 |

## Robust Evaluations

| Step | Mean Accuracy | Min-Depth Accuracy | Mean Loss | Top-3 | Target Rank | Entropy | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 19 | 100.00% | 100.00% | 0.012126 | 100.00% | 1.00 | 0.0814 | 96 |

## Latest Light Per-Depth Metrics

| Depth | Accuracy | Loss | Top-3 | Target Rank | Target Prob | Logit Margin | Entropy | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 100.00% | 0.012148 | 100.00% | 1.00 | 0.9879 | 5.6039 | 0.0816 | 8 |
| 0.5 | 100.00% | 0.012106 | 100.00% | 1.00 | 0.9880 | 5.6033 | 0.0813 | 8 |
| 0.9 | 100.00% | 0.012090 | 100.00% | 1.00 | 0.9880 | 5.6052 | 0.0812 | 8 |
