# MHDSRA2 2M NIAH Verification

## Summary

- status: `success`
- sequence length: `128`
- final eval source: `robust`
- best eval source: `robust`
- final eval mean accuracy: `100.00%`
- final eval min-depth accuracy: `100.00%`
- best eval mean accuracy: `100.00%`
- best eval min-depth accuracy: `100.00%`
- best accuracy step: `19`
- best accuracy loss: `0.012990`
- final step train accuracy: `100.00%`
- best step train accuracy: `100.00%`
- passed target accuracy: `True`
- passed success criteria: `True`
- final loss: `0.012990`
- elapsed seconds: `2.737`
- peak allocated memory: `86.05 MB`
- peak reserved memory: `112.00 MB`
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
| 0 | 0.1 | 2.878826 | 0.00% | 0.00% | 0.00% | 1.727412 | 100.00% | 3.00 | 1.5789 |
| 1 | 0.5 | 2.413667 | 0.00% | 91.67% | 75.00% | 1.414361 | 100.00% | 1.08 | 1.5788 |
| 2 | 0.9 | 1.977065 | 100.00% | 100.00% | 100.00% | 1.129559 | 100.00% | 1.00 | 1.5454 |
| 3 | 0.1 | 1.606800 | 100.00% | 100.00% | 100.00% | 0.879838 | 100.00% | 1.00 | 1.4703 |
| 4 | 0.5 | 1.248685 | 100.00% | 100.00% | 100.00% | 0.675758 | 100.00% | 1.00 | 1.3568 |
| 5 | 0.9 | 0.960377 | 100.00% | 100.00% | 100.00% | 0.503450 | 100.00% | 1.00 | 1.2049 |
| 6 | 0.1 | 0.742031 | 100.00% | 100.00% | 100.00% | 0.367611 | 100.00% | 1.00 | 1.0313 |
| 7 | 0.5 | 0.540643 | 100.00% | 100.00% | 100.00% | 0.266618 | 100.00% | 1.00 | 0.8582 |
| 8 | 0.9 | 0.396727 | 100.00% | 100.00% | 100.00% | 0.190485 | 100.00% | 1.00 | 0.6935 |
| 9 | 0.1 | 0.296736 | 100.00% | 100.00% | 100.00% | 0.137040 | 100.00% | 1.00 | 0.5538 |
| 10 | 0.5 | 0.214192 | 100.00% | 100.00% | 100.00% | 0.098738 | 100.00% | 1.00 | 0.4372 |
| 11 | 0.9 | 0.157691 | 100.00% | 100.00% | 100.00% | 0.072467 | 100.00% | 1.00 | 0.3466 |
| 12 | 0.1 | 0.122607 | 100.00% | 100.00% | 100.00% | 0.053933 | 100.00% | 1.00 | 0.2759 |
| 13 | 0.5 | 0.089905 | 100.00% | 100.00% | 100.00% | 0.041190 | 100.00% | 1.00 | 0.2229 |
| 14 | 0.9 | 0.069393 | 100.00% | 100.00% | 100.00% | 0.032061 | 100.00% | 1.00 | 0.1823 |
| 15 | 0.1 | 0.055910 | 100.00% | 100.00% | 100.00% | 0.025679 | 100.00% | 1.00 | 0.1521 |
| 16 | 0.5 | 0.044493 | 100.00% | 100.00% | 100.00% | 0.021012 | 100.00% | 1.00 | 0.1290 |
| 17 | 0.9 | 0.036638 | 100.00% | 100.00% | 100.00% | 0.017569 | 100.00% | 1.00 | 0.1112 |
| 18 | 0.1 | 0.031198 | 100.00% | 100.00% | 100.00% | 0.015005 | 100.00% | 1.00 | 0.0974 |
| 19 | 0.5 | 0.026390 | 100.00% | 100.00% | 100.00% | 0.012992 | 100.00% | 1.00 | 0.0863 |

## Robust Evaluations

| Step | Mean Accuracy | Min-Depth Accuracy | Mean Loss | Top-3 | Target Rank | Entropy | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 19 | 100.00% | 100.00% | 0.012990 | 100.00% | 1.00 | 0.0863 | 96 |

## Latest Light Per-Depth Metrics

| Depth | Accuracy | Loss | Top-3 | Target Rank | Target Prob | Logit Margin | Entropy | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 100.00% | 0.013019 | 100.00% | 1.00 | 0.9871 | 5.5216 | 0.0865 | 8 |
| 0.5 | 100.00% | 0.012985 | 100.00% | 1.00 | 0.9871 | 5.5256 | 0.0863 | 8 |
| 0.9 | 100.00% | 0.012972 | 100.00% | 1.00 | 0.9871 | 5.5270 | 0.0862 | 8 |
