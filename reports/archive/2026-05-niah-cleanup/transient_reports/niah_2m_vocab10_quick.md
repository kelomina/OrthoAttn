# MHDSRA2 2M NIAH Verification

## Summary

- status: `completed`
- sequence length: `2097152`
- final eval source: `robust`
- best eval source: `robust`
- final eval mean accuracy: `8.33%`
- final eval min-depth accuracy: `0.00%`
- best eval mean accuracy: `16.67%`
- best eval min-depth accuracy: `0.00%`
- best accuracy step: `4`
- best accuracy loss: `2.267008`
- final step train accuracy: `100.00%`
- best step train accuracy: `100.00%`
- passed target accuracy: `False`
- passed success criteria: `False`
- re-eval mean accuracy: `8.33%`
- re-eval min-depth accuracy: `0.00%`
- final loss: `2.249043`
- elapsed seconds: `1835.571`
- peak allocated memory: `151.83 MB`
- peak reserved memory: `184.00 MB`
- parameter count: `50966`
- device: `cuda:0`
- CUDA device: `NVIDIA GeForce RTX 4070 Laptop GPU`
- torch: `2.11.0+cu130`
- torch CUDA: `13.0`

## Config

| Field | Value |
|---|---:|
| vocab_size | `10` |
| dim | `64` |
| num_layers | `2` |
| slots | `64` |
| read_topk | `8` |
| chunk_size | `1024` |
| batch_size | `1` |
| epochs | `5` |
| optimizer_steps | `5` |
| learning_rate | `0.001` |
| seed | `20260506` |
| target_accuracy | `1.0` |
| stop_loss | `0.1` |
| eval_interval | `5` |
| eval_batches_per_depth | `1` |
| light_eval_batches_per_depth | `1` |
| robust_eval_interval | `5` |
| robust_eval_batches_per_depth | `4` |
| eval_depths | `[0.1, 0.5, 0.9]` |
| cudnn_benchmark | `False` |

## Observed Steps

| Step | Train Depth | Train Loss | Train Accuracy | Light Mean Accuracy | Light Min-Depth Accuracy | Light Loss | Top-3 | Target Rank | Entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1 | 3.985739 | 0.00% | 0.00% | 0.00% | 2.775390 | 0.00% | 8.67 | 2.2350 |
| 1 | 0.5 | 3.956476 | 0.00% | 0.00% | 0.00% | 2.197520 | 66.67% | 3.67 | 2.2516 |
| 2 | 0.9 | 3.732491 | 0.00% | 0.00% | 0.00% | 2.251313 | 33.33% | 5.00 | 2.2559 |
| 3 | 0.1 | 3.928461 | 0.00% | 33.33% | 0.00% | 2.206536 | 33.33% | 4.67 | 2.2584 |
| 4 | 0.5 | 3.056499 | 100.00% | 0.00% | 0.00% | 2.185839 | 66.67% | 3.33 | 2.2507 |

## Robust Evaluations

| Step | Mean Accuracy | Min-Depth Accuracy | Mean Loss | Top-3 | Target Rank | Entropy | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 16.67% | 0.00% | 2.267008 | 33.33% | 5.00 | 2.2485 | 12 |

## Latest Light Per-Depth Metrics

| Depth | Accuracy | Loss | Top-3 | Target Rank | Target Prob | Logit Margin | Entropy | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.00% | 2.093590 | 100.00% | 2.00 | 0.1232 | -0.3601 | 2.2484 | 1 |
| 0.5 | 0.00% | 2.075708 | 100.00% | 2.00 | 0.1255 | -0.2988 | 2.2561 | 1 |
| 0.9 | 0.00% | 2.388218 | 0.00% | 6.00 | 0.0918 | -0.6468 | 2.2476 | 1 |
