# MHDSRA2 2M NIAH Verification

## Summary

- status: `completed`
- sequence length: `8192`
- final eval source: `robust`
- best eval source: `robust`
- final eval mean accuracy: `0.00%`
- final eval min-depth accuracy: `0.00%`
- best eval mean accuracy: `4.17%`
- best eval min-depth accuracy: `0.00%`
- best accuracy step: `1`
- best accuracy loss: `4.660746`
- final step train accuracy: `0.00%`
- best step train accuracy: `0.00%`
- passed target accuracy: `False`
- passed success criteria: `False`
- re-eval mean accuracy: `0.00%`
- re-eval min-depth accuracy: `0.00%`
- final loss: `4.694542`
- elapsed seconds: `7.358`
- peak allocated memory: `1504.55 MB`
- peak reserved memory: `1630.00 MB`
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
| robust_eval_batches_per_depth | `1` |
| eval_depths | `[0.1, 0.5, 0.9]` |
| cudnn_benchmark | `False` |

## Observed Steps

| Step | Train Depth | Train Loss | Train Accuracy | Light Mean Accuracy | Light Min-Depth Accuracy | Light Loss | Top-3 | Target Rank | Entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1 | 4.771426 | 0.00% | 0.00% | 0.00% | 4.773574 | 0.00% | 52.04 | 4.4795 |
| 1 | 0.5 | 4.434941 | 0.00% | 0.00% | 0.00% | 4.783760 | 4.17% | 53.04 | 4.4748 |

## Robust Evaluations

| Step | Mean Accuracy | Min-Depth Accuracy | Mean Loss | Top-3 | Target Rank | Entropy | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.17% | 0.00% | 4.660746 | 4.17% | 46.79 | 4.4748 | 24 |

## Latest Light Per-Depth Metrics

| Depth | Accuracy | Loss | Top-3 | Target Rank | Target Prob | Logit Margin | Entropy | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.00% | 4.961031 | 0.00% | 62.75 | 0.0077 | -1.3854 | 4.4755 | 8 |
| 0.5 | 0.00% | 4.743807 | 0.00% | 50.75 | 0.0101 | -1.1890 | 4.4746 | 8 |
| 0.9 | 0.00% | 4.646441 | 12.50% | 45.62 | 0.0109 | -1.1078 | 4.4742 | 8 |
