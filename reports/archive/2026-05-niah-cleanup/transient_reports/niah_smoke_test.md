# MHDSRA2 2M NIAH Verification

## Summary

- status: `completed`
- sequence length: `8192`
- final eval source: `robust`
- best eval source: `robust`
- final eval mean accuracy: `0.00%`
- final eval min-depth accuracy: `0.00%`
- best eval mean accuracy: `1.04%`
- best eval min-depth accuracy: `0.00%`
- best accuracy step: `4`
- best accuracy loss: `4.734577`
- final step train accuracy: `0.00%`
- best step train accuracy: `0.00%`
- passed target accuracy: `False`
- passed success criteria: `False`
- re-eval mean accuracy: `0.00%`
- re-eval min-depth accuracy: `0.00%`
- final loss: `4.724636`
- elapsed seconds: `28.268`
- peak allocated memory: `123.17 MB`
- peak reserved memory: `132.00 MB`
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
| robust_eval_batches_per_depth | `32` |
| eval_depths | `[0.1, 0.5, 0.9]` |
| cudnn_benchmark | `False` |

## Observed Steps

| Step | Train Depth | Train Loss | Train Accuracy | Light Mean Accuracy | Light Min-Depth Accuracy | Light Loss | Top-3 | Target Rank | Entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1 | 4.492902 | 0.00% | 0.00% | 0.00% | 4.978768 | 0.00% | 66.33 | 4.4822 |
| 1 | 0.5 | 4.506545 | 0.00% | 0.00% | 0.00% | 5.103531 | 0.00% | 70.00 | 4.4815 |
| 2 | 0.9 | 5.606001 | 0.00% | 0.00% | 0.00% | 4.671755 | 0.00% | 42.67 | 4.4823 |
| 3 | 0.1 | 4.646315 | 0.00% | 0.00% | 0.00% | 5.081620 | 0.00% | 67.67 | 4.4811 |
| 4 | 0.5 | 5.268114 | 0.00% | 0.00% | 0.00% | 5.170623 | 0.00% | 71.67 | 4.4809 |

## Robust Evaluations

| Step | Mean Accuracy | Min-Depth Accuracy | Mean Loss | Top-3 | Target Rank | Entropy | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1.04% | 0.00% | 4.734577 | 5.21% | 50.44 | 4.4810 | 96 |

## Latest Light Per-Depth Metrics

| Depth | Accuracy | Loss | Top-3 | Target Rank | Target Prob | Logit Margin | Entropy | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.00% | 5.460798 | 0.00% | 91.00 | 0.0043 | -1.6729 | 4.4810 | 1 |
| 0.5 | 0.00% | 4.371854 | 0.00% | 29.00 | 0.0126 | -0.5535 | 4.4805 | 1 |
| 0.9 | 0.00% | 5.679217 | 0.00% | 95.00 | 0.0034 | -1.8700 | 4.4812 | 1 |
