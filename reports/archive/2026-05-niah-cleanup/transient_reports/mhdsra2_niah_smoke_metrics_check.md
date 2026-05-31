# MHDSRA2 2M NIAH Verification

## Summary

- status: `completed`
- sequence length: `1024`
- final eval source: `robust`
- best eval source: `robust`
- final eval mean accuracy: `0.00%`
- final eval min-depth accuracy: `0.00%`
- best eval mean accuracy: `0.00%`
- best eval min-depth accuracy: `0.00%`
- best accuracy step: `1`
- best accuracy loss: `4.273588`
- final step train accuracy: `0.00%`
- best step train accuracy: `0.00%`
- passed target accuracy: `False`
- passed success criteria: `False`
- re-eval mean accuracy: `0.00%`
- re-eval min-depth accuracy: `0.00%`
- final loss: `4.254636`
- elapsed seconds: `0.549`
- peak allocated memory: `0.00 MB`
- peak reserved memory: `0.00 MB`
- parameter count: `62576`
- device: `cpu`
- CUDA device: `None`
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
| 0 | 0.1 | 4.859885 | 0.00% | 0.00% | 0.00% | 5.887185 | 0.00% | 99.00 | 4.4850 |
| 1 | 0.5 | 4.405683 | 0.00% | 0.00% | 0.00% | 4.981743 | 0.00% | 68.00 | 4.4857 |

## Robust Evaluations

| Step | Mean Accuracy | Min-Depth Accuracy | Mean Loss | Top-3 | Target Rank | Entropy | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00% | 0.00% | 4.273588 | 0.00% | 23.00 | 4.4856 | 3 |

## Latest Light Per-Depth Metrics

| Depth | Accuracy | Loss | Top-3 | Target Rank | Target Prob | Logit Margin | Entropy | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.00% | 5.322348 | 0.00% | 84.00 | 0.0049 | -1.5609 | 4.4856 | 1 |
| 0.5 | 0.00% | 4.677542 | 0.00% | 51.00 | 0.0093 | -0.9104 | 4.4859 | 1 |
| 0.9 | 0.00% | 4.945340 | 0.00% | 69.00 | 0.0071 | -1.1862 | 4.4855 | 1 |
