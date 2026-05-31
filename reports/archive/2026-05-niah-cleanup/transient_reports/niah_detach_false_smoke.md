# MHDSRA2 2M NIAH Verification

## Summary

- status: `completed`
- sequence length: `8192`
- final eval source: `robust`
- best eval source: `robust`
- final eval mean accuracy: `1.04%`
- final eval min-depth accuracy: `0.00%`
- best eval mean accuracy: `2.08%`
- best eval min-depth accuracy: `1.56%`
- best accuracy step: `2`
- best accuracy loss: `4.740303`
- final step train accuracy: `0.00%`
- best step train accuracy: `0.00%`
- passed target accuracy: `False`
- passed success criteria: `False`
- re-eval mean accuracy: `1.04%`
- re-eval min-depth accuracy: `0.00%`
- final loss: `4.744354`
- elapsed seconds: `39.716`
- peak allocated memory: `735.08 MB`
- peak reserved memory: `844.00 MB`
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
| epochs | `3` |
| optimizer_steps | `3` |
| learning_rate | `0.001` |
| seed | `20260506` |
| target_accuracy | `1.0` |
| stop_loss | `0.1` |
| eval_interval | `3` |
| eval_batches_per_depth | `1` |
| light_eval_batches_per_depth | `1` |
| robust_eval_interval | `3` |
| robust_eval_batches_per_depth | `32` |
| eval_depths | `[0.1, 0.5, 0.9]` |
| cudnn_benchmark | `False` |

## Observed Steps

| Step | Train Depth | Train Loss | Train Accuracy | Light Mean Accuracy | Light Min-Depth Accuracy | Light Loss | Top-3 | Target Rank | Entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1 | 4.895292 | 0.00% | 0.00% | 0.00% | 4.678303 | 0.00% | 47.17 | 4.4815 |
| 1 | 0.5 | 4.864531 | 0.00% | 8.33% | 0.00% | 4.472326 | 16.67% | 38.33 | 4.4803 |
| 2 | 0.9 | 4.739095 | 0.00% | 0.00% | 0.00% | 4.898166 | 0.00% | 61.42 | 4.4791 |

## Robust Evaluations

| Step | Mean Accuracy | Min-Depth Accuracy | Mean Loss | Top-3 | Target Rank | Entropy | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2.08% | 1.56% | 4.740303 | 4.69% | 50.72 | 4.4790 | 384 |

## Latest Light Per-Depth Metrics

| Depth | Accuracy | Loss | Top-3 | Target Rank | Target Prob | Logit Margin | Entropy | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.00% | 5.047785 | 0.00% | 73.25 | 0.0066 | -1.5874 | 4.4800 | 4 |
| 0.5 | 0.00% | 4.918134 | 0.00% | 58.75 | 0.0085 | -1.4973 | 4.4789 | 4 |
| 0.9 | 0.00% | 4.728580 | 0.00% | 52.25 | 0.0089 | -1.3025 | 4.4785 | 4 |
