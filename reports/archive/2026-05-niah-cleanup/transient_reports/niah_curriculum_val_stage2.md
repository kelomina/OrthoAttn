# MHDSRA2 2M NIAH Verification

## Summary

- status: `completed`
- sequence length: `32768`
- final eval source: `robust`
- best eval source: `robust`
- final eval mean accuracy: `4.17%`
- final eval min-depth accuracy: `0.00%`
- best eval mean accuracy: `0.00%`
- best eval min-depth accuracy: `0.00%`
- best accuracy step: `0`
- best accuracy loss: `4.776863`
- final step train accuracy: `12.50%`
- best step train accuracy: `12.50%`
- passed target accuracy: `False`
- passed success criteria: `False`
- re-eval mean accuracy: `4.17%`
- re-eval min-depth accuracy: `0.00%`
- final loss: `4.696999`
- elapsed seconds: `17.466`
- peak allocated memory: `4518.99 MB`
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
| eval_interval | `1` |
| eval_batches_per_depth | `1` |
| light_eval_batches_per_depth | `1` |
| robust_eval_interval | `1` |
| robust_eval_batches_per_depth | `1` |
| eval_depths | `[0.1, 0.5, 0.9]` |
| cudnn_benchmark | `False` |

## Observed Steps

| Step | Train Depth | Train Loss | Train Accuracy | Light Mean Accuracy | Light Min-Depth Accuracy | Light Loss | Top-3 | Target Rank | Entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1 | 4.371642 | 12.50% | 0.00% | 0.00% | 4.740908 | 0.00% | 49.54 | 4.4679 |

## Robust Evaluations

| Step | Mean Accuracy | Min-Depth Accuracy | Mean Loss | Top-3 | Target Rank | Entropy | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00% | 0.00% | 4.776863 | 0.00% | 51.12 | 4.4681 | 24 |

## Latest Light Per-Depth Metrics

| Depth | Accuracy | Loss | Top-3 | Target Rank | Target Prob | Logit Margin | Entropy | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.00% | 4.746080 | 0.00% | 50.25 | 0.0091 | -1.3228 | 4.4679 | 8 |
| 0.5 | 0.00% | 4.580626 | 0.00% | 41.38 | 0.0112 | -1.1594 | 4.4679 | 8 |
| 0.9 | 0.00% | 4.896017 | 0.00% | 57.00 | 0.0090 | -1.4751 | 4.4678 | 8 |
