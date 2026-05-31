# MHDSRA2 2M NIAH Verification

## Summary

- status: `completed`
- sequence length: `1024`
- final eval source: `light`
- best eval source: `light`
- final eval mean accuracy: `20.83%`
- final eval min-depth accuracy: `12.50%`
- best eval mean accuracy: `25.00%`
- best eval min-depth accuracy: `12.50%`
- best accuracy step: `1`
- best accuracy loss: `4.398318`
- final step train accuracy: `12.50%`
- best step train accuracy: `12.50%`
- passed target accuracy: `False`
- passed success criteria: `False`
- re-eval mean accuracy: `20.83%`
- re-eval min-depth accuracy: `12.50%`
- final loss: `4.430699`
- elapsed seconds: `2.280`
- peak allocated memory: `343.24 MB`
- peak reserved memory: `398.00 MB`
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
| chunk_size | `256` |
| batch_size | `8` |
| epochs | `3` |
| optimizer_steps | `3` |
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
| 0 | 0.1 | 6.892295 | 0.00% | 0.00% | 0.00% | 4.595254 | 20.83% | 40.83 | 4.4867 |
| 1 | 0.5 | 6.731616 | 0.00% | 25.00% | 12.50% | 4.398318 | 29.17% | 36.08 | 4.4796 |
| 2 | 0.9 | 6.967551 | 12.50% | 12.50% | 0.00% | 4.695147 | 12.50% | 53.71 | 4.4699 |

## Latest Light Per-Depth Metrics

| Depth | Accuracy | Loss | Top-3 | Target Rank | Target Prob | Logit Margin | Entropy | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.00% | 4.658136 | 0.00% | 46.25 | 0.0110 | -1.3071 | 4.4719 | 8 |
| 0.5 | 12.50% | 4.890352 | 12.50% | 64.75 | 0.0106 | -1.5816 | 4.4694 | 8 |
| 0.9 | 25.00% | 4.536952 | 25.00% | 50.12 | 0.0153 | -1.2119 | 4.4685 | 8 |
