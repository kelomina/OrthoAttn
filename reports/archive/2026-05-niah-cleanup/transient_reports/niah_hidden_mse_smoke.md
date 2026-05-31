# MHDSRA2 2M NIAH Verification

## Summary

- status: `completed`
- sequence length: `1024`
- final eval source: `robust`
- best eval source: `robust`
- final eval mean accuracy: `1.04%`
- final eval min-depth accuracy: `0.00%`
- best eval mean accuracy: `1.04%`
- best eval min-depth accuracy: `0.00%`
- best accuracy step: `9`
- best accuracy loss: `4.796744`
- final step train accuracy: `0.00%`
- best step train accuracy: `12.50%`
- passed target accuracy: `False`
- passed success criteria: `False`
- re-eval mean accuracy: `1.04%`
- re-eval min-depth accuracy: `0.00%`
- final loss: `4.723534`
- elapsed seconds: `14.593`
- peak allocated memory: `348.89 MB`
- peak reserved memory: `400.00 MB`
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
| epochs | `10` |
| optimizer_steps | `10` |
| learning_rate | `0.001` |
| seed | `20260506` |
| target_accuracy | `1.0` |
| stop_loss | `0.1` |
| eval_interval | `10` |
| eval_batches_per_depth | `1` |
| light_eval_batches_per_depth | `1` |
| robust_eval_interval | `10` |
| robust_eval_batches_per_depth | `4` |
| eval_depths | `[0.1, 0.5, 0.9]` |
| cudnn_benchmark | `False` |

## Observed Steps

| Step | Train Depth | Train Loss | Train Accuracy | Light Mean Accuracy | Light Min-Depth Accuracy | Light Loss | Top-3 | Target Rank | Entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1 | 7.596416 | 0.00% | 0.00% | 0.00% | 4.752448 | 0.00% | 51.54 | 4.4816 |
| 1 | 0.5 | 7.845876 | 0.00% | 0.00% | 0.00% | 4.819749 | 4.17% | 54.42 | 4.4816 |
| 2 | 0.9 | 7.540827 | 0.00% | 0.00% | 0.00% | 4.810929 | 4.17% | 54.50 | 4.4810 |
| 3 | 0.1 | 7.393407 | 0.00% | 4.17% | 0.00% | 4.548433 | 4.17% | 40.04 | 4.4785 |
| 4 | 0.5 | 7.707782 | 12.50% | 0.00% | 0.00% | 4.884825 | 0.00% | 55.92 | 4.4752 |
| 5 | 0.9 | 7.782372 | 0.00% | 0.00% | 0.00% | 4.683066 | 0.00% | 46.46 | 4.4733 |
| 6 | 0.1 | 7.632689 | 0.00% | 0.00% | 0.00% | 4.548169 | 4.17% | 40.00 | 4.4705 |
| 7 | 0.5 | 7.825584 | 0.00% | 0.00% | 0.00% | 4.841584 | 4.17% | 56.12 | 4.4684 |
| 8 | 0.9 | 7.734477 | 0.00% | 4.17% | 0.00% | 4.525926 | 4.17% | 37.67 | 4.4685 |
| 9 | 0.1 | 7.421311 | 0.00% | 0.00% | 0.00% | 4.842300 | 0.00% | 55.92 | 4.4654 |

## Robust Evaluations

| Step | Mean Accuracy | Min-Depth Accuracy | Mean Loss | Top-3 | Target Rank | Entropy | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | 1.04% | 0.00% | 4.796744 | 3.12% | 51.40 | 4.4663 | 96 |

## Latest Light Per-Depth Metrics

| Depth | Accuracy | Loss | Top-3 | Target Rank | Target Prob | Logit Margin | Entropy | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.00% | 4.806946 | 0.00% | 55.00 | 0.0091 | -1.4577 | 4.4682 | 8 |
| 0.5 | 0.00% | 4.874229 | 0.00% | 56.12 | 0.0089 | -1.5507 | 4.4646 | 8 |
| 0.9 | 0.00% | 4.845726 | 0.00% | 56.62 | 0.0087 | -1.5489 | 4.4635 | 8 |
