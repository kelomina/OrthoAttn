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
- best accuracy loss: `0.012112`
- final step train accuracy: `100.00%`
- best step train accuracy: `100.00%`
- passed target accuracy: `True`
- passed success criteria: `True`
- final loss: `0.012112`
- elapsed seconds: `9.970`
- peak allocated memory: `244.43 MB`
- peak reserved memory: `278.00 MB`
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
| 0 | 0.1 | 2.876474 | 0.00% | 0.00% | 0.00% | 1.702701 | 100.00% | 3.00 | 1.5790 |
| 1 | 0.5 | 2.399204 | 0.00% | 91.67% | 75.00% | 1.378889 | 100.00% | 1.08 | 1.5758 |
| 2 | 0.9 | 1.934627 | 100.00% | 100.00% | 100.00% | 1.081574 | 100.00% | 1.00 | 1.5343 |
| 3 | 0.1 | 1.553577 | 100.00% | 100.00% | 100.00% | 0.819099 | 100.00% | 1.00 | 1.4416 |
| 4 | 0.5 | 1.180579 | 100.00% | 100.00% | 100.00% | 0.620192 | 100.00% | 1.00 | 1.3137 |
| 5 | 0.9 | 0.884768 | 100.00% | 100.00% | 100.00% | 0.456561 | 100.00% | 1.00 | 1.1505 |
| 6 | 0.1 | 0.711757 | 100.00% | 100.00% | 100.00% | 0.326383 | 100.00% | 1.00 | 0.9655 |
| 7 | 0.5 | 0.482360 | 100.00% | 100.00% | 100.00% | 0.234013 | 100.00% | 1.00 | 0.7915 |
| 8 | 0.9 | 0.338729 | 100.00% | 100.00% | 100.00% | 0.165617 | 100.00% | 1.00 | 0.6309 |
| 9 | 0.1 | 0.276102 | 100.00% | 100.00% | 100.00% | 0.118214 | 100.00% | 1.00 | 0.4982 |
| 10 | 0.5 | 0.181421 | 100.00% | 100.00% | 100.00% | 0.088278 | 100.00% | 1.00 | 0.4020 |
| 11 | 0.9 | 0.129270 | 100.00% | 100.00% | 100.00% | 0.063771 | 100.00% | 1.00 | 0.3142 |
| 12 | 0.1 | 0.117090 | 100.00% | 100.00% | 100.00% | 0.048090 | 100.00% | 1.00 | 0.2520 |
| 13 | 0.5 | 0.075569 | 100.00% | 100.00% | 100.00% | 0.036513 | 100.00% | 1.00 | 0.2023 |
| 14 | 0.9 | 0.057102 | 100.00% | 100.00% | 100.00% | 0.028788 | 100.00% | 1.00 | 0.1669 |
| 15 | 0.1 | 0.055429 | 100.00% | 100.00% | 100.00% | 0.022891 | 100.00% | 1.00 | 0.1383 |
| 16 | 0.5 | 0.038623 | 100.00% | 100.00% | 100.00% | 0.019020 | 100.00% | 1.00 | 0.1186 |
| 17 | 0.9 | 0.031444 | 100.00% | 100.00% | 100.00% | 0.016030 | 100.00% | 1.00 | 0.1029 |
| 18 | 0.1 | 0.030498 | 100.00% | 100.00% | 100.00% | 0.013919 | 100.00% | 1.00 | 0.0914 |
| 19 | 0.5 | 0.024017 | 100.00% | 100.00% | 100.00% | 0.012100 | 100.00% | 1.00 | 0.0812 |

## Robust Evaluations

| Step | Mean Accuracy | Min-Depth Accuracy | Mean Loss | Top-3 | Target Rank | Entropy | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 19 | 100.00% | 100.00% | 0.012112 | 100.00% | 1.00 | 0.0813 | 96 |

## Latest Light Per-Depth Metrics

| Depth | Accuracy | Loss | Top-3 | Target Rank | Target Prob | Logit Margin | Entropy | Samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 100.00% | 0.012167 | 100.00% | 1.00 | 0.9879 | 5.5973 | 0.0816 | 8 |
| 0.5 | 100.00% | 0.012106 | 100.00% | 1.00 | 0.9880 | 5.6041 | 0.0813 | 8 |
| 0.9 | 100.00% | 0.012028 | 100.00% | 1.00 | 0.9880 | 5.6104 | 0.0808 | 8 |
