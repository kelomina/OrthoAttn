# MHDSRA2 2M NIAH Verification

## Summary

- status: `completed`
- sequence length: `65536`
- final eval mean accuracy: `0.00%`
- final eval min-depth accuracy: `0.00%`
- best eval mean accuracy: `8.33%`
- best eval min-depth accuracy: `0.00%`
- best accuracy epoch: `0`
- best accuracy loss: `4.984651`
- final step train accuracy: `0.00%`
- best step train accuracy: `0.00%`
- passed target accuracy: `False`
- passed success criteria: `False`
- re-eval mean accuracy: `0.00%`
- re-eval min-depth accuracy: `0.00%`
- final loss: `4.790822`
- elapsed seconds: `41.342`
- peak allocated memory: `123.17 MB`
- peak reserved memory: `152.00 MB`
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
| learning_rate | `0.001` |
| seed | `20260506` |
| target_accuracy | `1.0` |
| stop_loss | `0.1` |
| eval_interval | `20` |
| eval_batches_per_depth | `4` |
| eval_depths | `[0.1, 0.5, 0.9]` |
| cudnn_benchmark | `False` |

## Observed Epochs

| Epoch | Train Depth | Train Loss | Train Accuracy | Eval Mean Accuracy | Eval Min-Depth Accuracy | Eval Mean Loss |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1 | 4.677691 | 0.00% | 8.33% | 0.00% | 4.984651 |
| 4 | 0.5 | 3.860393 | 0.00% | 8.33% | 0.00% | 4.599930 |
