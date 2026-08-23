# MHDSRA2 2M Larger-Scale NIAH Benchmark

## Summary

- status: `completed`
- sequence length: `2097152`
- best accuracy: `100.00%`
- passed target accuracy: `True`
- final loss: `2.540358`
- elapsed seconds: `113.063`
- peak allocated memory: `188.82 MB`
- peak reserved memory: `218.00 MB`
- parameter count: `420476`
- device: `cuda:0`
- CUDA device: `NVIDIA GeForce RTX 4070 Laptop GPU`
- torch: `2.11.0+cu130`
- torch CUDA: `13.0`

## Config

| Field | Value |
|---|---:|
| vocab_size | `100` |
| dim | `128` |
| num_layers | `4` |
| slots | `128` |
| read_topk | `16` |
| chunk_size | `1024` |
| batch_size | `1` |
| epochs | `3` |
| learning_rate | `0.001` |
| seed | `20260506` |
| target_accuracy | `1.0` |
| stop_loss | `0.0` |

## Observed Epochs

| Epoch | Depth | Loss | Accuracy |
|---:|---:|---:|---:|
| 0 | 0.1 | 4.126402 | 0.00% |
| 1 | 0.1 | 3.287579 | 100.00% |
| 2 | 0.9 | 2.540358 | 100.00% |
