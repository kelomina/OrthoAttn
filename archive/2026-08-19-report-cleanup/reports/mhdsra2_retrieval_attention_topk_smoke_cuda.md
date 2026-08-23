# mhdsra2_retrieval_attention_topk_smoke_cuda

- device: `cuda:0`
- topks: `None, 16`
- seed: `101`
- epochs: `1`
- retrieval_max_tokens: `128`

| topk | final_acc | best_acc | target_hit | value_top | span_top1 | span_rank | elapsed_sec |
|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.3333 | 0.3333 | 1.0000 | 0.6667 | 0.0000 | 1.0000 | 17.14 |
| 16 | 0.3333 | 0.3333 | 1.0000 | 0.6667 | 0.0000 | 1.0000 | 12.70 |

说明：这是 1 epoch 接入 smoke，只证明配置能跑通 NIAH retrieval/readout 链路，不作为质量结论。
