# MHDSRA2 Practical Effectiveness Report - 2026-06-30

## 结论

不能诚实地说 MHDSRA2 现在已经“实战有效”。更准确的结论是：本轮把它从“只有候选协议有效”推进到了“有一个可复现的实战入口信号”，但还没有达到完整实战验收。

具体说：在 WikiText 字符级 tiny LM 上，`seq_len=1024`、300 步、`mhdsra2_chunk_size=1024` 时，MHDSRA2 验证 PPL 为 `12.61`，标准注意力为 `12.67`，质量接近且略好；但训练时间 `52s` 对 `21s`，仍慢约 `2.5x`。在 raw JSON 原生生成上，历史多 seed 报告仍显示 exact match 为 `0`，真正答题仍依赖默认关闭的结构化 `extract_then_compose` readout。

所以本轮的可站住结论不是“已经实战有效”，而是：`mhdsra2_chunk_size` 是一个实际有效的工程调参入口，能把 tiny LM 长上下文训练从过慢状态推进到可继续验证；模型质量在一个短训真实文本入口上没有崩，但 raw generation 和效率还没达标。

## 已执行结果

### Tiny LM 正式 CLI 对照

命令：

```powershell
python scripts\tiny_llama_compare.py --seq-len 1024 --batch-size 1 --max-steps 300 --dim 128 --heads 4 --layers 2 --mhdsra2-chunk-size 1024 --device cuda:0
```

| model | validation PPL | time | 说明 |
|---|---:|---:|---|
| Standard Attention | 12.67 | 21s | tiny LLaMA RoPE baseline |
| MHDSRA2 | 12.61 | 52s | `mhdsra2_chunk_size=1024` |

判断：PPL 质量接近，MHDSRA2 略好；但训练时间仍慢约 `2.5x`。这是项目代理指标 smoke，不是 50,000 step 正式 PPL。

### 反例：默认 chunk/短上下文不达标

命令：

```powershell
python scripts\tiny_llama_compare.py --seq-len 256 --batch-size 4 --max-steps 1000 --dim 128 --heads 4 --layers 2 --device cuda:0
```

| model | validation PPL | time |
|---|---:|---:|
| Standard Attention | 9.18 | 34s |
| MHDSRA2 | 11.62 | 384s |

判断：在这个口径下，MHDSRA2 不实战有效。

### Chunk Size 消融

同 seed、同 `seq_len=1024`、同 100 步，只改变 MHDSRA2 chunk：

| chunk | validation PPL | wall time |
|---:|---:|---:|
| 128 | 15.94 | 301.8s |
| 1024 | 16.02 | 36.0s |

判断：调大 chunk 带来约 `8.38x` 速度提升，PPL 只差 `0.08`。这是本轮最实际的推进点。

## 历史参考结果

JSON retrieval 的历史 CUDA 多 seed 报告显示：不使用结构化 readout 时，raw generation exact match 仍为 `0`；使用 `extract_then_compose` 后能达到高 exact match。这个结果只能说明“模型预测 evidence window + 结构化读出器”这条路径有效，不能说明模型本体已经能原生生成完整正确答案。

## 本轮代码修改

- `scripts/tiny_llama_shared.py`：增加 `mhdsra2_chunk_size` 与 `seed` 配置，新增 `set_reproducible_seed()`。
- `scripts/tiny_llama_baseline.py` / `scripts/tiny_llama_mhdsra2.py`：训练入口调用固定 seed。
- `scripts/tiny_llama_mhdsra2.py`：`main_mhdsra2()` 读取 `mhdsra2_chunk_size`。
- `scripts/tiny_llama_compare.py`：增加 `--mhdsra2-chunk-size` 与 `--seed`，并把 Windows 终端不兼容的非 ASCII 状态符改成普通文本。
- `tests/test_tiny_llama_ppl_regressions.py`：增加 CLI 参数和 seed helper 回归测试。

## 验证命令

```powershell
python -m py_compile scripts\tiny_llama_compare.py scripts\tiny_llama_mhdsra2.py scripts\tiny_llama_baseline.py scripts\tiny_llama_shared.py tests\test_tiny_llama_ppl_regressions.py
# passed

python -m pytest tests\test_tiny_llama_ppl_regressions.py -q
# 6 passed
```


## 继续验证补充 - 2026-06-30

用户要求继续验证后，本轮先检查了当前 GPU 状态：`cuda:0` 显存约 `7508 / 8188 MB` 已被其它项目占用，GPU 利用率 `100%`。在这个状态下继续跑新的 CUDA 训练会污染耗时结论，因此没有把新的多 seed CUDA 训练伪装成已完成结果。

为避免再次出现长脚本超时后没有任何产物，本轮新增了可恢复 runner：

```powershell
python scripts\tiny_lm_practical_multiseed.py --seeds 1234,2025,3036 --seq-len 1024 --batch-size 1 --max-steps 300 --dim 128 --heads 4 --layers 2 --mhdsra2-chunk-size 1024 --device cuda:0 --resume
```

该 runner 会逐 seed 调用正式 `tiny_llama_compare.py`，每个 seed 完成后立即写入：

- `reports/mhdsra2_practical_tiny_lm_multiseed.json`
- `reports/mhdsra2_practical_tiny_lm_multiseed.md`

本轮已验证 runner 的解析、汇总、失败行处理和 timeout 容错逻辑。CPU timeout smoke 已生成 `reports/mhdsra2_practical_tiny_lm_runner_timeout_smoke.json/.md`，证明某个 seed 超时也会留下报告；但未在 GPU 满载状态下继续执行多 seed 训练。

## 未完成项

- 3-seed `seq_len=1024` / 300-step 汇总脚本超过 20 分钟上限并被终止；没有生成结果，不作为完成证据。
- 未执行 50,000 step tiny LM 正式 PPL。
- 未执行 LongBench、RULER、MMLU 等外部 LLM 评估。

## 下一步建议

1. 用 `--mhdsra2-chunk-size 1024` 固定为 tiny LM 长上下文实验默认候选，再跑 3 seed / 1000 step。
2. raw JSON generation 不要继续靠 `extract_then_compose` 计分；下一轮应针对原生生成失败做错误分析。
3. 若目标是工程实战，需要优先把 MHDSRA2 chunk 循环和状态更新做 CUDA 友好的批处理/融合，否则质量接近也很难产品化。
