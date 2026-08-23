# Milestone 2: Code Modification Summary

## Overview
Implemented Milestone 2 deliverables: Standard Causal Transformer Baseline, Ground Truth Oracle Probe integration, and unified Multi-Model MQAR Benchmark Runner in `scripts/benchmark_mqar.py`, with structured benchmark results exported to `reports/mqar_benchmark_results.json`.

---

## 1. File Modifications

### `scripts/benchmark_mqar.py`
- **Standard Causal Transformer Baseline (`StandardCausalTransformer` / `StandardAttentionLM`)**:
  - `RotaryPositionalEmbedding`: Multi-head RoPE rotary frequency module computing cos/sin projections for $d_{\text{head}}$.
  - `apply_rotary_pos_emb`: Applies rotary rotation to queries and keys in head dimension.
  - `CausalSelfAttentionBlock`: PyTorch `F.scaled_dot_product_attention(..., is_causal=True)` multi-head self-attention with RoPE.
  - `StandardTransformerBlock`: Pre-LayerNorm decoder block with attention and GELU MLP.
  - `StandardCausalTransformer`: Complete causal language model baseline with token embedding, Pre-LN blocks, final LayerNorm, and vocabulary projection.
  - Exported historical alias `StandardAttentionLM = StandardCausalTransformer` for backwards compatibility.
- **Ground Truth Oracle Integration (`MQAROracleModel`)**:
  - Seamlessly integrates `MQAROracleModel` from `src.dsra.domain.mqar` for instant causal lookup verification (100.0% accuracy, 0.0 loss).
- **Multi-Model Benchmark CLI & Training Loop**:
  - Added `--model` argument supporting choices: `transformer`, `oracle`, `mhdsra2`, `all`.
  - Added `get_cosine_warmup_scheduler` implementing linear warmup (e.g. 50 steps) + cosine annealing decay.
  - Integrated AdamW optimizer (`lr=1e-3`, `weight_decay=1e-4`, `betas=(0.9, 0.98)`), gradient clipping (`clip_grad_norm_=1.0`), and periodic evaluation via `evaluate_mqar`.
  - Added early stopping threshold (`early_stop_acc=0.999`, `min_steps=50`).
  - Added explicit CUDA device placement (`cuda:0` with CPU fallback) and `torch.cuda.empty_cache()` / `gc.collect()` after each run.
- **Structured JSON & Markdown Output**:
  - Primary target: `reports/mqar_benchmark_results.json`.
  - Secondary targets: `docs/reports/verify_technical_report/mqar/mqar_benchmark_results.json`, `docs/reports/verify_technical_report/mqar/mqar_benchmark_results.md`, `docs/figures/verify_technical_report/fig_mqar_benchmark.png`.
- **Documentation & Standards**:
  - Added full bilingual (Chinese & English) docstrings adhering to `AGENTS.md` (Calling party, Callee, Action, Args, Returns, Errors, Side effects).
  - All `ruff` checks passed cleanly.

### `reports/mqar_benchmark_results.json`
- Generated structured JSON benchmark results covering all three model architectures across the standard MQAR benchmark grid ($L=512, K=4$; $L=1024, K=8$).
