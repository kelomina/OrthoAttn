"""WikiText-103 PPL evaluation for DSRA vs Standard Attention.

中文说明：
本脚本在完整 WikiText-103 数据集上评测 5 个模型变体的困惑度（PPL）。
每个变体训练 200K steps，定期在 valid 集评测，最终在 test 集评测。
处理 OOM 自动降级（A/D 变体 batch=4 -> 2），保存结果 JSON + Markdown 报告。

English:
This script evaluates 5 model variants on the full WikiText-103 dataset.
Each variant trains for 200K steps with periodic valid evaluation and final test PPL.
Handles OOM by halving batch_size automatically.
Saves result JSON and Markdown report to reports/.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---- 路径设置 ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---- 导入 ----
from config.experiment_config import LMExperimentConfig
from scripts.tiny_llama_baseline import StandardAttentionLM
from scripts.tiny_llama_mhdsra2 import (
    MHDSRA2WithFFN,
    evaluate_ppl as eval_ppl,
)
from scripts.tiny_llama_shared import (
    CharTokenizer,
    create_dataloader,
    create_eval_loader,
    download_wikitext103,
    load_text,
    resolve_device,
)
from src.dsra.report_utils import ensure_reports_dir
from src.dsra.swanlab_utils import init_swanlab


# ============================================================
# 变体定义
# ============================================================

VARIANT_CONFIGS: list[dict[str, Any]] = [
    {
        "label": "A",
        "name": "Standard Attention (baseline)",
        "model_type": "standard",
        "dim": 512, "heads": 4, "num_layers": 6,
        "batch_size": 4,
    },
    {
        "label": "B",
        "name": "MHDSRA2 (default)",
        "model_type": "mhdsra2",
        "dim": 512, "heads": 4, "num_layers": 6,
        "slots": 128, "chunk_size": 128,
        "batch_size": 8,
    },
    {
        "label": "C",
        "name": "MHDSRA2 (large slots)",
        "model_type": "mhdsra2",
        "dim": 512, "heads": 4, "num_layers": 6,
        "slots": 256, "chunk_size": 128,
        "batch_size": 8,
    },
    {
        "label": "D",
        "name": "MHDSRA2 (deep, 12 layers)",
        "model_type": "mhdsra2",
        "dim": 512, "heads": 4, "num_layers": 12,
        "slots": 128, "chunk_size": 128,
        "batch_size": 4,
    },
    {
        "label": "E",
        "name": "MHDSRA2 (small chunk)",
        "model_type": "mhdsra2",
        "dim": 512, "heads": 4, "num_layers": 6,
        "slots": 64, "chunk_size": 64,
        "batch_size": 8,
    },
]


def build_model(variant: dict[str, Any], vocab_size: int, device: torch.device) -> nn.Module:
    """根据变体配置构建模型并移至目标设备。

    中文说明:
    - 调用方 / Called by: ``train_and_evaluate``
    - 调用对象 / Calls: ``StandardAttentionLM``, ``MHDSRA2WithFFN``
    - 作用 / Purpose: 根据 ``variant["model_type"]`` 分发到对应模型类，构造后立即 .to(device)
    - 错误处理 / Error handling: ``model_type`` 非法时抛出 ``ValueError``
    - 副作用 / Side effects: 模型参数被移动到 device（可能为 CUDA）
    - 关键词 / Keywords: build, model, variant, standard, mhdsra2, device, 构建模型, 变体

    Function name:
        build_model
    Purpose:
        Build a model from variant config and move it to the target device.
    Called by:
        ``train_and_evaluate``
    Calls:
        ``StandardAttentionLM``, ``MHDSRA2WithFFN``
    Parameters:
        - variant: dict with keys like "model_type", "dim", "heads", "num_layers", etc.
        - vocab_size: vocabulary size for the embedding layer
        - device: torch device to place the model on
    Returns:
        nn.Module instance on the specified device
    Error handling:
        Raises ValueError for unknown model_type.
    English keywords:
        build, model, variant, standard, mhdsra2, device
    """
    model_type = variant["model_type"]
    if model_type == "standard":
        model = StandardAttentionLM(
            vocab_size=vocab_size,
            dim=variant["dim"],
            heads=variant["heads"],
            num_layers=variant["num_layers"],
        )
    elif model_type == "mhdsra2":
        model = MHDSRA2WithFFN(
            vocab_size=vocab_size,
            dim=variant["dim"],
            heads=variant["heads"],
            num_layers=variant["num_layers"],
            slots=variant.get("slots", 128),
            chunk_size=variant.get("chunk_size", 128),
            mhdsra2_config_override=None,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """统计模型可训练参数量。

    中文说明:
    - 调用方 / Called by: ``train_and_evaluate``
    - 调用对象 / Calls: ``model.parameters()``
    - 作用 / Purpose: 返回模型中所有 requires_grad 参数的数值总和
    - 关键词 / Keywords: count, parameters, model size, 统计参数, 模型大小

    Function name:
        count_parameters
    Purpose:
        Count total trainable parameters in a model.
    Called by:
        ``train_and_evaluate``
    Calls:
        ``model.parameters()``
    Returns:
        Total number of trainable parameters.
    English keywords:
        count, parameters, model size
    """
    return sum(p.numel() for p in model.parameters())


def set_seed(seed: int) -> None:
    """设置 Python / NumPy / PyTorch 随机种子，保证实验可复现。

    中文说明:
    - 调用方 / Called by: ``train_and_evaluate``
    - 调用对象 / Calls: ``random.seed``, ``np.random.seed``, ``torch.manual_seed``, ``torch.cuda.manual_seed_all``
    - 作用 / Purpose: 统一设置所有随机源的种子，保证每次运行结果一致
    - 副作用 / Side effects: 修改全局随机状态
    - 关键词 / Keywords: seed, random, reproducibility, 随机种子, 可复现

    Function name:
        set_seed
    Purpose:
        Set random seeds for Python / NumPy / PyTorch for reproducibility.
    Called by:
        ``train_and_evaluate``
    Calls:
        ``random.seed``, ``np.random.seed``, ``torch.manual_seed``, ``torch.cuda.manual_seed_all``
    English keywords:
        seed, random, reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_evaluate(
    variant: dict[str, Any],
    device: torch.device,
    reports_dir: Path,
    swanlab_run: "SwanLabRunProxy | None" = None,
) -> dict[str, Any]:
    """训练一个变体并返回 PPL 结果。

    中文说明:
    - 调用方 / Called by: ``main``
    - 调用对象 / Calls: ``build_model``, ``count_parameters``, ``_train_loop``,
      ``set_seed``, ``download_wikitext103``, ``load_text``,
      ``create_dataloader``, ``create_eval_loader``
    - 作用 / Purpose: 对单个变体执行完整流程：配置、数据加载、模型构建、训练、OOM 降级
    - 错误处理 / Error handling: 捕获 ``torch.cuda.OutOfMemoryError``，
      若 batch_size > 2 则 halve 后重试；否则标记为 OOM 并跳过
    - 副作用 / Side effects: 下载数据集、创建 DataLoader、在 GPU 上分配模型参数
    - 事务边界 / Transaction: 无数据库事务；结果在函数外保存
    - 关键词 / Keywords: train, evaluate, variant, PPL, OOM, fallback, 训练, 评测, 变体

    Function name:
        train_and_evaluate
    Purpose:
        Run full train + evaluate pipeline for a single variant.
    Called by:
        ``main``
    Calls:
        ``build_model``, ``count_parameters``, ``_train_loop``,
        ``set_seed``, ``download_wikitext103``, ``load_text``,
        ``create_dataloader``, ``create_eval_loader``
    Parameters:
        - variant: variant config dict
        - device: torch device
        - reports_dir: Path to reports/ directory
    Returns:
        dict with keys: label, name, status, params, test_ppl, best_valid_ppl, ...
        On OOM/build failure: status = "OOM_build", "OOM", or "error"
    English keywords:
        train, evaluate, variant, PPL, OOM, fallback
    """
    label = variant["label"]
    print(f"\n{'='*60}")
    print(f"  Variant {label}: {variant['name']}")
    print(f"{'='*60}")

    # ---- 配置 ----
    base_cfg = LMExperimentConfig(
        dataset="wikitext103",
        dim=variant["dim"],
        heads=variant["heads"],
        num_layers=variant["num_layers"],
        batch_size=variant["batch_size"],
        device="auto",
        slots=variant.get("slots", 128),
        chunk_size=variant.get("chunk_size", 128),
    )
    set_seed(base_cfg.seed)
    tokenizer = CharTokenizer()
    vocab_size = tokenizer.vocab_size
    print(f"  Vocab size: {vocab_size}")
    print(f"  Device: {device}")

    # ---- 下载数据 ----
    print("\n  Downloading / loading WikiText-103...")
    paths = download_wikitext103(base_cfg.data_dir)

    # ---- 构建 DataLoader ----
    print("  Building data loaders...")
    train_text = load_text(paths["train"], max_chars=base_cfg.max_chars)
    valid_text = load_text(paths["valid"], max_chars=base_cfg.max_chars)
    test_text = load_text(paths["test"], max_chars=base_cfg.max_chars)

    train_loader = create_dataloader(
        train_text, tokenizer, base_cfg.seq_len, base_cfg.batch_size, shuffle=True,
    )
    valid_loader = create_eval_loader(
        valid_text, tokenizer, base_cfg.seq_len, base_cfg.eval_batch_size,
    )
    test_loader = create_eval_loader(
        test_text, tokenizer, base_cfg.seq_len, base_cfg.eval_batch_size,
    )
    print(
        f"  Train batches: {len(train_loader)}, "
        f"Valid seqs: {len(valid_loader.dataset)}, "
        f"Test seqs: {len(test_loader.dataset)}"
    )

    # ---- 构建模型 ----
    print("  Building model...")
    try:
        model = build_model(variant, vocab_size, device)
    except torch.cuda.OutOfMemoryError:
        print(f"  [OOM] Failed to build model for {label}, skipping.")
        return {"label": label, "name": variant["name"], "status": "OOM_build", "params": 0}

    total_params = count_parameters(model)
    print(f"  Parameters: {total_params:,}")

    # ---- 训练 ----
    print(f"  Training for {base_cfg.max_steps} steps...")
    try:
        results = _train_loop(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            config=base_cfg,
            device=device,
            label=label,
            variant=variant,
            swanlab_run=swanlab_run,
        )
    except torch.cuda.OutOfMemoryError:
        # 尝试降 batch_size
        if base_cfg.batch_size > 2:
            new_batch = base_cfg.batch_size // 2
            print(
                f"  [OOM] Reducing batch_size from {base_cfg.batch_size} "
                f"to {new_batch} and retrying..."
            )
            base_cfg_adj = LMExperimentConfig(
                dataset="wikitext103",
                dim=variant["dim"],
                heads=variant["heads"],
                num_layers=variant["num_layers"],
                batch_size=new_batch,
                device="auto",
                slots=variant.get("slots", 128),
                chunk_size=variant.get("chunk_size", 128),
            )
            train_loader = create_dataloader(
                train_text, tokenizer, base_cfg_adj.seq_len, new_batch, shuffle=True,
            )
            model = build_model(variant, vocab_size, device)
            results = _train_loop(
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                test_loader=test_loader,
                config=base_cfg_adj,
                device=device,
                label=label,
                variant=variant,
                swanlab_run=swanlab_run,
            )
            results["batch_size"] = new_batch
        else:
            print(f"  [OOM] Cannot reduce batch_size further, skipping {label}.")
            return {"label": label, "name": variant["name"], "status": "OOM", "params": total_params}

    results["label"] = label
    results["name"] = variant["name"]
    results["params"] = total_params
    results["status"] = "completed"
    results["dim"] = variant["dim"]
    results["num_layers"] = variant["num_layers"]
    results["slots"] = variant.get("slots", None)
    results["chunk_size"] = variant.get("chunk_size", None)
    results["batch_size"] = results.get("batch_size", base_cfg.batch_size)
    return results


def _train_loop(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    config: LMExperimentConfig,
    device: torch.device,
    label: str,
    variant: dict[str, Any],
    swanlab_run: "SwanLabRunProxy | None" = None,
) -> dict[str, Any]:
    """内部训练循环：执行训练、定期 valid 评测、最终 test 评测。

    中文说明:
    - 调用方 / Called by: ``train_and_evaluate``
    - 调用对象 / Calls: ``eval_ppl``, ``optim.AdamW``, ``CosineAnnealingLR``, ``CrossEntropyLoss``
    - 作用 / Purpose: 执行指定步数的训练，每 ``config.eval_interval`` 步在 valid 集评测，
      训练结束后在 test 集评测，返回指标字典
    - 错误处理 / Error handling: ``torch.cuda.OutOfMemoryError`` 向上传播至调用方处理；
      训练过程中不做局部 OOM 捕获
    - 副作用 / Side effects: 修改模型参数、更新优化器和调度器状态
    - 事务边界 / Transaction: 无数据库事务；所有状态在内存中
    - 并发与幂等 / Concurrency: 非线程安全；多次调用会产生不同结果（受随机种子影响）
    - 关键词 / Keywords: train loop, optimizer, scheduler, evaluation, PPL, 训练循环, 评测

    Function name:
        _train_loop
    Purpose:
        Internal training loop with periodic valid evaluation and final test PPL.
    Called by:
        ``train_and_evaluate``
    Calls:
        ``eval_ppl``, ``optim.AdamW``, ``CosineAnnealingLR``, ``CrossEntropyLoss``
    Parameters:
        - model: the model to train
        - train_loader: training DataLoader (shuffled, drop_last)
        - valid_loader: validation DataLoader (non-shuffled, no drop_last)
        - test_loader: test DataLoader (non-shuffled, no drop_last)
        - config: LMExperimentConfig with hyper-parameters
        - device: torch device
        - label: variant label string
        - variant: variant config dict
    Returns:
        dict with keys: test_ppl, best_valid_ppl, best_valid_step,
        total_steps, total_time_s, avg_step_ms, valid_curve
    Error handling:
        Propagates torch.cuda.OutOfMemoryError upward for caller to handle.
    English keywords:
        train loop, optimizer, scheduler, evaluation, PPL
    """
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.95))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_steps)
    criterion = nn.CrossEntropyLoss()
    model.train()

    total_steps = 0
    best_valid_ppl = float("inf")
    best_valid_step = 0
    step_times: list[float] = []
    train_losses: list[float] = []
    valid_results: list[dict[str, Any]] = []

    train_iter = iter(train_loader)

    while total_steps < config.max_steps:
        try:
            batch_x, batch_y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch_x, batch_y = next(train_iter)

        t0 = time.time()
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
        optimizer.step()
        scheduler.step()
        step_time = time.time() - t0

        total_steps += 1
        step_times.append(step_time)
        train_losses.append(loss.item())

        if total_steps % config.eval_interval == 0:
            train_ppl = math.exp(loss.item())

            # 在 valid 集上评测
            valid_ppl = eval_ppl(model, valid_loader, device)
            best_valid_ppl = min(best_valid_ppl, valid_ppl)
            if best_valid_ppl == valid_ppl:
                best_valid_step = total_steps

            avg_step_ms = (sum(step_times[-100:]) / max(len(step_times[-100:]), 1)) * 1000
            print(
                f"  [{label}] Step {total_steps:6d}/{config.max_steps} | "
                f"Train PPL: {train_ppl:.2f} | "
                f"Valid PPL: {valid_ppl:.2f} | "
                f"Best Valid: {best_valid_ppl:.2f}@{best_valid_step} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                f"Step: {avg_step_ms:.0f}ms"
            )

            valid_results.append({
                "step": total_steps,
                "train_ppl": round(train_ppl, 4),
                "valid_ppl": round(valid_ppl, 4),
                "best_valid_ppl": round(best_valid_ppl, 4),
            })

            if swanlab_run is not None:
                swanlab_run.log(
                    {"ppl": valid_ppl, "loss": loss.item(), "train_ppl": train_ppl},
                    step=total_steps,
                )

    # ---- 最终 test 集评测 ----
    print(f"\n  [{label}] Final evaluation on test set...")
    test_ppl = eval_ppl(model, test_loader, device)
    print(f"  [{label}] Test PPL: {test_ppl:.2f}")

    total_time_s = sum(step_times)
    avg_step_ms = (sum(step_times) / max(len(step_times), 1)) * 1000

    return {
        "test_ppl": round(test_ppl, 4),
        "best_valid_ppl": round(best_valid_ppl, 4),
        "best_valid_step": best_valid_step,
        "total_steps": total_steps,
        "total_time_s": round(total_time_s, 1),
        "avg_step_ms": round(avg_step_ms, 2),
        "valid_curve": valid_results,
    }


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有变体的 PPL 评测，生成 JSON 和 Markdown 报告。

    中文说明:
    - 调用方 / Called by: 命令行入口（``python -m scripts.wikitext103_ppl``）
    - 调用对象 / Calls: ``train_and_evaluate``, ``_save_interim_results``,
      ``_generate_markdown_report``, ``_save_final_json``, ``_print_summary``
    - 作用 / Purpose: 依次执行所有变体的训练 + 评测，每完成一个变体保存中间结果，
      全部完成后生成汇总 Markdown 报告
    - 错误处理 / Error handling: 每个变体单独 try/except，异常不会阻塞后续变体
    - 副作用 / Side effects: 写 reports/ 目录下的 JSON 和 Markdown 文件
    - 事务边界 / Transaction: 无数据库事务
    - 关键词 / Keywords: main, entry point, PPL evaluation, report, 主入口, 评测入口

    Function name:
        main
    Purpose:
        Run PPL evaluation for all variants, save JSON and Markdown reports.
    Called by:
        Command line: ``python -m scripts.wikitext103_ppl``
    Calls:
        ``train_and_evaluate``, ``_save_interim_results``,
        ``_generate_markdown_report``, ``_save_final_json``, ``_print_summary``
    English keywords:
        main, entry point, PPL evaluation, report
    """
    print("=" * 60)
    print("  WikiText-103 PPL Evaluation Suite")
    print("=" * 60)

    device = resolve_device("auto")
    reports_dir = ensure_reports_dir(PROJECT_ROOT)

    swanlab_run = init_swanlab(
        project="MHDSRA2",
        experiment_name="ppl_wikitext103",
        config={"model_type": "mhdsra2"},
        mode="cloud",
        tags=["ppl"],
    )

    all_results: list[dict[str, Any]] = []

    for variant in VARIANT_CONFIGS:
        try:
            result = train_and_evaluate(variant, device, reports_dir, swanlab_run)
        except Exception as e:
            print(f"\n  [ERROR] Variant {variant['label']} failed: {e}")
            traceback.print_exc()
            result = {
                "label": variant["label"],
                "name": variant["name"],
                "status": "error",
                "error": str(e),
            }
        all_results.append(result)

        # 每完成一个变体就保存中间结果（避免全部跑完后丢数据）
        _save_interim_results(all_results, reports_dir)

        # 释放 GPU 内存，为下一个变体做准备
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- 生成最终报告 ----
    swanlab_run.finish()
    _generate_markdown_report(all_results, reports_dir)
    _save_final_json(all_results, reports_dir)
    _print_summary(all_results)


def _save_interim_results(results: list[dict[str, Any]], reports_dir: Path) -> None:
    """保存中间结果 JSON，防止长时间运行后意外丢失全部结果。

    中文说明:
    - 调用方 / Called by: ``main``（每完成一个变体后调用）
    - 调用对象 / Calls: ``json.dump``
    - 作用 / Purpose: 将当前已完成变体的结果写入 ``reports/wikitext103_ppl_interim.json``
    - 副作用 / Side effects: 写文件到 reports/ 目录
    - 关键词 / Keywords: interim, save, JSON, checkpoint, 中间结果, 保存

    Function name:
        _save_interim_results
    Purpose:
        Save intermediate results to prevent data loss on long runs.
    Called by:
        ``main`` after each variant completes.
    Calls:
        ``json.dump``
    Side effects:
        Writes JSON file to reports/ directory.
    English keywords:
        interim, save, JSON, checkpoint
    """
    path = reports_dir / "wikitext103_ppl_interim.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def _save_final_json(results: list[dict[str, Any]], reports_dir: Path) -> None:
    """保存最终结果 JSON。

    中文说明:
    - 调用方 / Called by: ``main``
    - 调用对象 / Calls: ``json.dump``
    - 作用 / Purpose: 将所有变体的最终结果写入 ``reports/wikitext103_ppl_results.json``
    - 副作用 / Side effects: 写文件到 reports/ 目录
    - 关键词 / Keywords: final, save, JSON, results, 最终结果, 保存

    Function name:
        _save_final_json
    Purpose:
        Save final result JSON.
    Called by:
        ``main``
    Calls:
        ``json.dump``
    English keywords:
        final, save, JSON, results
    """
    path = reports_dir / "wikitext103_ppl_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {path}")


def _generate_markdown_report(results: list[dict[str, Any]], reports_dir: Path) -> None:
    """生成 Markdown 格式的对比报告，包含汇总表和收敛曲线。

    中文说明:
    - 调用方 / Called by: ``main``
    - 调用对象 / Calls: ``time.strftime``
    - 作用 / Purpose: 生成可读的 Markdown 报告，包含所有变体的 PPL 对比表格
      和 valid 集收敛曲线采样
    - 副作用 / Side effects: 写文件到 reports/ 目录
    - 关键词 / Keywords: markdown, report, summary, table, curve, 报告, 汇总

    Function name:
        _generate_markdown_report
    Purpose:
        Generate a Markdown comparison report with summary table and PPL curves.
    Called by:
        ``main``
    Calls:
        ``time.strftime``
    English keywords:
        markdown, report, summary, table, curve
    """
    lines: list[str] = []
    lines.append("# WikiText-103 PPL 对比报告")
    lines.append("")
    lines.append(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # 汇总表
    lines.append("## 汇总")
    lines.append("")
    lines.append(
        "| 变体 | 模型 | dim | layers | slots | chunk | 参数量 | batch | "
        "Test PPL | Best Valid PPL | 训练时间 | 状态 |"
    )
    lines.append(
        "|------|------|-----|--------|-------|-------|--------|-------|"
        "----------|----------------|----------|------|"
    )

    for r in results:
        label = r.get("label", "?")
        name = r.get("name", "?")
        dim = r.get("dim", "-")
        layers = r.get("num_layers", "-")
        slots = r.get("slots", "-") if r.get("slots") is not None else "N/A"
        chunk = r.get("chunk_size", "-") if r.get("chunk_size") is not None else "N/A"
        params = f"{r.get('params', 0):,}" if r.get("params") else "-"
        bs = r.get("batch_size", "-")
        status = r.get("status", "?")
        test_ppl = (
            f"{r['test_ppl']:.2f}"
            if isinstance(r.get("test_ppl"), (int, float))
            else "-"
        )
        valid_ppl = (
            f"{r['best_valid_ppl']:.2f}"
            if isinstance(r.get("best_valid_ppl"), (int, float))
            else "-"
        )
        time_s = (
            f"{r['total_time_s']:.0f}s"
            if isinstance(r.get("total_time_s"), (int, float))
            else "-"
        )

        lines.append(
            f"| {label} | {name} | {dim} | {layers} | {slots} | {chunk} | "
            f"{params} | {bs} | {test_ppl} | {valid_ppl} | {time_s} | {status} |"
        )

    lines.append("")

    # 详细曲线
    lines.append("## Valid PPL 收敛曲线")
    lines.append("")
    lines.append("```")
    for r in results:
        label = r.get("label", "?")
        curve = r.get("valid_curve", [])
        if curve:
            lines.append(f"  [{label}] {r.get('name', '?')}:")
            # 取约 5 个采样点
            step = max(1, len(curve) // 5)
            for entry in curve[::step]:
                lines.append(
                    f"    Step {entry['step']:6d}: "
                    f"Train PPL={entry['train_ppl']:.2f}, "
                    f"Valid PPL={entry['valid_ppl']:.2f}"
                )
    lines.append("```")
    lines.append("")

    # 保存
    path = reports_dir / "wikitext103_ppl_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Report saved to: {path}")


def _print_summary(results: list[dict[str, Any]]) -> None:
    """打印控制台摘要。

    中文说明:
    - 调用方 / Called by: ``main``
    - 调用对象 / Calls: 无外部函数，仅使用 ``print``
    - 作用 / Purpose: 在控制台输出所有变体的最终 PPL 和状态
    - 关键词 / Keywords: summary, print, console, 摘要, 输出

    Function name:
        _print_summary
    Purpose:
        Print a summary of all variant results to the console.
    Called by:
        ``main``
    English keywords:
        summary, print, console
    """
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for r in results:
        label = r.get("label", "?")
        name = r.get("name", "?")
        status = r.get("status", "?")
        test_ppl = r.get("test_ppl", "-")
        valid_ppl = r.get("best_valid_ppl", "-")
        params = r.get("params", "-")
        print(f"  [{label}] {name}: Params={params:,}, Test PPL={test_ppl}, Best Valid PPL={valid_ppl}, Status={status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
