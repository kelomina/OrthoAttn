import json
from pathlib import Path

import matplotlib.pyplot as plt


def ensure_reports_dir(base_dir):
    base_dir = Path(base_dir)
    reports_dir = base_dir if base_dir.name == "reports" else base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def write_markdown(path, lines):
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 150) -> Path:
    """Save a matplotlib figure, ensuring parent directory exists.

    中文说明:
    - 调用方 / Called by: report scripts that produce figures.
    - 调用对象 / Calls: ``Path.mkdir``, ``fig.savefig``.
    - 作用 / Purpose: 统一保存 matplotlib 图到 reports/figures/，确保目录存在。
    - 变量 / Variables: ``fig`` 是要保存的图对象, ``path`` 是输出路径。
    - 接入 / Integration: 新增图表的报告脚本应优先使用本函数。
    - 错误处理 / Error handling: 写入失败直接抛出。
    - 关键词 / Keywords: figure, matplotlib, save, reports, visualization, dpi, png, 保存, 图, 可视化

    English documentation:
    Function name:
        save_figure
    Purpose:
        Save a matplotlib figure, ensuring parent directory exists.
    Called by:
        Report scripts that produce figures.
    Calls:
        ``Path.mkdir``, ``fig.savefig``.
    Parameters:
        - fig: matplotlib Figure to save.
        - path: output file path (str or Path).
        - dpi: resolution (default 150).
    Returns:
        Resolved Path of the saved figure.
    Integration:
        New report scripts with figures should use this function.
    Error handling:
        Write failures propagate directly.
    English keywords:
        figure, matplotlib, save, reports, visualization, dpi, png
    """
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(resolved), dpi=dpi, bbox_inches="tight")
    return resolved


def build_capacity_markdown(forward_results, train_results):
    lengths = sorted(set(forward_results.keys()) | set(train_results.keys()))
    lines = [
        "# Needle Capacity Results",
        "",
        "| Context | Forward Only | Forward Peak Mem (MB) | Train Step | Train Peak Mem (MB) |",
        "|---:|:---:|---:|:---:|---:|",
    ]
    for seq_len in lengths:
        forward = forward_results.get(seq_len, {"status": "missing"})
        train = train_results.get(seq_len, {"status": "missing"})
        forward_status = "PASS" if forward.get("status") == "ok" else "OOM"
        train_status = "PASS" if train.get("status") == "ok" else "OOM"
        forward_mem = f"{forward.get('peak_mem_mb', 0.0):.2f}" if forward.get("status") == "ok" else "-"
        train_mem = f"{train.get('peak_mem_mb', 0.0):.2f}" if train.get("status") == "ok" else "-"
        lines.append(f"| {seq_len} | {forward_status} | {forward_mem} | {train_status} | {train_mem} |")
    return lines


def build_ablation_markdown(results):
    lines = [
        "# Ablation Summary",
        "",
        "Main table reports per-learning-rate aggregates across the same seed set for every variant. Best single runs are listed separately as tuning diagnostics only.",
        "",
        "| Variant | LR | Seeds | Final Eval Acc Mean | Final Eval Acc Std | Best Eval Acc Mean | Final Eval Loss Mean |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        if "by_lr" not in result:
            lines.append(
                f"| {name} | {result['lr']:.0e} | - | {result['final_eval_acc']*100:.2f}% | "
                f"- | {result['best_eval_acc']*100:.2f}% | {result['final_eval_loss']:.4f} |"
            )
            continue
        for row in result["by_lr"]:
            seed_text = ",".join(str(seed) for seed in row["seeds"])
            lines.append(
                f"| {name} | {row['lr']:.0e} | {seed_text} | "
                f"{row['final_eval_acc_mean']*100:.2f}% | "
                f"{row['final_eval_acc_std']*100:.2f}% | "
                f"{row['best_eval_acc_mean']*100:.2f}% | "
                f"{row['final_eval_loss_mean']:.4f} |"
            )
    if any("best_single_run" in result for result in results.values()):
        lines.extend(
            [
                "",
                "## Best Single Runs (Tuning Appendix)",
                "",
                "| Variant | LR | Seed | Final Eval Acc | Best Eval Acc | Final Eval Loss |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, result in results.items():
            best_run = result.get("best_single_run")
            if best_run is None:
                continue
            lines.append(
                f"| {name} | {best_run['lr']:.0e} | {best_run['seed']} | "
                f"{best_run['final_eval_acc']*100:.2f}% | "
                f"{best_run['best_eval_acc']*100:.2f}% | "
                f"{best_run['final_eval_loss']:.4f} |"
            )
    return lines
