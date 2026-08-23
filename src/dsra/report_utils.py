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


def _json_default(obj):
    if hasattr(obj, "detach") and hasattr(obj, "cpu"):
        return obj.detach().cpu().tolist()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 150) -> Path:
    """Save a matplotlib figure, ensuring parent directory exists."""
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
