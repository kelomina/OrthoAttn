import json
from pathlib import Path


def ensure_reports_dir(base_dir):
    base_dir = Path(base_dir)
    reports_dir = base_dir if base_dir.name == "reports" else base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def write_markdown(path, lines):
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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
        "| Variant | Best LR | Final Eval Acc | Best Eval Acc | Final Eval Loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result['lr']:.0e} | {result['final_eval_acc']*100:.2f}% | "
            f"{result['best_eval_acc']*100:.2f}% | {result['final_eval_loss']:.4f} |"
        )
    return lines
