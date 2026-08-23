"""Run resumable multi-seed tiny LM practical comparisons."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_NAME = "mhdsra2_practical_tiny_lm_multiseed"


def parse_csv_ints(value: str) -> tuple[int, ...]:
    """Parse a comma-separated integer list for CLI seed arguments.

    中文说明:
    - 调用方 / Called by: `build_parser`.
    - 调用对象 / Calls: `str.split`, `int`.
    - 作用 / Purpose: 让多 seed tiny LM 验证可以用 `--seeds 1234,2025` 明确复现。
    - 参数 / Parameters: `value` 是逗号分隔的整数字符串。
    - 返回 / Returns: 整数 seed 元组。
    - 错误处理 / Error handling: 空列表或非法整数会抛 `ValueError`。
    - 副作用 / Side effects: 无。
    """
    seeds = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if not seeds:
        raise ValueError("seeds must contain at least one integer")
    return seeds


def _extract_float(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"Could not parse pattern: {pattern}")
    return float(match.group(1))


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    """Return mean/std/n for a metric vector.

    中文说明:
    - 调用方 / Called by: `build_payload`.
    - 调用对象 / Calls: `statistics.mean`, `statistics.stdev`.
    - 作用 / Purpose: 在报告里同时记录均值、波动和样本数，避免只看单次最好结果。
    - 参数 / Parameters: `values` 是同一指标的多 seed 数值。
    - 返回 / Returns: 包含 mean/std/n 的字典；空列表返回空统计。
    - 错误处理 / Error handling: 非数值由调用方保证。
    - 副作用 / Side effects: 无。
    """
    if not values:
        return {"mean": None, "std": None, "n": 0}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0, "n": 1}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
        "n": len(values),
    }


def parse_compare_output(seed: int, stdout: str, command: list[str]) -> dict[str, Any]:
    """Parse `tiny_llama_compare.py` stdout into one seed result row.

    中文说明:
    - 调用方 / Called by: `run_one_seed`.
    - 调用对象 / Calls: `_extract_float`.
    - 作用 / Purpose: 将正式 CLI 的文本输出转换成机器可汇总 JSON 指标。
    - 参数 / Parameters: `seed` 是当前 seed，`stdout` 是子进程标准输出，`command` 是命令。
    - 返回 / Returns: 单 seed 指标行。
    - 错误处理 / Error handling: 输出格式异常时抛 `ValueError`，避免写入半解析结果。
    - 副作用 / Side effects: 无。
    """
    standard_ppl = _extract_float(r"Standard Attention Validation PPL:\s+([0-9.]+)", stdout)
    mhdsra2_ppl = _extract_float(r"MHDSRA2 Validation PPL:\s+([0-9.]+)", stdout)
    ratio = _extract_float(r"Ratio:\s+([0-9.]+)x", stdout)
    standard_time_s = _extract_float(r"Training Time Std:\s+([0-9.]+)s", stdout)
    mhdsra2_time_s = _extract_float(r"Training Time MHDSRA2:\s+([0-9.]+)s", stdout)
    return {
        "seed": int(seed),
        "status": "completed",
        "command": command,
        "standard_validation_ppl": standard_ppl,
        "mhdsra2_validation_ppl": mhdsra2_ppl,
        "ppl_ratio_mhdsra2_over_standard": ratio,
        "standard_time_s": standard_time_s,
        "mhdsra2_time_s": mhdsra2_time_s,
        "time_ratio_mhdsra2_over_standard": (
            mhdsra2_time_s / standard_time_s if standard_time_s else None
        ),
    }


def run_one_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    """Run one tiny LM comparison seed through the public CLI.

    中文说明:
    - 调用方 / Called by: `run_multiseed`.
    - 调用对象 / Calls: `subprocess.run`, `parse_compare_output`.
    - 作用 / Purpose: 每个 seed 独立执行并解析，方便中断后继续。
    - 参数 / Parameters: `args` 是 CLI 参数，`seed` 是当前随机种子。
    - 返回 / Returns: 单 seed 完成或失败行。
    - 错误处理 / Error handling: 子进程失败时记录 stdout/stderr 与 returncode。
    - 副作用 / Side effects: 启动训练子进程，会使用指定 device。
    """
    command = [
        sys.executable,
        "scripts/tiny_llama_compare.py",
        "--seq-len",
        str(args.seq_len),
        "--batch-size",
        str(args.batch_size),
        "--max-steps",
        str(args.max_steps),
        "--dim",
        str(args.dim),
        "--heads",
        str(args.heads),
        "--layers",
        str(args.layers),
        "--seed",
        str(seed),
        "--mhdsra2-chunk-size",
        str(args.mhdsra2_chunk_size),
        "--device",
        str(args.device),
    ]
    try:
        proc = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=args.seed_timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "seed": int(seed),
            "status": "timeout",
            "command": command,
            "timeout_sec": int(args.seed_timeout_sec),
            "stdout_tail": (exc.stdout or "")[-4000:],
            "stderr_tail": (exc.stderr or "")[-4000:],
        }
    if proc.returncode != 0:
        return {
            "seed": int(seed),
            "status": "failed",
            "command": command,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    row = parse_compare_output(seed, proc.stdout, command)
    row["stdout_tail"] = proc.stdout[-4000:]
    return row


def build_payload(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a resumable multi-seed report payload.

    中文说明:
    - 调用方 / Called by: `save_reports`, `run_multiseed`.
    - 调用对象 / Calls: `summarize_values`.
    - 作用 / Purpose: 聚合已完成 seed 的 PPL 与耗时指标，并保留失败行。
    - 参数 / Parameters: `args` 是实验配置，`rows` 是已有与新增结果。
    - 返回 / Returns: 可写入 JSON 的报告对象。
    - 错误处理 / Error handling: 缺失指标的失败行不会进入均值统计。
    - 副作用 / Side effects: 无。
    """
    completed = [row for row in rows if row.get("status") == "completed"]
    return {
        "name": args.report_name,
        "config": {
            "seeds": list(args.seeds),
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "max_steps": args.max_steps,
            "dim": args.dim,
            "heads": args.heads,
            "layers": args.layers,
            "mhdsra2_chunk_size": args.mhdsra2_chunk_size,
            "device": args.device,
            "seed_timeout_sec": args.seed_timeout_sec,
        },
        "rows": rows,
        "summary": {
            "completed_count": len(completed),
            "failed_count": len(rows) - len(completed),
            "standard_validation_ppl": summarize_values(
                [float(row["standard_validation_ppl"]) for row in completed]
            ),
            "mhdsra2_validation_ppl": summarize_values(
                [float(row["mhdsra2_validation_ppl"]) for row in completed]
            ),
            "ppl_ratio_mhdsra2_over_standard": summarize_values(
                [float(row["ppl_ratio_mhdsra2_over_standard"]) for row in completed]
            ),
            "time_ratio_mhdsra2_over_standard": summarize_values(
                [
                    float(row["time_ratio_mhdsra2_over_standard"])
                    for row in completed
                    if row.get("time_ratio_mhdsra2_over_standard") is not None
                ]
            ),
        },
    }


def build_markdown(payload: dict[str, Any]) -> list[str]:
    """Render the multi-seed JSON payload as Markdown.

    中文说明:
    - 调用方 / Called by: `save_reports`.
    - 调用对象 / Calls: string formatting.
    - 作用 / Purpose: 让非技术读者能直接查看多 seed PPL 和耗时结论。
    - 参数 / Parameters: `payload` 是 `build_payload` 的结果。
    - 返回 / Returns: Markdown 行列表。
    - 错误处理 / Error handling: 缺失字段按 `NA` 显示。
    - 副作用 / Side effects: 无。
    """
    summary = payload["summary"]
    lines = [
        "# MHDSRA2 Practical Tiny LM Multiseed",
        "",
        "## Config",
        "",
    ]
    for key, value in payload["config"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- completed_count: `{summary['completed_count']}`",
            f"- failed_count: `{summary['failed_count']}`",
            f"- standard_validation_ppl_mean: `{summary['standard_validation_ppl']['mean']}`",
            f"- mhdsra2_validation_ppl_mean: `{summary['mhdsra2_validation_ppl']['mean']}`",
            f"- ppl_ratio_mean: `{summary['ppl_ratio_mhdsra2_over_standard']['mean']}`",
            f"- time_ratio_mean: `{summary['time_ratio_mhdsra2_over_standard']['mean']}`",
            "",
            "## Rows",
            "",
            "| seed | status | standard PPL | MHDSRA2 PPL | PPL ratio | standard s | MHDSRA2 s | time ratio |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            "| {seed} | {status} | {standard} | {mh} | {ratio} | {std_time} | {mh_time} | {time_ratio} |".format(
                seed=row.get("seed"),
                status=row.get("status"),
                standard=row.get("standard_validation_ppl", "NA"),
                mh=row.get("mhdsra2_validation_ppl", "NA"),
                ratio=row.get("ppl_ratio_mhdsra2_over_standard", "NA"),
                std_time=row.get("standard_time_s", "NA"),
                mh_time=row.get("mhdsra2_time_s", "NA"),
                time_ratio=row.get("time_ratio_mhdsra2_over_standard", "NA"),
            )
        )
    return lines


def save_reports(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Path]:
    """Write JSON and Markdown reports after each completed seed.

    中文说明:
    - 调用方 / Called by: `run_multiseed`.
    - 调用对象 / Calls: `build_payload`, `build_markdown`, `Path.write_text`.
    - 作用 / Purpose: 保证长实验即使中断，也会留下已完成 seed 的机器报告。
    - 参数 / Parameters: `args` 是实验配置，`rows` 是当前全部结果。
    - 返回 / Returns: JSON 与 Markdown 路径。
    - 错误处理 / Error handling: 文件写入异常直接抛出，避免误以为报告已保存。
    - 副作用 / Side effects: 写入 `reports/`。
    """
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(args, rows)
    json_path = reports_dir / f"{args.report_name}.json"
    md_path = reports_dir / f"{args.report_name}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_markdown(payload)) + "\n", encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def load_existing_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load previous report rows for resumable runs.

    中文说明:
    - 调用方 / Called by: `run_multiseed`.
    - 调用对象 / Calls: `json.loads`, `Path.read_text`.
    - 作用 / Purpose: 支持 `--resume` 跳过已经完成的 seed。
    - 参数 / Parameters: `args` 提供 reports_dir 与 report_name。
    - 返回 / Returns: 已有行列表；无文件时为空。
    - 错误处理 / Error handling: JSON 损坏会抛错，避免覆盖坏账本。
    - 副作用 / Side effects: 只读文件。
    """
    if not args.resume:
        return []
    path = Path(args.reports_dir) / f"{args.report_name}.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("rows", []))


def run_multiseed(args: argparse.Namespace) -> dict[str, Path]:
    """Run configured seeds and save a resumable practical report.

    中文说明:
    - 调用方 / Called by: `main`.
    - 调用对象 / Calls: `load_existing_rows`, `run_one_seed`, `save_reports`.
    - 作用 / Purpose: 对 tiny LM 实战入口做可恢复的多 seed 验证。
    - 参数 / Parameters: `args` 是 CLI 配置。
    - 返回 / Returns: 最终报告路径。
    - 错误处理 / Error handling: 单 seed 失败会写入失败行并继续保存。
    - 副作用 / Side effects: 启动训练子进程并写报告。
    """
    rows = load_existing_rows(args)
    completed_seeds = {
        int(row["seed"]) for row in rows if row.get("status") == "completed"
    }
    save_paths = save_reports(args, rows)
    for seed in args.seeds:
        if int(seed) in completed_seeds:
            continue
        row = run_one_seed(args, int(seed))
        rows.append(row)
        save_paths = save_reports(args, rows)
        if row.get("status") != "completed" and args.stop_on_failure:
            break
    return save_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run resumable tiny LM practical multi-seed checks")
    parser.add_argument("--seeds", type=parse_csv_ints, default=(1234, 2025, 3036))
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--mhdsra2-chunk-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "docs" / "reports")
    parser.add_argument("--report-name", type=str, default=DEFAULT_REPORT_NAME)
    parser.add_argument("--seed-timeout-sec", type=int, default=900)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Path]:
    args = build_parser().parse_args(argv)
    paths = run_multiseed(args)
    print(f"TINY_LM_PRACTICAL_MULTI_SEED_JSON={paths['json']}")
    print(f"TINY_LM_PRACTICAL_MULTI_SEED_MD={paths['markdown']}")
    return paths


if __name__ == "__main__":
    main()
