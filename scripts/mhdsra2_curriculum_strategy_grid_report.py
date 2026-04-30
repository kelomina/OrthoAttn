"""CLI report for MHDSRA2 curriculum strategy grid scans."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dsra.application import (  # noqa: E402
    build_curriculum_strategy_grid_markdown,
    build_curriculum_strategy_grid_payload,
)
from src.dsra.application.arithmetic_emergence_service import (  # noqa: E402
    CURRICULUM_STAGE_EXACT_MATCH_THRESHOLD,
    DEFAULT_CURRICULUM_EVAL_INTERVAL,
    DEFAULT_MAX_STEPS_PER_STAGE,
    DEFAULT_SEEDS,
    DEFAULT_STRATEGY_GRID_LAYER_COUNTS,
    DEFAULT_STRATEGY_GRID_REPLAY_RATIOS,
    DEFAULT_STRATEGY_GRID_STEP_BUDGETS,
    DEFAULT_STRATEGY_GRID_STAGE_PATIENCES,
)
from src.dsra.report_utils import ensure_reports_dir, write_json, write_markdown  # noqa: E402


def parse_csv_ints(value: str) -> tuple[int, ...]:
    """Parse comma-separated integers from CLI input.

    中文说明:
    - 调用方 / Called by: `build_parser`.
    - 调用对象 / Calls: `str.split`, `int`.
    - 作用 / Purpose: 将 `--layers`, `--seeds`, `--stage-patiences` 转成强类型整数元组。
    - 变量 / Variables: `value` 是逗号分隔文本, 返回值是过滤空片段后的整数元组。
    - 接入 / Integration: 作为 argparse `type=` 回调接入网格扫描 CLI。
    - 错误处理 / Error handling: 非整数输入由 `int` 抛出 `ValueError`, argparse 会展示错误。
    - 关键词 / Keywords:
      parse|csv|integers|layers|seeds|stage_patience|cli|grid|解析|整数
    """
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_csv_floats(value: str) -> tuple[float, ...]:
    """Parse comma-separated floats from CLI input.

    中文说明:
    - 调用方 / Called by: `build_parser`.
    - 调用对象 / Calls: `str.split`, `float`.
    - 作用 / Purpose: 将 `--replay-ratios` 转成强类型浮点元组。
    - 变量 / Variables: `value` 是逗号分隔文本, 返回值是 replay ratio 候选集合。
    - 接入 / Integration: 作为 argparse `type=` 回调接入网格扫描 CLI。
    - 错误处理 / Error handling: 非浮点输入由 `float` 抛出 `ValueError`, argparse 会展示错误。
    - 关键词 / Keywords:
      parse|csv|floats|replay_ratio|strategy|cli|grid|mhdsra2|解析|浮点
    """
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def save_curriculum_strategy_grid_reports(
    payload: dict[str, object],
    reports_dir: Path | str,
) -> tuple[Path, Path]:
    """Write curriculum strategy grid JSON and Markdown reports.

    中文说明:
    - 调用方 / Called by: `main`, tests, `scripts.main.run_mhdsra2_curriculum_strategy_grid`.
    - 调用对象 / Calls: `ensure_reports_dir`, `write_json`, `write_markdown`,
      `build_curriculum_strategy_grid_markdown`.
    - 作用 / Purpose: 将 replay ratio 与 stage patience 小型网格扫描写入 `reports/`。
    - 变量 / Variables: `payload` 是应用层报告数据, `reports_dir` 是目标目录。
    - 接入 / Integration: 统一产物名供主脚本和回归测试引用。
    - 错误处理 / Error handling: 文件系统或 payload 类型错误直接抛出, 不静默忽略。
    - 关键词 / Keywords:
      save|reports|json|markdown|strategy_grid|replay_ratio|patience|mhdsra2|报告|保存
    """
    resolved_reports_dir = ensure_reports_dir(reports_dir)
    json_path = resolved_reports_dir / "mhdsra2_curriculum_strategy_grid.json"
    markdown_path = resolved_reports_dir / "mhdsra2_curriculum_strategy_grid.md"
    write_json(json_path, payload)
    write_markdown(markdown_path, build_curriculum_strategy_grid_markdown(payload))
    return json_path, markdown_path


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for curriculum strategy grid scans.

    中文说明:
    - 调用方 / Called by: `main`, tests.
    - 调用对象 / Calls: `argparse.ArgumentParser`, `parse_csv_ints`, `parse_csv_floats`.
    - 作用 / Purpose: 暴露 replay ratio, stage patience, layer, seed 和训练节奏参数。
    - 变量 / Variables: `parser` 是命令行解析器, 各参数映射到应用层 payload 构建器。
    - 接入 / Integration: `python scripts/main.py mhdsra2_curriculum_strategy_grid`。
    - 错误处理 / Error handling: 非法输入由 argparse 报错并退出。
    - 关键词 / Keywords:
      parser|cli|strategy_grid|replay_ratio|stage_patience|layers|seeds|mhdsra2|入口|参数
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-ratios",
        type=parse_csv_floats,
        default=DEFAULT_STRATEGY_GRID_REPLAY_RATIOS,
        help="Comma-separated replay ratios to scan.",
    )
    parser.add_argument(
        "--stage-patiences",
        type=parse_csv_ints,
        default=DEFAULT_STRATEGY_GRID_STAGE_PATIENCES,
        help="Comma-separated stage patience values to scan.",
    )
    parser.add_argument(
        "--layers",
        type=parse_csv_ints,
        default=DEFAULT_STRATEGY_GRID_LAYER_COUNTS,
        help="Comma-separated MHDSRA2 layer counts to scan.",
    )
    parser.add_argument("--seeds", type=parse_csv_ints, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--max-steps-per-stage",
        type=int,
        default=DEFAULT_MAX_STEPS_PER_STAGE,
        help="Maximum active-stage training steps before a stage can stop.",
    )
    parser.add_argument(
        "--max-steps-per-stage-values",
        type=parse_csv_ints,
        default=None,
        help=(
            "Comma-separated active-stage step budgets to scan. "
            f"Default: {','.join(str(item) for item in DEFAULT_STRATEGY_GRID_STEP_BUDGETS)}."
        ),
    )
    parser.add_argument(
        "--curriculum-eval-interval",
        type=int,
        default=DEFAULT_CURRICULUM_EVAL_INTERVAL,
        help="Evaluate open-stage EM every N active-stage training steps.",
    )
    parser.add_argument(
        "--stage-threshold",
        type=float,
        default=CURRICULUM_STAGE_EXACT_MATCH_THRESHOLD,
        help="Exact-match threshold required for all open stages.",
    )
    parser.add_argument(
        "--target-stage-count",
        type=int,
        default=2,
        help="Number of early curriculum stages that must be retained across all seeds.",
    )
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    """Run the curriculum strategy grid report CLI.

    中文说明:
    - 调用方 / Called by: command line and `scripts.main.run_mhdsra2_curriculum_strategy_grid`.
    - 调用对象 / Calls: `build_parser`, `build_curriculum_strategy_grid_payload`,
      `save_curriculum_strategy_grid_reports`.
    - 作用 / Purpose: 扫描 replay ratio 与 stage patience, 寻找稳定保留前 N 阶段的策略。
    - 变量 / Variables: `argv` 是可选 CLI 参数, `payload` 是 JSON/Markdown 共用报告数据。
    - 接入 / Integration: 通过统一主入口或本脚本直接运行。
    - 错误处理 / Error handling: 参数、训练或写报告失败都会直接抛出并导致命令失败。
    - 关键词 / Keywords:
      main|cli|strategy_grid|replay_ratio|stage_patience|retention|mhdsra2|report|运行|网格
    """
    args = build_parser().parse_args(argv)
    payload = build_curriculum_strategy_grid_payload(
        replay_ratios=args.replay_ratios,
        stage_patiences=args.stage_patiences,
        layer_counts=args.layers,
        seeds=args.seeds,
        max_steps_per_stage=args.max_steps_per_stage,
        max_steps_per_stage_values=args.max_steps_per_stage_values,
        curriculum_eval_interval=args.curriculum_eval_interval,
        stage_threshold=args.stage_threshold,
        target_stage_count=args.target_stage_count,
    )
    json_path, markdown_path = save_curriculum_strategy_grid_reports(
        payload,
        args.reports_dir,
    )
    print(f"MHDSRA2_CURRICULUM_STRATEGY_GRID_JSON={json_path}")
    print(f"MHDSRA2_CURRICULUM_STRATEGY_GRID_MARKDOWN={markdown_path}")
    stable_status = "true" if payload["summary"]["has_stable_target_strategy"] else "false"
    print(f"MHDSRA2_CURRICULUM_STRATEGY_STABLE={stable_status}")
    return payload


if __name__ == "__main__":
    main()
