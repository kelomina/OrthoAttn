"""CLI report for MHDSRA2 decimal arithmetic emergence."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dsra.application import (  # noqa: E402
    build_layer_emergence_markdown,
    build_layer_emergence_payload,
)
from src.dsra.application.arithmetic_emergence_service import (  # noqa: E402
    CURRICULUM_STAGE_EXACT_MATCH_THRESHOLD,
    DEFAULT_ARITHMETIC_EMERGENCE_LEARNING_RATE,
    DEFAULT_ARITHMETIC_EMERGENCE_MAX_STEPS_PER_STAGE,
    DEFAULT_ARITHMETIC_EMERGENCE_DEVICE,
    DEFAULT_ARITHMETIC_EMERGENCE_REPLAY_RATIO,
    DEFAULT_ARITHMETIC_EMERGENCE_STAGE_PATIENCE,
    DEFAULT_CURRICULUM_EVAL_INTERVAL,
    DEFAULT_LAYER_COUNTS,
    DEFAULT_SEEDS,
)
from src.dsra.report_utils import ensure_reports_dir, write_json, write_markdown  # noqa: E402


def parse_csv_ints(value: str) -> tuple[int, ...]:
    """Parse a comma-separated integer list from CLI input.

    中文说明:
    - 调用方 / Called by: `build_parser`.
    - 调用对象 / Calls: `str.split`, `int`.
    - 作用 / Purpose: 支持命令行传入层数和 seed 列表。
    - 变量 / Variables: `value` 是逗号分隔字符串, 返回值是整数元组。
    - 接入 / Integration: argparse `type=` 参数直接使用本函数。
    - 错误处理 / Error handling: 非整数由 `int` 抛出 `ValueError`。
    - 关键词 / Keywords:
      cli|parse|integers|layers|seeds|csv|argparse|mhdsra2|arithmetic|解析
    """
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def save_layer_emergence_reports(payload: dict[str, object], reports_dir: Path | str) -> tuple[Path, Path]:
    """Write arithmetic emergence JSON and Markdown reports.

    中文说明:
    - 调用方 / Called by: `main`, tests, `scripts.main.run_mhdsra2_layer_emergence`.
    - 调用对象 / Calls: `ensure_reports_dir`, `write_json`, `write_markdown`,
      `build_layer_emergence_markdown`.
    - 作用 / Purpose: 将十进制加法规律涌现实验正式写入 `reports/`。
    - 变量 / Variables:
      `payload` 是报告数据, `reports_dir` 是目标目录, `json_path/md_path` 是产物路径。
    - 接入 / Integration: 统一产物名供 run summary 和测试引用。
    - 错误处理 / Error handling: 文件系统错误直接抛出, 不静默忽略。
    - 关键词 / Keywords:
      save|reports|json|markdown|arithmetic|emergence|mhdsra2|artifact|write|报告
    """
    resolved_reports_dir = ensure_reports_dir(reports_dir)
    json_path = resolved_reports_dir / "mhdsra2_layer_emergence_curve.json"
    markdown_path = resolved_reports_dir / "mhdsra2_layer_emergence_curve.md"
    write_json(json_path, payload)
    write_markdown(markdown_path, build_layer_emergence_markdown(payload))
    return json_path, markdown_path


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the arithmetic emergence report.

    中文说明:
    - 调用方 / Called by: `main`, tests.
    - 调用对象 / Calls: `argparse.ArgumentParser`, `parse_csv_ints`.
    - 作用 / Purpose: 暴露层数、seed、训练步数、标准注意力参考和报告目录配置。
    - 变量 / Variables: `parser` 是命令行解析器。
    - 接入 / Integration: `python scripts/mhdsra2_layer_emergence_report.py`.
    - 错误处理 / Error handling: 非法参数由 argparse 输出错误并退出。
    - 关键词 / Keywords:
      parser|cli|layers|seeds|training_steps|reports|mhdsra2|arithmetic|argparse|入口
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=parse_csv_ints, default=DEFAULT_LAYER_COUNTS)
    parser.add_argument("--seeds", type=parse_csv_ints, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--max-steps-per-stage",
        type=int,
        default=DEFAULT_ARITHMETIC_EMERGENCE_MAX_STEPS_PER_STAGE,
        help="Maximum training steps allowed for each curriculum stage before stopping.",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=None,
        help="Deprecated alias for --max-steps-per-stage.",
    )
    parser.add_argument(
        "--curriculum-eval-interval",
        type=int,
        default=DEFAULT_CURRICULUM_EVAL_INTERVAL,
        help="Evaluate stage EM every N active-stage training steps.",
    )
    parser.add_argument(
        "--stage-threshold",
        type=float,
        default=CURRICULUM_STAGE_EXACT_MATCH_THRESHOLD,
        help="Stage exact-match threshold required to advance to the next curriculum stage.",
    )
    parser.add_argument(
        "--replay-ratio",
        type=float,
        default=DEFAULT_ARITHMETIC_EMERGENCE_REPLAY_RATIO,
        help="Fraction of active-stage training steps assigned to cumulative replay.",
    )
    parser.add_argument(
        "--stage-patience",
        type=int,
        default=DEFAULT_ARITHMETIC_EMERGENCE_STAGE_PATIENCE,
        help="Consecutive stage evaluations required before advancing or stopping.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_ARITHMETIC_EMERGENCE_LEARNING_RATE,
        help="AdamW learning rate for arithmetic emergence training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_ARITHMETIC_EMERGENCE_DEVICE,
        choices=("auto", "cpu", "cuda"),
        help="Torch device for arithmetic emergence training.",
    )
    parser.add_argument(
        "--skip-standard-baseline",
        action="store_true",
        help="Skip StandardAttentionModel reference rows; MHDSRA2 success criteria are unchanged.",
    )
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    """Run the arithmetic emergence report CLI.

    中文说明:
    - 调用方 / Called by: command line and `scripts.main.run_mhdsra2_layer_emergence`.
    - 调用对象 / Calls: `build_parser`, `build_layer_emergence_payload`,
      `save_layer_emergence_reports`.
    - 作用 / Purpose: 运行十进制加法规律外推实验并写入正式报告。
    - 变量 / Variables: `argv` 是可选 CLI 参数, `payload` 是报告数据。
    - 接入 / Integration: 可通过 `python scripts/main.py mhdsra2_layer_emergence` 运行。
    - 错误处理 / Error handling: 参数、训练或写文件错误直接抛出/退出。
    - 关键词 / Keywords:
      main|cli|arithmetic|emergence|100+100|mhdsra2|report|json|markdown|运行
    """
    args = build_parser().parse_args(argv)
    max_steps_per_stage = (
        args.max_steps_per_stage if args.training_steps is None else args.training_steps
    )
    payload = build_layer_emergence_payload(
        layer_counts=args.layers,
        seeds=args.seeds,
        max_steps_per_stage=max_steps_per_stage,
        curriculum_eval_interval=args.curriculum_eval_interval,
        stage_threshold=args.stage_threshold,
        replay_ratio=args.replay_ratio,
        stage_patience=args.stage_patience,
        learning_rate=args.learning_rate,
        device=args.device,
        include_standard_baseline=not args.skip_standard_baseline,
    )
    json_path, markdown_path = save_layer_emergence_reports(payload, args.reports_dir)
    print(f"MHDSRA2_LAYER_EMERGENCE_JSON={json_path}")
    print(f"MHDSRA2_LAYER_EMERGENCE_MARKDOWN={markdown_path}")
    minimum_layers = payload["summary"]["minimum_arithmetic_emergent_layers"]
    minimum_layers_text = "null" if minimum_layers is None else str(minimum_layers)
    print(
        "MHDSRA2_ARITHMETIC_EMERGENT_MIN_LAYERS="
        f"{minimum_layers_text}"
    )
    return payload


if __name__ == "__main__":
    main()
