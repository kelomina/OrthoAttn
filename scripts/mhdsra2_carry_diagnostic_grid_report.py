"""CLI report for MHDSRA2 carry-rule diagnostic grids."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dsra.application import (  # noqa: E402
    build_carry_diagnostic_dataset_specs,
    build_carry_diagnostic_grid_markdown,
    build_carry_diagnostic_grid_payload,
)
from src.dsra.application.arithmetic_emergence_service import (  # noqa: E402
    DEFAULT_CARRY_DIAGNOSTIC_EVAL_INTERVALS,
    DEFAULT_CARRY_DIAGNOSTIC_LAYER_COUNTS,
    DEFAULT_CARRY_DIAGNOSTIC_LEARNING_RATES,
    DEFAULT_CARRY_DIAGNOSTIC_STEP_BUDGETS,
    DEFAULT_CARRY_REPLAY_RATIO,
    DEFAULT_SEEDS,
    DEFAULT_STAGE_LOSS_WEIGHTS,
    TRAINING_STRATEGIES,
    run_one_carry_diagnostic_grid_point,
    serialize_carry_diagnostic_run,
    validate_stage_loss_weights,
    validate_training_strategy,
)
from src.dsra.report_utils import ensure_reports_dir, write_json, write_markdown  # noqa: E402


def parse_csv_ints(value: str) -> tuple[int, ...]:
    """Parse comma-separated integers from CLI input.

    中文说明:
    - 调用方 / Called by: `build_parser`.
    - 调用对象 / Calls: `str.split`, `int`.
    - 作用 / Purpose: 解析 layers、steps、eval intervals 与 seeds。
    - 变量 / Variables: `value` 是逗号分隔文本。
    - 接入 / Integration: argparse `type=` 回调。
    - 错误处理 / Error handling: 非整数输入由 `int` 抛出并由 argparse 展示。
    - 关键词 / Keywords:
      parse|csv|int|layers|steps|seeds|eval_interval|carry_diagnostic|解析|整数
    """
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_csv_floats(value: str) -> tuple[float, ...]:
    """Parse comma-separated floats from CLI input.

    中文说明:
    - 调用方 / Called by: `build_parser`.
    - 调用对象 / Calls: `str.split`, `float`.
    - 作用 / Purpose: 解析 learning rates 等浮点扫描维度。
    - 变量 / Variables: `value` 是逗号分隔文本。
    - 接入 / Integration: argparse `type=` 回调。
    - 错误处理 / Error handling: 非浮点输入由 `float` 抛出并由 argparse 展示。
    - 关键词 / Keywords:
      parse|csv|float|learning_rate|grid|carry_diagnostic|mhdsra2|cli|解析|浮点
    """
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def parse_csv_strs(value: str) -> tuple[str, ...]:
    """Parse comma-separated strategy names from CLI input.

    中文说明:
    - 调用方 / Called by: `build_parser`.
    - 调用对象 / Calls: `validate_training_strategy`.
    - 作用 / Purpose: 将训练策略列表规范化为支持的策略名。
    - 变量 / Variables: `value` 是逗号分隔策略文本。
    - 接入 / Integration: argparse `type=` 回调。
    - 错误处理 / Error handling: 非法策略由校验函数抛出。
    - 关键词 / Keywords:
      parse|strategy|baseline|carry_replay|combined|stage_weighted_loss|mhdsra2|cli|策略|解析
    """
    return tuple(
        validate_training_strategy(part.strip())
        for part in value.split(",")
        if part.strip()
    )


def parse_stage_loss_weights(value: str) -> dict[str, float]:
    """Parse stage loss weights from `stage=value` comma-separated text.

    中文说明:
    - 调用方 / Called by: `build_parser`.
    - 调用对象 / Calls: `validate_stage_loss_weights`.
    - 作用 / Purpose: 允许 CLI 覆盖默认 `unit_with_carry=2.0` 权重。
    - 变量 / Variables: `weights` 是阶段名到 loss 倍数的映射。
    - 接入 / Integration: `--stage-loss-weights unit_with_carry=2.0`。
    - 错误处理 / Error handling: 缺少 `=` 或非法权重会抛出 `ValueError`。
    - 关键词 / Keywords:
      parse|stage_loss_weights|loss|weight|unit_with_carry|cli|mhdsra2|diagnostic|权重|解析
    """
    weights: dict[str, float] = {}
    for part in value.split(","):
        stripped_part = part.strip()
        if not stripped_part:
            continue
        if "=" not in stripped_part:
            raise ValueError("stage loss weights must use stage=value entries.")
        stage_name, raw_weight = stripped_part.split("=", maxsplit=1)
        weights[stage_name.strip()] = float(raw_weight.strip())
    validate_stage_loss_weights(weights)
    return weights


def build_checkpoint_key(
    *,
    dataset_name: str,
    training_strategy: str,
    learning_rate: float,
    curriculum_eval_interval: int,
    max_steps_per_stage: int,
    num_layers: int,
    seed: int,
) -> str:
    """Build a stable checkpoint key for one carry diagnostic grid point.

    中文说明:
    - 调用方 / Called by: checkpoint read/write loop and tests.
    - 调用对象 / Calls: `json.dumps`.
    - 作用 / Purpose: 用确定性 key 支持 `--resume` 跳过已完成 cell。
    - 变量 / Variables: 所有入参共同唯一标识一个 dataset/strategy/lr/eval/step/layer/seed。
    - 接入 / Integration: JSONL 每行包含本 key 和对应 run row。
    - 错误处理 / Error handling: 纯序列化逻辑，不吞异常。
    - 关键词 / Keywords:
      checkpoint|key|resume|carry_diagnostic|learning_rate|strategy|seed|mhdsra2|键|恢复
    """
    return json.dumps(
        {
            "dataset_name": dataset_name,
            "training_strategy": training_strategy,
            "learning_rate": f"{learning_rate:.12g}",
            "curriculum_eval_interval": curriculum_eval_interval,
            "max_steps_per_stage": max_steps_per_stage,
            "num_layers": num_layers,
            "seed": seed,
        },
        sort_keys=True,
    )


def load_checkpoint_rows(checkpoint_path: Path) -> tuple[dict[str, object], ...]:
    """Load completed carry diagnostic rows from a checkpoint JSONL file.

    中文说明:
    - 调用方 / Called by: `main`, tests.
    - 调用对象 / Calls: `json.loads`, `Path.read_text`.
    - 作用 / Purpose: 支持长网格中断后恢复并聚合已有结果。
    - 变量 / Variables: `rows` 是已完成 run row 列表。
    - 接入 / Integration: `--resume` 开启时先读取 checkpoint。
    - 错误处理 / Error handling: 空文件返回空元组，坏 JSON 直接抛出。
    - 关键词 / Keywords:
      checkpoint|jsonl|load|resume|carry_diagnostic|rows|mhdsra2|report|加载|恢复
    """
    if not checkpoint_path.exists():
        return ()
    rows: list[dict[str, object]] = []
    for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if not isinstance(entry, dict):
            raise TypeError("checkpoint entries must be JSON objects.")
        row = entry["row"]
        if not isinstance(row, dict):
            raise TypeError("checkpoint entry row must be a JSON object.")
        rows.append(row)
    return tuple(rows)


def append_checkpoint_row(
    *,
    checkpoint_path: Path,
    checkpoint_key: str,
    row: Mapping[str, object],
) -> None:
    """Append one completed carry diagnostic row to checkpoint JSONL.

    中文说明:
    - 调用方 / Called by: `main`, tests.
    - 调用对象 / Calls: `Path.open`, `json.dumps`.
    - 作用 / Purpose: 每完成一个 grid cell 立即落盘，降低长运行损失。
    - 变量 / Variables: `checkpoint_key` 是可恢复去重 key, `row` 是序列化 run。
    - 接入 / Integration: CLI 主循环在每个点训练完成后调用。
    - 错误处理 / Error handling: 文件写入失败直接抛出，避免假装完成。
    - 关键词 / Keywords:
      checkpoint|append|jsonl|resume|carry_diagnostic|write|mhdsra2|cell|追加|检查点
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"key": checkpoint_key, "row": dict(row)}
    with checkpoint_path.open("a", encoding="utf-8") as checkpoint_file:
        checkpoint_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def checkpoint_keys_from_rows(rows: Sequence[Mapping[str, object]]) -> set[str]:
    """Rebuild checkpoint keys from loaded run rows.

    中文说明:
    - 调用方 / Called by: `main`.
    - 调用对象 / Calls: `build_checkpoint_key`.
    - 作用 / Purpose: 在 `--resume` 时根据已加载 rows 跳过已完成点。
    - 变量 / Variables: `rows` 是 checkpoint 中的序列化诊断行。
    - 接入 / Integration: 兼容历史 checkpoint 行中只保存 row 的情况。
    - 错误处理 / Error handling: 缺字段会抛出异常，防止错误跳过。
    - 关键词 / Keywords:
      checkpoint|keys|resume|rows|carry_diagnostic|learning_rate|seed|mhdsra2|键|去重
    """
    keys: set[str] = set()
    for row in rows:
        keys.add(
            build_checkpoint_key(
                dataset_name=str(row["dataset_name"]),
                training_strategy=str(row["training_strategy"]),
                learning_rate=float(row["learning_rate"]),
                curriculum_eval_interval=int(row["curriculum_eval_interval"]),
                max_steps_per_stage=int(row["max_steps_per_stage"]),
                num_layers=int(row["num_layers"]),
                seed=int(row["seed"]),
            )
        )
    return keys


def save_carry_diagnostic_grid_reports(
    payload: dict[str, object],
    reports_dir: Path | str,
) -> tuple[Path, Path]:
    """Write carry diagnostic JSON and Markdown reports.

    中文说明:
    - 调用方 / Called by: `main`, tests.
    - 调用对象 / Calls: `ensure_reports_dir`, `write_json`, `write_markdown`.
    - 作用 / Purpose: 写入 `reports/mhdsra2_carry_diagnostic_grid.{json,md}`。
    - 变量 / Variables: `payload` 是报告数据, `reports_dir` 是目标目录。
    - 接入 / Integration: 独立于 strategy grid 报告，不覆盖旧报告。
    - 错误处理 / Error handling: 文件写入异常直接抛出。
    - 关键词 / Keywords:
      save|reports|json|markdown|carry_diagnostic|mhdsra2|grid|artifact|保存|报告
    """
    resolved_reports_dir = ensure_reports_dir(reports_dir)
    json_path = resolved_reports_dir / "mhdsra2_carry_diagnostic_grid.json"
    markdown_path = resolved_reports_dir / "mhdsra2_carry_diagnostic_grid.md"
    write_json(json_path, payload)
    write_markdown(markdown_path, build_carry_diagnostic_grid_markdown(payload))
    return json_path, markdown_path


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for carry diagnostic grids.

    中文说明:
    - 调用方 / Called by: `main`, tests.
    - 调用对象 / Calls: `argparse.ArgumentParser`.
    - 作用 / Purpose: 暴露完整网格、checkpoint/resume 和强化策略参数。
    - 变量 / Variables: `parser` 是命令行解析器。
    - 接入 / Integration: `python scripts/mhdsra2_carry_diagnostic_grid_report.py --full-grid`。
    - 错误处理 / Error handling: 非法参数由 argparse 或解析函数报错退出。
    - 关键词 / Keywords:
      parser|cli|full_grid|resume|checkpoint|learning_rate|strategy|mhdsra2|入口|参数
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-grid", action="store_true", help="Use the default full grid.")
    parser.add_argument("--resume", action="store_true", help="Skip completed checkpoint rows.")
    parser.add_argument(
        "--layers",
        type=parse_csv_ints,
        default=DEFAULT_CARRY_DIAGNOSTIC_LAYER_COUNTS,
    )
    parser.add_argument(
        "--max-steps-per-stage-values",
        type=parse_csv_ints,
        default=DEFAULT_CARRY_DIAGNOSTIC_STEP_BUDGETS,
    )
    parser.add_argument(
        "--curriculum-eval-intervals",
        type=parse_csv_ints,
        default=DEFAULT_CARRY_DIAGNOSTIC_EVAL_INTERVALS,
    )
    parser.add_argument(
        "--learning-rates",
        type=parse_csv_floats,
        default=DEFAULT_CARRY_DIAGNOSTIC_LEARNING_RATES,
    )
    parser.add_argument(
        "--training-strategies",
        type=parse_csv_strs,
        default=TRAINING_STRATEGIES,
    )
    parser.add_argument("--seeds", type=parse_csv_ints, default=DEFAULT_SEEDS)
    parser.add_argument("--replay-ratio", type=float, default=0.75)
    parser.add_argument("--stage-patience", type=int, default=3)
    parser.add_argument("--carry-replay-ratio", type=float, default=DEFAULT_CARRY_REPLAY_RATIO)
    parser.add_argument(
        "--stage-loss-weights",
        type=parse_stage_loss_weights,
        default=dict(DEFAULT_STAGE_LOSS_WEIGHTS),
    )
    parser.add_argument("--target-stage-count", type=int, default=2)
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Checkpoint JSONL path. Defaults under reports dir.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    """Run the carry diagnostic grid report CLI.

    中文说明:
    - 调用方 / Called by: command line, tests and optional `scripts.main` integration.
    - 调用对象 / Calls: checkpoint helpers, `run_one_carry_diagnostic_grid_point`,
      `build_carry_diagnostic_grid_payload`, `save_carry_diagnostic_grid_reports`.
    - 作用 / Purpose: 执行可恢复完整网格并输出 carry 诊断报告。
    - 变量 / Variables: `rows` 是已完成或新完成 run rows, `completed_keys` 用于 resume。
    - 接入 / Integration: `--resume` 支持长运行中断后继续。
    - 错误处理 / Error handling: 训练、checkpoint、报告写入失败均直接抛出。
    - 关键词 / Keywords:
      main|carry_diagnostic|full_grid|resume|checkpoint|learning_rate|strategy|mhdsra2|运行|诊断
    """
    args = build_parser().parse_args(argv)
    reports_dir = ensure_reports_dir(args.reports_dir)
    checkpoint_path = (
        args.checkpoint_path
        if args.checkpoint_path is not None
        else reports_dir / "mhdsra2_carry_diagnostic_grid.checkpoint.jsonl"
    )
    rows = list(load_checkpoint_rows(checkpoint_path)) if args.resume else []
    completed_keys = checkpoint_keys_from_rows(rows) if args.resume else set()
    dataset_specs = build_carry_diagnostic_dataset_specs()
    for dataset_spec in dataset_specs:
        dataset_spec.validate_training_scope()
        for training_strategy in args.training_strategies:
            normalized_strategy = validate_training_strategy(training_strategy)
            for learning_rate in args.learning_rates:
                for curriculum_eval_interval in args.curriculum_eval_intervals:
                    for max_steps_per_stage in args.max_steps_per_stage_values:
                        for num_layers in args.layers:
                            for seed in args.seeds:
                                checkpoint_key = build_checkpoint_key(
                                    dataset_name=dataset_spec.name,
                                    training_strategy=normalized_strategy,
                                    learning_rate=learning_rate,
                                    curriculum_eval_interval=curriculum_eval_interval,
                                    max_steps_per_stage=max_steps_per_stage,
                                    num_layers=num_layers,
                                    seed=seed,
                                )
                                if checkpoint_key in completed_keys:
                                    continue
                                diagnostic_run = run_one_carry_diagnostic_grid_point(
                                    dataset_spec=dataset_spec,
                                    training_strategy=normalized_strategy,
                                    learning_rate=learning_rate,
                                    curriculum_eval_interval=curriculum_eval_interval,
                                    max_steps_per_stage=max_steps_per_stage,
                                    num_layers=num_layers,
                                    seed=seed,
                                    replay_ratio=args.replay_ratio,
                                    stage_patience=args.stage_patience,
                                    carry_replay_ratio=args.carry_replay_ratio,
                                    stage_loss_weights=args.stage_loss_weights,
                                )
                                row = serialize_carry_diagnostic_run(diagnostic_run)
                                append_checkpoint_row(
                                    checkpoint_path=checkpoint_path,
                                    checkpoint_key=checkpoint_key,
                                    row=row,
                                )
                                rows.append(row)
                                completed_keys.add(checkpoint_key)
    payload = build_carry_diagnostic_grid_payload(
        run_rows=rows,
        layer_counts=args.layers,
        max_steps_per_stage_values=args.max_steps_per_stage_values,
        curriculum_eval_intervals=args.curriculum_eval_intervals,
        learning_rates=args.learning_rates,
        training_strategies=args.training_strategies,
        seeds=args.seeds,
        replay_ratio=args.replay_ratio,
        stage_patience=args.stage_patience,
        carry_replay_ratio=args.carry_replay_ratio,
        stage_loss_weights=args.stage_loss_weights,
        target_stage_count=args.target_stage_count,
        checkpoint_path=str(checkpoint_path),
        resume_supported=True,
    )
    json_path, markdown_path = save_carry_diagnostic_grid_reports(payload, reports_dir)
    print(f"MHDSRA2_CARRY_DIAGNOSTIC_GRID_JSON={json_path}")
    print(f"MHDSRA2_CARRY_DIAGNOSTIC_GRID_MARKDOWN={markdown_path}")
    print(f"MHDSRA2_CARRY_DIAGNOSTIC_CHECKPOINT={checkpoint_path}")
    print(f"MHDSRA2_CARRY_DIAGNOSTIC_RUNS={len(rows)}")
    return payload


if __name__ == "__main__":
    main()
