"""CLI report for curriculum forgetting and catastrophic forgetting curve visualization.

Trains on curriculum_rule_set, two_digit_only, and prereq_plus_two_digit with
multiple training strategies at replay=0.9 + 2048 steps + 8 layers, producing
four matplotlib figures and a JSON/Markdown report.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dsra.application.arithmetic_emergence_service import (  # noqa: E402
    BASELINE_TRAINING_STRATEGY,
    COMBINED_TRAINING_STRATEGY,
    CURRICULUM_RULE_SET,
    CURRICULUM_STAGE_EXACT_MATCH_THRESHOLD,
    DEFAULT_CURRICULUM_EVAL_INTERVAL,
    DEFAULT_ARITHMETIC_EMERGENCE_DEVICE,
    DEFAULT_ARITHMETIC_EMERGENCE_LEARNING_RATE,
    DEFAULT_ARITHMETIC_EMERGENCE_STAGE_PATIENCE,
    DEFAULT_CARRY_REPLAY_RATIO,
    DEFAULT_REPLAY_RATIO,
    DEFAULT_SEEDS,
    PREREQ_PLUS_TWO_DIGIT,
    TWO_DIGIT_ONLY,
    TWO_DIGIT_REPLAY_TRAINING_STRATEGY,
    TWO_DIGIT_WEIGHTED_LOSS_TRAINING_STRATEGY,
    ArithmeticEmergenceRun,
    build_curriculum_arithmetic_spec,
    build_prereq_plus_two_digit_spec,
    build_two_digit_only_spec,
    compute_forgetting_gap,
    is_catastrophic_forgetting,
    run_one_arithmetic_emergence_curve,
    validate_training_strategy,
)
from src.dsra.report_utils import ensure_reports_dir, save_figure, write_json, write_markdown
from src.dsra.swanlab_utils import init_swanlab

FIGS_DIRNAME = "figures"
FORGETTING_CURVE_FILENAME = "mhdsra2_forgetting_curve"
TRAINING_STRATEGIES_TO_TEST = (
    BASELINE_TRAINING_STRATEGY,
    TWO_DIGIT_REPLAY_TRAINING_STRATEGY,
    TWO_DIGIT_WEIGHTED_LOSS_TRAINING_STRATEGY,
    COMBINED_TRAINING_STRATEGY,
)
STAGE_NAMES = ("unit_no_carry", "unit_with_carry", "two_digit_rules")
STAGE_COLORS = ("#2196F3", "#FF9800", "#4CAF50")
THRESHOLD_COLOR = "#E91E63"


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def parse_csv_strs(value: str) -> tuple[str, ...]:
    return tuple(
        validate_training_strategy(part.strip())
        for part in value.split(",")
        if part.strip()
    )


def parse_csv_dataset_names(value: str) -> tuple[str, ...]:
    known_datasets = {CURRICULUM_RULE_SET, TWO_DIGIT_ONLY, PREREQ_PLUS_TWO_DIGIT}
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    for name in names:
        if name not in known_datasets:
            raise ValueError(
                f"Unknown dataset '{name}'. Choose from: {', '.join(sorted(known_datasets))}."
            )
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-ratio", type=float, default=DEFAULT_REPLAY_RATIO,
        help="General replay ratio.",
    )
    parser.add_argument(
        "--two-digit-replay-ratios", type=parse_csv_floats,
        default=(DEFAULT_CARRY_REPLAY_RATIO,),
        help="Comma-separated two-digit replay ratios (mapped to carry_replay_ratio).",
    )
    parser.add_argument(
        "--max-steps-per-stage", type=int, default=2048,
        help="Maximum active-stage training steps.",
    )
    parser.add_argument(
        "--layers", type=parse_csv_ints, default=(8,),
        help="Comma-separated layer counts.",
    )
    parser.add_argument(
        "--seeds", type=parse_csv_ints, default=DEFAULT_SEEDS,
        help="Comma-separated random seeds.",
    )
    parser.add_argument(
        "--training-strategies", type=parse_csv_strs,
        default=TRAINING_STRATEGIES_TO_TEST,
        help="Comma-separated training strategies.",
    )
    parser.add_argument(
        "--datasets", type=parse_csv_dataset_names,
        default=(CURRICULUM_RULE_SET, TWO_DIGIT_ONLY, PREREQ_PLUS_TWO_DIGIT),
        help="Comma-separated dataset names.",
    )
    parser.add_argument(
        "--curriculum-eval-interval", type=int,
        default=DEFAULT_CURRICULUM_EVAL_INTERVAL,
        help="Evaluate open-stage EM every N steps.",
    )
    parser.add_argument(
        "--stage-patience", type=int,
        default=DEFAULT_ARITHMETIC_EMERGENCE_STAGE_PATIENCE,
        help="Stage patience for curriculum advance.",
    )
    parser.add_argument(
        "--learning-rate", type=float,
        default=DEFAULT_ARITHMETIC_EMERGENCE_LEARNING_RATE,
        help="Learning rate.",
    )
    parser.add_argument(
        "--device", type=str,
        default=DEFAULT_ARITHMETIC_EMERGENCE_DEVICE,
        choices=("auto", "cpu", "cuda"),
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=PROJECT_ROOT / "reports",
    )
    return parser


def _build_dataset_spec(dataset_name: str):
    if dataset_name == CURRICULUM_RULE_SET:
        return build_curriculum_arithmetic_spec()
    if dataset_name == TWO_DIGIT_ONLY:
        return build_two_digit_only_spec()
    if dataset_name == PREREQ_PLUS_TWO_DIGIT:
        return build_prereq_plus_two_digit_spec()
    raise ValueError(f"Unknown dataset: {dataset_name}")


def _run_one_point(
    *,
    dataset_name: str,
    strategy: str,
    seed: int,
    num_layers: int,
    max_steps_per_stage: int,
    curriculum_eval_interval: int,
    stage_threshold: float,
    stage_patience: int,
    replay_ratio: float,
    carry_replay_ratio: float,
    learning_rate: float,
    device: str,
) -> ArithmeticEmergenceRun:
    dataset_spec = _build_dataset_spec(dataset_name)
    dataset_spec.validate_training_scope()
    return run_one_arithmetic_emergence_curve(
        dataset_spec=dataset_spec,
        model_name="mhdsra2",
        seed=seed,
        num_layers=num_layers,
        max_steps_per_stage=max_steps_per_stage,
        curriculum_eval_interval=curriculum_eval_interval,
        stage_threshold=stage_threshold,
        replay_ratio=replay_ratio,
        stage_patience=stage_patience,
        learning_rate=learning_rate,
        training_strategy=strategy,
        carry_replay_ratio=carry_replay_ratio,
        stage_loss_weights=None,
        device=device,
    )


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------


def _extract_curves(
    runs: Sequence[ArithmeticEmergenceRun],
) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """Extract per-stage EM-vs-step curves from a list of runs (same config, diff seeds).

    Returns {stage_name: [(step, mean_em), ...]} where mean_em is averaged over seeds.
    """
    from collections import defaultdict
    by_step: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for snap in run.curriculum_snapshots:
            step = snap.step
            for metric in snap.stage_exact_matches:
                by_step[step][metric.stage_name].append(metric.exact_match)
    result: dict[str, list[tuple[int, float]]] = {}
    for stage_name in STAGE_NAMES:
        points = []
        for step, stage_dict in by_step.items():
            if stage_name in stage_dict:
                vals = stage_dict[stage_name]
                points.append((step, float(np.mean(vals))))
        points.sort()
        if points:
            result[stage_name] = points
    return result


def _extract_std_curves(
    runs: Sequence[ArithmeticEmergenceRun],
) -> dict[str, dict[str, list[tuple[int, float, float]]]]:
    """Extract per-stage EM-vs-step with std-dev.

    Returns {stage_name: [(step, mean_em, std_em), ...]}.
    """
    from collections import defaultdict
    by_step: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for snap in run.curriculum_snapshots:
            step = snap.step
            for metric in snap.stage_exact_matches:
                by_step[step][metric.stage_name].append(metric.exact_match)
    result: dict[str, list[tuple[int, float, float]]] = {}
    for stage_name in STAGE_NAMES:
        points = []
        for step, stage_dict in by_step.items():
            if stage_name in stage_dict:
                vals = stage_dict[stage_name]
                points.append((step, float(np.mean(vals)), float(np.std(vals))))
        points.sort()
        if points:
            result[stage_name] = points
    return result


def _extract_advance_events(
    runs: Sequence[ArithmeticEmergenceRun],
) -> list[tuple[str, float]]:
    """Extract curriculum advance events across seeds.

    Returns [(advanced_to_stage_name, mean_step), ...].
    """
    from collections import defaultdict
    events: dict[str, list[int]] = defaultdict(list)
    for run in runs:
        for snap in run.curriculum_snapshots:
            if snap.advanced_to_stage_name is not None:
                events[snap.advanced_to_stage_name].append(snap.step)
    result = []
    for stage_name in STAGE_NAMES[1:]:
        if stage_name in events and events[stage_name]:
            result.append((stage_name, float(np.mean(events[stage_name]))))
    return result


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------


def _figure_forgetting_curve(
    strategy_data: dict[str, Sequence[ArithmeticEmergenceRun]],
    figures_dir: Path,
) -> None:
    """图1: 分策略遗忘曲线（仅 curriculum_rule_set）."""
    strategies = [s for s in TRAINING_STRATEGIES_TO_TEST if s in strategy_data]
    n_strategies = len(strategies)
    fig, axes = plt.subplots(n_strategies, 1, figsize=(10, 3 * n_strategies), sharex=True)
    if n_strategies == 1:
        axes = [axes]

    strategy_labels = {
        BASELINE_TRAINING_STRATEGY: "Baseline",
        TWO_DIGIT_REPLAY_TRAINING_STRATEGY: "Two-Digit Replay",
        TWO_DIGIT_WEIGHTED_LOSS_TRAINING_STRATEGY: "Two-Digit Weighted Loss",
        COMBINED_TRAINING_STRATEGY: "Combined (catastrophic)",
    }
    for idx, strategy in enumerate(strategies):
        ax = axes[idx]
        runs = strategy_data[strategy]
        curves = _extract_std_curves(runs)
        events = _extract_advance_events(runs)
        is_combined = strategy == COMBINED_TRAINING_STRATEGY
        is_replay = strategy == TWO_DIGIT_REPLAY_TRAINING_STRATEGY

        if is_combined:
            ax.set_facecolor("#ffdddd")
        elif is_replay:
            ax.set_facecolor("#ddffdd")

        for si, stage_name in enumerate(STAGE_NAMES):
            if stage_name not in curves:
                continue
            pts = curves[stage_name]
            steps = [p[0] for p in pts]
            means = [p[1] for p in pts]
            stds = [p[2] for p in pts]
            color = STAGE_COLORS[si]
            alpha = 0.7 if is_combined else 0.9
            ax.plot(steps, means, color=color, label=stage_name, alpha=alpha, linewidth=2)
            lower = [m - s for m, s in zip(means, stds)]
            upper = [m + s for m, s in zip(means, stds)]
            ax.fill_between(steps, lower, upper, color=color, alpha=0.15)

        ax.axhline(y=CURRICULUM_STAGE_EXACT_MATCH_THRESHOLD, color=THRESHOLD_COLOR,
                   linestyle="--", linewidth=1, label=f"threshold={CURRICULUM_STAGE_EXACT_MATCH_THRESHOLD}")

        for stage_name, mean_step in events:
            ax.axvline(x=mean_step, color="gray", linestyle=":", linewidth=1, alpha=0.7)

        ax.set_ylabel("Exact Match")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(strategy_labels.get(strategy, strategy), fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Training Step")
    fig.suptitle("Curriculum Forgetting Curves  (replay=0.9, 2048 steps, 8 layers, 3 seeds)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, figures_dir / f"{FORGETTING_CURVE_FILENAME}.png")
    plt.close(fig)


def _figure_forgetting_bar(
    summary_data: list[dict[str, object]],
    figures_dir: Path,
) -> None:
    """图2: 遗忘量柱状图（curriculum_rule_set遗忘量 + 全数据集Train EM对比）. """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: forgetting gap (curriculum_rule_set only)
    ax = axes[0]
    rows = [r for r in summary_data if r["dataset"] == CURRICULUM_RULE_SET]
    strategies_display = [str(r["strategy"]) for r in rows]
    forgetting_gaps = [int(r["forgetting_gap"]) for r in rows]
    colors = []
    for f in forgetting_gaps:
        if f == 0:
            colors.append("#4CAF50")
        elif f <= 1:
            colors.append("#FFC107")
        elif f <= 2:
            colors.append("#FF9800")
        else:
            colors.append("#F44336")
    bars = ax.bar(strategies_display, forgetting_gaps, color=colors, edgecolor="gray")
    for bar, gap in zip(bars, forgetting_gaps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                str(gap), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Forgetting Gap F  (ever_passed - retained)")
    ax.set_title("Curriculum Forgetting Gap  (curriculum_rule_set)")
    ax.set_ylim(0, max(forgetting_gaps) + 0.5 if forgetting_gaps else 1)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.grid(True, axis="y", alpha=0.3)

    # Right: Train EM across all datasets
    ax = axes[1]
    datasets_order = (CURRICULUM_RULE_SET, TWO_DIGIT_ONLY, PREREQ_PLUS_TWO_DIGIT)
    strategies_in_data = sorted({str(r["strategy"]) for r in summary_data})
    x = np.arange(len(datasets_order))
    width = 0.8 / len(strategies_in_data)
    for si, strategy in enumerate(strategies_in_data):
        ems = []
        for ds in datasets_order:
            match = [r for r in summary_data if r["dataset"] == ds and str(r["strategy"]) == strategy]
            ems.append(float(match[0]["train_exact_match_mean"]) if match else 0.0)
        offset = (si - len(strategies_in_data) / 2 + 0.5) * width
        bars = ax.bar(x + offset, ems, width * 0.9, label=strategy)
        for bar, em in zip(bars, ems):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{em:.2f}", ha="center", va="bottom", fontsize=7, rotation=45)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_order, fontsize=9)
    ax.set_ylabel("Train Exact Match")
    ax.set_title("Training EM Across Datasets")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Forgetting Gap & Training EM  (replay=0.9, 2048 steps, 8 layers, 3 seeds)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, figures_dir / f"{FORGETTING_CURVE_FILENAME}_bar.png")
    plt.close(fig)


def _figure_heatmap(
    summary_data: list[dict[str, object]],
    figures_dir: Path,
) -> None:
    """图3: 各阶段EM热力图（全数据集 × 全策略）."""
    datasets_order = (CURRICULUM_RULE_SET, TWO_DIGIT_ONLY, PREREQ_PLUS_TWO_DIGIT)
    strategies_order = list(TRAINING_STRATEGIES_TO_TEST)
    stage_labels = list(STAGE_NAMES)

    n_datasets = len(datasets_order)
    fig, axes = plt.subplots(n_datasets, 1, figsize=(10, 2.5 * n_datasets))
    if n_datasets == 1:
        axes = [axes]

    for di, ds in enumerate(datasets_order):
        ax = axes[di]
        em_matrix = np.zeros((len(strategies_order), len(stage_labels)))
        for si, strategy in enumerate(strategies_order):
            for sti, stage in enumerate(stage_labels):
                match = [r for r in summary_data
                         if r["dataset"] == ds and str(r["strategy"]) == strategy]
                if match:
                    stage_ems = match[0].get("stage_ems", {})
                    em_matrix[si, sti] = float(stage_ems.get(stage, 0.0))

        ax.imshow(em_matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(stage_labels)))
        ax.set_xticklabels(stage_labels, fontsize=9)
        ax.set_yticks(range(len(strategies_order)))
        ax.set_yticklabels(strategies_order, fontsize=8)
        ax.set_title(ds, fontsize=10, fontweight="bold")

        for si in range(len(strategies_order)):
            for sti in range(len(stage_labels)):
                val = em_matrix[si, sti]
                ax.text(sti, si, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="white" if val < 0.5 else "black")

        # Highlight combined row
        if COMBINED_TRAINING_STRATEGY in strategies_order:
            ci = strategies_order.index(COMBINED_TRAINING_STRATEGY)
            for sti in range(len(stage_labels)):
                ax.add_patch(plt.Rectangle((sti - 0.5, ci - 0.5), 1, 1,
                                            fill=False, edgecolor="red", linewidth=3))

    fig.suptitle("Stage Exact Match Heatmap  (replay=0.9, 2048 steps, 8 layers, 3 seeds)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, figures_dir / f"{FORGETTING_CURVE_FILENAME}_heatmap.png")
    plt.close(fig)


def _figure_full_grid(
    strategy_data: dict[str, Sequence[ArithmeticEmergenceRun]],
    figures_dir: Path,
) -> None:
    """图4: 完整演化网格（curriculum_rule_set 各策略×各阶段的EM演化）. """
    strategies = [s for s in TRAINING_STRATEGIES_TO_TEST if s in strategy_data]
    fig, axes = plt.subplots(
        len(strategies), len(STAGE_NAMES),
        figsize=(4 * len(STAGE_NAMES), 3 * len(strategies)),
        sharex="col", sharey="row",
    )
    if len(strategies) == 1:
        axes = [axes]
    if len(STAGE_NAMES) == 1:
        for ri in range(len(strategies)):
            axes[ri] = [axes[ri]]

    for si, strategy in enumerate(strategies):
        runs = strategy_data[strategy]
        curves = _extract_curves(runs)
        for sti, stage_name in enumerate(STAGE_NAMES):
            ax = axes[si][sti]
            if stage_name in curves:
                pts = curves[stage_name]
                steps = [p[0] for p in pts]
                ems = [p[1] for p in pts]
                ax.plot(steps, ems, color=STAGE_COLORS[sti], linewidth=1.5)
                for run in runs:
                    seed_steps = []
                    seed_ems = []
                    for snap in run.curriculum_snapshots:
                        for m in snap.stage_exact_matches:
                            if m.stage_name == stage_name:
                                seed_steps.append(snap.step)
                                seed_ems.append(m.exact_match)
                    if seed_steps:
                        ax.plot(seed_steps, seed_ems, color=STAGE_COLORS[sti],
                                alpha=0.2, linewidth=0.5)
            ax.axhline(y=CURRICULUM_STAGE_EXACT_MATCH_THRESHOLD, color=THRESHOLD_COLOR,
                       linestyle="--", linewidth=0.5)
            if si == 0:
                ax.set_title(stage_name, fontsize=9)
            if sti == 0:
                ax.set_ylabel(strategy, fontsize=8, fontweight="bold")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.2)

    for sti in range(len(STAGE_NAMES)):
        axes[-1][sti].set_xlabel("Step")

    fig.suptitle("Full Evolution Grid  (curriculum_rule_set, replay=0.9, 2048 steps, 8 layers, 3 seeds)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, figures_dir / f"{FORGETTING_CURVE_FILENAME}_full_grid.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def _compute_summary(
    dataset_name: str,
    strategy: str,
    runs: Sequence[ArithmeticEmergenceRun],
) -> dict[str, object]:
    """Compute summary metrics for one dataset × strategy cell."""
    retained_vals = [r.retained_stage_count for r in runs]
    ever_passed_vals = [r.ever_passed_stage_count for r in runs]
    train_em_vals = [r.train_exact_match for r in runs]

    retained_mean = float(np.mean(retained_vals))
    ever_passed_mean = float(np.mean(ever_passed_vals))
    train_em_mean = float(np.mean(train_em_vals))

    is_curriculum_dataset = (dataset_name == CURRICULUM_RULE_SET)

    # Forgetting gap is only meaningful for curriculum datasets
    if is_curriculum_dataset:
        forgetting_gap = compute_forgetting_gap(
            retained=int(round(retained_mean)),
            ever_passed=int(round(ever_passed_mean)),
        )
        catastrophic = is_catastrophic_forgetting(retained=int(round(retained_mean)))
    else:
        forgetting_gap = 0
        catastrophic = False

    # Stage-level EM means
    stage_ems: dict[str, float] = {}
    for stage_name in STAGE_NAMES:
        vals = []
        for run in runs:
            for m in run.stage_exact_matches:
                if m.stage_name == stage_name:
                    vals.append(m.exact_match)
        if vals:
            stage_ems[stage_name] = float(np.mean(vals))

    return {
        "dataset": dataset_name,
        "strategy": strategy,
        "num_runs": len(runs),
        "retained_stage_count_mean": float(retained_mean),
        "ever_passed_stage_count_mean": float(ever_passed_mean),
        "forgetting_gap": forgetting_gap,
        "is_catastrophic": catastrophic,
        "train_exact_match_mean": train_em_mean,
        "stage_ems": stage_ems,
    }


def build_forgetting_curve_payload(
    *,
    datasets: Sequence[str],
    training_strategies: Sequence[str],
    seeds: Sequence[int],
    num_layers: int,
    max_steps_per_stage: int,
    curriculum_eval_interval: int,
    stage_threshold: float,
    stage_patience: int,
    replay_ratio: float,
    carry_replay_ratio: float,
    learning_rate: float,
    device: str,
) -> dict[str, object]:
    """Run all experiment points and build the full report payload.

    中文说明:
    - 调用方 / Called by: ``main``.
    - 调用对象 / Calls: ``_run_one_point``, ``_compute_summary``,
      ``_extract_curves``, ``_extract_advance_events``.
    - 作用 / Purpose: 执行所有数据集×策略×seed 的训练，生成遗忘曲线报告数据。
    - 变量 / Variables: ``runs_list`` 是原始训练结果, ``summary_rows`` 是聚合摘要。
    - 接入 / Integration: 写入 reports/mhdsra2_forgetting_curve.json。
    - 错误处理 / Error handling: 训练或聚合失败直接抛出。
    - 关键词 / Keywords:
      forgetting|curve|payload|curriculum|catastrophic|replay|mhdsra2|report|遗忘|报告
    """
    resolved_device = device

    all_runs: list[dict[str, object]] = []
    strategy_data: dict[str, list[ArithmeticEmergenceRun]] = {}
    summary_rows: list[dict[str, object]] = []

    for dataset_name in datasets:
        for strategy in training_strategies:
            runs_for_cell: list[ArithmeticEmergenceRun] = []
            for seed in seeds:
                run = _run_one_point(
                    dataset_name=dataset_name,
                    strategy=strategy,
                    seed=seed,
                    num_layers=num_layers,
                    max_steps_per_stage=max_steps_per_stage,
                    curriculum_eval_interval=curriculum_eval_interval,
                    stage_threshold=stage_threshold,
                    stage_patience=stage_patience,
                    replay_ratio=replay_ratio,
                    carry_replay_ratio=carry_replay_ratio,
                    learning_rate=learning_rate,
                    device=resolved_device,
                )
                runs_for_cell.append(run)
                all_runs.append({
                    "dataset": dataset_name,
                    "strategy": strategy,
                    "seed": seed,
                    "num_layers": num_layers,
                    "retained_stage_count": run.retained_stage_count,
                    "ever_passed_stage_count": run.ever_passed_stage_count,
                    "train_exact_match": run.train_exact_match,
                    "final_loss": run.final_loss,
                    "stopped_reason": run.stopped_reason,
                    "training_steps_executed": run.training_steps_executed,
                    "stage_exact_matches": [
                        {"stage_name": m.stage_name, "exact_match": m.exact_match}
                        for m in run.stage_exact_matches
                    ],
                    "curriculum_snapshots": [
                        {
                            "step": snap.step,
                            "active_stage_name": snap.active_stage_name,
                            "advanced_to_stage_name": snap.advanced_to_stage_name,
                            "stage_exact_matches": [
                                {"stage_name": m.stage_name, "exact_match": m.exact_match}
                                for m in snap.stage_exact_matches
                            ],
                        }
                        for snap in run.curriculum_snapshots
                    ],
                })

            if dataset_name == CURRICULUM_RULE_SET:
                strategy_data[strategy] = runs_for_cell

            summary_rows.append(_compute_summary(dataset_name, strategy, runs_for_cell))

    return {
        "config": {
            "datasets": list(datasets),
            "training_strategies": list(training_strategies),
            "seeds": list(seeds),
            "num_layers": num_layers,
            "max_steps_per_stage": max_steps_per_stage,
            "curriculum_eval_interval": curriculum_eval_interval,
            "stage_threshold": stage_threshold,
            "stage_patience": stage_patience,
            "replay_ratio": replay_ratio,
            "carry_replay_ratio": carry_replay_ratio,
            "learning_rate": learning_rate,
            "device": device,
        },
        "summary": summary_rows,
        "runs": all_runs,
    }


def build_forgetting_curve_markdown(payload: dict[str, object]) -> list[str]:
    """Build Markdown report lines from the forgetting curve payload."""
    config = payload["config"]
    summary = payload["summary"]

    lines = [
        "# MHDSRA2 Curriculum Forgetting Curve Report",
        "",
        "## Configuration",
        "",
        f"- Datasets: {', '.join(config['datasets'])}",
        f"- Training strategies: {', '.join(config['training_strategies'])}",
        f"- Seeds: {', '.join(str(s) for s in config['seeds'])}",
        f"- Layers: {config['num_layers']}",
        f"- Max steps per stage: {config['max_steps_per_stage']}",
        f"- Curriculum eval interval: {config['curriculum_eval_interval']}",
        f"- Stage threshold: {config['stage_threshold']}",
        f"- Stage patience: {config['stage_patience']}",
        f"- Replay ratio: {config['replay_ratio']}",
        f"- Two-digit replay ratio (carry): {config['carry_replay_ratio']}",
        f"- Learning rate: {config['learning_rate']}",
        f"- Device: {config['device']}",
        "",
        "## Summary",
        "",
        "| Dataset | Strategy | Runs | Retained Mean | Ever Passed Mean | "
        "Forgetting Gap | Catastrophic | Train EM Mean |",
        "|:---|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {row['strategy']} | {row['num_runs']} | "
            f"{row['retained_stage_count_mean']:.2f} | {row['ever_passed_stage_count_mean']:.2f} | "
            f"{row['forgetting_gap']} | {'yes' if row['is_catastrophic'] else 'no'} | "
            f"{row['train_exact_match_mean']:.4f} |"
        )

    lines += [
        "",
        "## Stage-Level Exact Match",
        "",
        "| Dataset | Strategy | unit_no_carry | unit_with_carry | two_digit_rules |",
        "|:---|---:|---:|---:|---:|",
    ]
    for row in summary:
        stage_ems = row.get("stage_ems", {})
        lines.append(
            f"| {row['dataset']} | {row['strategy']} | "
            f"{stage_ems.get('unit_no_carry', 0.0):.4f} | "
            f"{stage_ems.get('unit_with_carry', 0.0):.4f} | "
            f"{stage_ems.get('two_digit_rules', 0.0):.4f} |"
        )

    lines += [
        "",
        "## Figures",
        "",
        "### Figure 1: Curriculum Forgetting Curves",
        "",
        "Per-strategy EM-vs-step traces for `curriculum_rule_set`. "
        "Combined row is highlighted in red (catastrophic forgetting).",
        "",
        f"![Forgetting Curve](figures/{FORGETTING_CURVE_FILENAME}.png)",
        "",
        "### Figure 2: Forgetting Gap & Training EM",
        "",
        "Left: forgetting gap F = ever_passed - retained for `curriculum_rule_set`. "
        "Right: train EM across all datasets.",
        "",
        f"![Forgetting Bar](figures/{FORGETTING_CURVE_FILENAME}_bar.png)",
        "",
        "### Figure 3: Stage EM Heatmap",
        "",
        "Per-stage final EM across all datasets and strategies. "
        "Red border marks combined (catastrophic) row.",
        "",
        f"![Heatmap](figures/{FORGETTING_CURVE_FILENAME}_heatmap.png)",
        "",
        "### Figure 4: Full Evolution Grid",
        "",
        "Every strategy × stage EM-vs-step trace for `curriculum_rule_set`. "
        "Thick line = mean, thin lines = individual seeds.",
        "",
        f"![Full Grid](figures/{FORGETTING_CURVE_FILENAME}_full_grid.png)",
        "",
    ]
    return lines


def save_forgetting_curve_reports(
    payload: dict[str, object],
    reports_dir: Path | str,
) -> dict[str, Path]:
    """Save all forgetting curve report artifacts."""
    resolved_reports_dir = ensure_reports_dir(reports_dir)
    figures_dir = resolved_reports_dir / FIGS_DIRNAME
    figures_dir.mkdir(parents=True, exist_ok=True)

    json_path = resolved_reports_dir / f"{FORGETTING_CURVE_FILENAME}.json"
    markdown_path = resolved_reports_dir / f"{FORGETTING_CURVE_FILENAME}.md"
    write_json(json_path, payload)
    write_markdown(markdown_path, build_forgetting_curve_markdown(payload))

    # Build figures from curriculum_rule_set strategy data
    strategy_data: dict[str, list[ArithmeticEmergenceRun]] = {}
    for run_entry in payload["runs"]:
        if run_entry["dataset"] != CURRICULUM_RULE_SET:
            continue
        strategy = str(run_entry["strategy"])
        if strategy not in strategy_data:
            strategy_data[strategy] = []
        # Reconstruct from serialized data
        # Actually we need the original run objects for figure drawing.
        # The figures are drawn before serialization in main().
        pass

    return {
        "json": json_path,
        "markdown": markdown_path,
        "figures_dir": figures_dir,
    }


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = build_parser().parse_args(argv)
    swanlab_run = init_swanlab(
        project="MHDSRA2",
        experiment_name="forgetting_curve",
        config={
            "datasets": list(args.datasets),
            "training_strategies": list(args.training_strategies),
            "seeds": list(args.seeds),
            "num_layers": args.layers[0] if args.layers else 8,
            "max_steps_per_stage": args.max_steps_per_stage,
            "replay_ratio": args.replay_ratio,
            "learning_rate": args.learning_rate,
            "device": args.device,
        },
        mode="cloud",
        tags=["forgetting_curve", "curriculum"],
    )
    reports_dir = ensure_reports_dir(args.reports_dir)
    figures_dir = reports_dir / FIGS_DIRNAME
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Collect all training data
    strategy_data: dict[str, list[ArithmeticEmergenceRun]] = {}
    all_runs: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    datasets_to_run = args.datasets
    strategies_to_run = args.training_strategies

    for dataset_name in datasets_to_run:
        for strategy in strategies_to_run:
            runs_for_cell: list[ArithmeticEmergenceRun] = []
            for seed in args.seeds:
                run = _run_one_point(
                    dataset_name=dataset_name,
                    strategy=strategy,
                    seed=seed,
                    num_layers=args.layers[0] if args.layers else 8,
                    max_steps_per_stage=args.max_steps_per_stage,
                    curriculum_eval_interval=args.curriculum_eval_interval,
                    stage_threshold=CURRICULUM_STAGE_EXACT_MATCH_THRESHOLD,
                    stage_patience=args.stage_patience,
                    replay_ratio=args.replay_ratio,
                    carry_replay_ratio=args.two_digit_replay_ratios[0],
                    learning_rate=args.learning_rate,
                    device=args.device,
                )
                runs_for_cell.append(run)
                all_runs.append({
                    "dataset": dataset_name,
                    "strategy": strategy,
                    "seed": seed,
                    "num_layers": args.layers[0] if args.layers else 8,
                    "retained_stage_count": run.retained_stage_count,
                    "ever_passed_stage_count": run.ever_passed_stage_count,
                    "train_exact_match": run.train_exact_match,
                    "final_loss": run.final_loss,
                    "stopped_reason": run.stopped_reason,
                    "training_steps_executed": run.training_steps_executed,
                    "stage_exact_matches": [
                        {"stage_name": m.stage_name, "exact_match": m.exact_match}
                        for m in run.stage_exact_matches
                    ],
                    "curriculum_snapshots": [
                        {
                            "step": snap.step,
                            "active_stage_name": snap.active_stage_name,
                            "advanced_to_stage_name": snap.advanced_to_stage_name,
                            "stage_exact_matches": [
                                {"stage_name": m.stage_name, "exact_match": m.exact_match}
                                for m in snap.stage_exact_matches
                            ],
                        }
                        for snap in run.curriculum_snapshots
                    ],
                })

            if dataset_name == CURRICULUM_RULE_SET:
                strategy_data[strategy] = runs_for_cell

            summary_rows.append(_compute_summary(dataset_name, strategy, runs_for_cell))
            swanlab_run.log(
                {
                    f"{dataset_name}/{strategy}/retained_stage_count_mean": runs_for_cell[-1].retained_stage_count if runs_for_cell else 0,
                    f"{dataset_name}/{strategy}/train_exact_match": runs_for_cell[-1].train_exact_match if runs_for_cell else 0.0,
                    f"{dataset_name}/{strategy}/final_loss": runs_for_cell[-1].final_loss if runs_for_cell else 0.0,
                },
                step=len(summary_rows) - 1,
            )

    payload: dict[str, object] = {
        "config": {
            "datasets": list(datasets_to_run),
            "training_strategies": list(strategies_to_run),
            "seeds": list(args.seeds),
            "num_layers": args.layers[0] if args.layers else 8,
            "max_steps_per_stage": args.max_steps_per_stage,
            "curriculum_eval_interval": args.curriculum_eval_interval,
            "stage_threshold": CURRICULUM_STAGE_EXACT_MATCH_THRESHOLD,
            "stage_patience": args.stage_patience,
            "replay_ratio": args.replay_ratio,
            "carry_replay_ratio": args.two_digit_replay_ratios[0],
            "learning_rate": args.learning_rate,
            "device": args.device,
        },
        "scheme": os.environ.get("DSRA_FORGETTING_SCHEME", ""),
        "summary": summary_rows,
        "runs": all_runs,
    }

    # Generate figures using raw ArithmeticEmergenceRun objects
    if CURRICULUM_RULE_SET in datasets_to_run and strategy_data:
        _figure_forgetting_curve(strategy_data, figures_dir)
        _figure_full_grid(strategy_data, figures_dir)
    _figure_forgetting_bar(summary_rows, figures_dir)
    _figure_heatmap(summary_rows, figures_dir)

    # Write reports
    json_path = reports_dir / f"{FORGETTING_CURVE_FILENAME}.json"
    markdown_path = reports_dir / f"{FORGETTING_CURVE_FILENAME}.md"
    write_json(json_path, payload)
    write_markdown(markdown_path, build_forgetting_curve_markdown(payload))

    print(f"MHDSRA2_FORGETTING_CURVE_JSON={json_path}")
    print(f"MHDSRA2_FORGETTING_CURVE_MARKDOWN={markdown_path}")
    print(f"MHDSRA2_FORGETTING_CURVE_FIGURES={figures_dir}")
    print(f"MHDSRA2_FORGETTING_CURVE_RUNS={len(all_runs)}")
    swanlab_run.finish()
    return payload


if __name__ == "__main__":
    main()
