from __future__ import annotations

from scripts.ablation_study import summarize_ablation_runs


def test_summarize_ablation_runs_keeps_all_lr_seed_rows() -> None:
    """Ensure ablation summaries cannot hide losing `(lr, seed)` runs.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `summarize_ablation_runs`.
    - 作用 / Purpose: 防止消融实验只报告最优 learning rate 或单次最佳 seed。
    - 变量 / Variables: `runs` 模拟两个学习率、两个 seed 的完整实验矩阵。
    - 接入 / Integration: 保护 `scripts.ablation_study.run_ablation` 的报告结构。
    - 错误处理 / Error handling: 断言失败直接暴露 cherry-pick 回归。
    - 副作用 / Side effects: 无。
    """
    runs = [
        {"lr": 1e-3, "seed": 101, "final_eval_acc": 0.8, "best_eval_acc": 0.9, "final_eval_loss": 0.4, "best_eval_loss": 0.3},
        {"lr": 1e-3, "seed": 202, "final_eval_acc": 0.6, "best_eval_acc": 0.7, "final_eval_loss": 0.6, "best_eval_loss": 0.5},
        {"lr": 1e-4, "seed": 101, "final_eval_acc": 0.7, "best_eval_acc": 0.8, "final_eval_loss": 0.5, "best_eval_loss": 0.4},
        {"lr": 1e-4, "seed": 202, "final_eval_acc": 0.7, "best_eval_acc": 0.8, "final_eval_loss": 0.5, "best_eval_loss": 0.4},
    ]

    summary = summarize_ablation_runs(runs)

    assert summary["runs"] == runs
    assert len(summary["by_lr"]) == 2
    lr_1e3 = next(row for row in summary["by_lr"] if row["lr"] == 1e-3)
    assert lr_1e3["seeds"] == [101, 202]
    assert lr_1e3["final_eval_acc_mean"] == 0.7
    assert lr_1e3["final_eval_acc_std"] == 0.10000000000000003
    assert summary["best_single_run"]["final_eval_acc"] == 0.8

