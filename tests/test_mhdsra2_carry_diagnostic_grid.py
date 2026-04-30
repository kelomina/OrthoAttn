import unittest
from pathlib import Path

from scripts.mhdsra2_carry_diagnostic_grid_report import (
    append_checkpoint_row,
    build_checkpoint_key,
    build_parser,
    load_checkpoint_rows,
    save_carry_diagnostic_grid_reports,
)
from src.dsra.application.arithmetic_emergence_service import (
    CARRY_REPLAY_TRAINING_STRATEGY,
    COMBINED_TRAINING_STRATEGY,
    STAGE_WEIGHTED_LOSS_TRAINING_STRATEGY,
    UNIT_WITH_CARRY_ONLY,
    UNIT_WITH_CARRY_STAGE,
    build_carry_diagnostic_grid_markdown,
    build_carry_diagnostic_grid_payload,
    build_curriculum_arithmetic_spec,
    build_unit_with_carry_only_spec,
    resolve_stage_loss_multiplier,
    run_one_arithmetic_emergence_curve,
    select_adaptive_curriculum_training_example,
    serialize_carry_diagnostic_run,
    run_one_carry_diagnostic_grid_point,
    validate_stage_loss_weights,
    validate_training_strategy,
)


class TestMHDSRA2CarryDiagnosticGrid(unittest.TestCase):
    """Regression tests for MHDSRA2 carry-rule diagnostic grids.

    中文说明:
    - 调用方 / Called by: `python -m unittest` and `pytest`.
    - 调用对象 / Calls: carry diagnostic app service, CLI parser and checkpoint helpers.
    - 作用 / Purpose: 保护进位规则诊断、强化策略、报告字段和恢复能力。
    - 变量 / Variables: `payload` 是小参数报告, `checkpoint_path` 是临时 JSONL。
    - 接入 / Integration: 测试文件放在 `tests/`，报告文件仍写入 `reports/`。
    - 错误处理 / Error handling: 参数错误和报告字段缺失通过断言暴露。
    - 关键词 / Keywords:
      unittest|carry_diagnostic|unit_with_carry|learning_rate|checkpoint|resume|mhdsra2|reports|测试|进位
    """

    def test_unit_with_carry_only_spec_contains_only_carry_examples(self) -> None:
        """Validate the isolated carry dataset contains only carry examples.

        中文说明:
        - 调用方 / Called by: `unittest`.
        - 调用对象 / Calls: `build_unit_with_carry_only_spec`,
          `ArithmeticRuleDatasetSpec.validate_training_scope`.
        - 作用 / Purpose: 确保 carry-only 诊断不混入无进位、两位数或 headline/OOD。
        - 变量 / Variables: `equations` 是训练等式集合。
        - 接入 / Integration: 修改诊断数据集时必须保持本测试通过。
        - 错误处理 / Error handling: 泄漏或样例错误会触发断言。
        - 关键词 / Keywords:
          unit_with_carry_only|dataset|carry|leakage|headline|ood|mhdsra2|test|数据集|进位
        """
        spec = build_unit_with_carry_only_spec()
        spec.validate_training_scope()
        equations = {example.equation for example in spec.training_examples}

        self.assertEqual(spec.name, UNIT_WITH_CARRY_ONLY)
        self.assertEqual(equations, {"5+5=10", "8+2=10", "9+1=10", "9+9=18"})
        self.assertFalse(spec.curriculum_stages)
        self.assertNotIn("1+1=2", equations)
        self.assertNotIn("10+10=20", equations)
        self.assertNotIn("100+100=200", equations)

    def test_strategy_sampling_and_loss_weighting_are_explicit(self) -> None:
        """Validate carry replay and weighted loss are controlled by strategy.

        中文说明:
        - 调用方 / Called by: `unittest`.
        - 调用对象 / Calls: `select_adaptive_curriculum_training_example`,
          `resolve_stage_loss_multiplier`.
        - 作用 / Purpose: 证明 carry replay 只在策略开启时影响 replay 分支。
        - 变量 / Variables: `carry_example` 是 carry replay 选出的样例。
        - 接入 / Integration: 调整采样策略时保护 baseline 兼容性。
        - 错误处理 / Error handling: 非法策略和权重由独立测试覆盖。
        - 关键词 / Keywords:
          carry_replay|weighted_loss|strategy|sampling|baseline|unit_with_carry|mhdsra2|test|策略|采样
        """
        spec = build_curriculum_arithmetic_spec()
        baseline_example = select_adaptive_curriculum_training_example(
            spec,
            active_stage_index=1,
            local_step=1,
            replay_ratio=0.75,
        )
        carry_example = select_adaptive_curriculum_training_example(
            spec,
            active_stage_index=1,
            local_step=1,
            replay_ratio=0.75,
            training_strategy=CARRY_REPLAY_TRAINING_STRATEGY,
            carry_replay_ratio=1.0,
        )

        self.assertIn(
            baseline_example,
            spec.curriculum_stages[0].examples,
        )
        self.assertIn(
            carry_example,
            spec.curriculum_stages[1].examples,
        )
        self.assertEqual(
            resolve_stage_loss_multiplier(
                dataset_spec=spec,
                example=carry_example,
                training_strategy=STAGE_WEIGHTED_LOSS_TRAINING_STRATEGY,
                stage_loss_weights={UNIT_WITH_CARRY_STAGE: 3.0},
            ),
            3.0,
        )
        self.assertEqual(
            resolve_stage_loss_multiplier(
                dataset_spec=spec,
                example=carry_example,
                training_strategy=CARRY_REPLAY_TRAINING_STRATEGY,
                stage_loss_weights={UNIT_WITH_CARRY_STAGE: 3.0},
            ),
            1.0,
        )

    def test_invalid_training_controls_raise(self) -> None:
        """Validate illegal lr, strategy and stage weights fail loudly.

        中文说明:
        - 调用方 / Called by: `unittest`.
        - 调用对象 / Calls: `validate_training_strategy`, `validate_stage_loss_weights`,
          `run_one_arithmetic_emergence_curve`.
        - 作用 / Purpose: 避免诊断实验静默使用错误优化配置。
        - 变量 / Variables: `spec` 是 carry-only 最小训练规约。
        - 接入 / Integration: 新增 CLI 参数时应同步增加非法输入测试。
        - 错误处理 / Error handling: 断言 `ValueError`。
        - 关键词 / Keywords:
          invalid|learning_rate|strategy|stage_loss_weights|error|mhdsra2|diagnostic|test|错误|校验
        """
        spec = build_unit_with_carry_only_spec()

        with self.assertRaises(ValueError):
            validate_training_strategy("unknown")
        with self.assertRaises(ValueError):
            validate_stage_loss_weights({UNIT_WITH_CARRY_STAGE: 0.0})
        with self.assertRaises(ValueError):
            run_one_arithmetic_emergence_curve(
                dataset_spec=spec,
                model_name="mhdsra2",
                seed=101,
                num_layers=1,
                max_steps_per_stage=2,
                curriculum_eval_interval=1,
                stage_threshold=0.95,
                replay_ratio=0.75,
                stage_patience=1,
                learning_rate=0.0,
            )

    def test_learning_rate_and_strategy_are_recorded_on_runs(self) -> None:
        """Validate optimizer controls are visible in serialized run output.

        中文说明:
        - 调用方 / Called by: `unittest`.
        - 调用对象 / Calls: `run_one_carry_diagnostic_grid_point`,
          `serialize_carry_diagnostic_run`.
        - 作用 / Purpose: 确认 learning_rate 不再是隐藏硬编码参数。
        - 变量 / Variables: `row` 是用于 checkpoint/report 的序列化结果。
        - 接入 / Integration: 报告和 checkpoint 复用这些字段。
        - 错误处理 / Error handling: 字段缺失通过断言失败暴露。
        - 关键词 / Keywords:
          learning_rate|strategy|serialize|checkpoint|optimizer|carry_diagnostic|mhdsra2|test|学习率|策略
        """
        diagnostic_run = run_one_carry_diagnostic_grid_point(
            dataset_spec=build_unit_with_carry_only_spec(),
            training_strategy=COMBINED_TRAINING_STRATEGY,
            learning_rate=0.01,
            curriculum_eval_interval=1,
            max_steps_per_stage=2,
            num_layers=1,
            seed=101,
            replay_ratio=0.75,
            stage_patience=1,
            carry_replay_ratio=0.5,
            stage_loss_weights={UNIT_WITH_CARRY_STAGE: 2.0},
        )
        row = serialize_carry_diagnostic_run(diagnostic_run)

        self.assertEqual(row["learning_rate"], 0.01)
        self.assertEqual(row["training_strategy"], COMBINED_TRAINING_STRATEGY)
        self.assertEqual(row["run"]["learning_rate"], 0.01)
        self.assertEqual(row["run"]["training_strategy"], COMBINED_TRAINING_STRATEGY)

    def test_carry_diagnostic_payload_report_and_checkpoint_fields(self) -> None:
        """Validate a tiny carry diagnostic report and checkpoint round trip.

        中文说明:
        - 调用方 / Called by: `unittest`.
        - 调用对象 / Calls: `build_carry_diagnostic_grid_payload`,
          `build_carry_diagnostic_grid_markdown`, checkpoint helpers and report saver.
        - 作用 / Purpose: 用小参数保护 JSON/Markdown 字段与 resume checkpoint 结构。
        - 变量 / Variables: `payload` 是小网格报告, `row` 是首条运行明细。
        - 接入 / Integration: 修改报告 schema 时必须同步本测试。
        - 错误处理 / Error handling: 字段缺失或 checkpoint 读取失败会断言失败。
        - 关键词 / Keywords:
          payload|report|checkpoint|resume|carry_exact_match_mean|target_retention_rate|mhdsra2|test|报告|恢复
        """
        payload = build_carry_diagnostic_grid_payload(
            layer_counts=(1,),
            max_steps_per_stage_values=(2,),
            curriculum_eval_intervals=(1,),
            learning_rates=(0.01,),
            training_strategies=("baseline", CARRY_REPLAY_TRAINING_STRATEGY),
            seeds=(101,),
            replay_ratio=0.75,
            stage_patience=1,
            carry_replay_ratio=0.5,
            stage_loss_weights={UNIT_WITH_CARRY_STAGE: 2.0},
            checkpoint_path="reports/test_mhdsra2_carry_diagnostic_grid.checkpoint.jsonl",
        )
        aggregate_row = payload["aggregates"][0]
        run_row = payload["runs"][0]
        markdown_text = "\n".join(build_carry_diagnostic_grid_markdown(payload))

        self.assertIn("learning_rate", run_row)
        self.assertIn("training_strategy", run_row)
        self.assertIn("unit_with_carry_only", {dataset["name"] for dataset in payload["datasets"]})
        self.assertIn("carry_exact_match_mean", aggregate_row)
        self.assertIn("target_retention_rate", aggregate_row)
        self.assertIn("checkpoint_path", payload["config"])
        self.assertTrue(payload["config"]["resume_supported"])
        self.assertIn("Carry EM Mean", markdown_text)

        project_root = Path(__file__).resolve().parents[1]
        reports_dir = project_root / "reports"
        json_path, markdown_path = save_carry_diagnostic_grid_reports(payload, reports_dir)
        self.assertEqual(json_path.name, "mhdsra2_carry_diagnostic_grid.json")
        self.assertEqual(markdown_path.name, "mhdsra2_carry_diagnostic_grid.md")

        checkpoint_path = reports_dir / "test_mhdsra2_carry_diagnostic_checkpoint.jsonl"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        checkpoint_key = build_checkpoint_key(
            dataset_name=str(run_row["dataset_name"]),
            training_strategy=str(run_row["training_strategy"]),
            learning_rate=float(run_row["learning_rate"]),
            curriculum_eval_interval=int(run_row["curriculum_eval_interval"]),
            max_steps_per_stage=int(run_row["max_steps_per_stage"]),
            num_layers=int(run_row["num_layers"]),
            seed=int(run_row["seed"]),
        )
        try:
            append_checkpoint_row(
                checkpoint_path=checkpoint_path,
                checkpoint_key=checkpoint_key,
                row=run_row,
            )
            loaded_rows = load_checkpoint_rows(checkpoint_path)
        finally:
            if checkpoint_path.exists():
                checkpoint_path.unlink()

        self.assertEqual(len(loaded_rows), 1)
        self.assertEqual(loaded_rows[0]["dataset_name"], run_row["dataset_name"])

    def test_cli_parser_exposes_full_grid_controls(self) -> None:
        """Validate the carry diagnostic CLI exposes requested scan dimensions.

        中文说明:
        - 调用方 / Called by: `unittest`.
        - 调用对象 / Calls: `build_parser`, argparse parsing helpers.
        - 作用 / Purpose: 保护完整网格命令所需参数和 `--resume` 开关。
        - 变量 / Variables: `args` 是解析后的命名空间。
        - 接入 / Integration: 回归命令直接依赖这些参数。
        - 错误处理 / Error handling: argparse 负责非法输入。
        - 关键词 / Keywords:
          cli|parser|full_grid|resume|learning_rates|strategies|stage_loss_weights|mhdsra2|参数|完整网格
        """
        args = build_parser().parse_args(
            [
                "--full-grid",
                "--resume",
                "--layers",
                "4,8,16",
                "--max-steps-per-stage-values",
                "128,256",
                "--curriculum-eval-intervals",
                "4,8,16",
                "--learning-rates",
                "0.003,0.01,0.03",
                "--training-strategies",
                "baseline,carry_replay,stage_weighted_loss,combined",
                "--stage-loss-weights",
                "unit_with_carry=2.0",
            ]
        )

        self.assertTrue(args.full_grid)
        self.assertTrue(args.resume)
        self.assertEqual(args.layers, (4, 8, 16))
        self.assertEqual(args.max_steps_per_stage_values, (128, 256))
        self.assertEqual(args.curriculum_eval_intervals, (4, 8, 16))
        self.assertEqual(args.learning_rates, (0.003, 0.01, 0.03))
        self.assertEqual(
            args.training_strategies,
            (
                "baseline",
                "carry_replay",
                "stage_weighted_loss",
                "combined",
            ),
        )
        self.assertEqual(args.stage_loss_weights, {UNIT_WITH_CARRY_STAGE: 2.0})


if __name__ == "__main__":
    unittest.main()
