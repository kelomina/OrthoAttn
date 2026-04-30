import unittest
from pathlib import Path

from scripts.mhdsra2_two_digit_diagnostic_grid_report import (
    append_checkpoint_row,
    build_checkpoint_key,
    build_parser,
    load_checkpoint_rows,
    save_two_digit_diagnostic_grid_reports,
)
from src.dsra.application.arithmetic_emergence_service import (
    BASELINE_TRAINING_STRATEGY,
    COMBINED_TRAINING_STRATEGY,
    PREREQ_PLUS_TWO_DIGIT,
    TWO_DIGIT_ONLY,
    TWO_DIGIT_REPLAY_TRAINING_STRATEGY,
    TWO_DIGIT_RULES_STAGE,
    TWO_DIGIT_WEIGHTED_LOSS_TRAINING_STRATEGY,
    build_prereq_plus_two_digit_spec,
    build_two_digit_diagnostic_grid_markdown,
    build_two_digit_diagnostic_grid_payload,
    build_two_digit_only_spec,
    resolve_stage_loss_multiplier,
    select_adaptive_curriculum_training_example,
    select_two_digit_diagnostic_dataset_specs,
    serialize_two_digit_diagnostic_run,
    run_one_two_digit_diagnostic_grid_point,
    validate_stage_loss_weights,
    validate_training_strategy,
)


class TestMHDSRA2TwoDigitDiagnosticGrid(unittest.TestCase):
    """Regression tests for MHDSRA2 two-digit-rule diagnostic grids.

    中文说明:
    - 调用方 / Called by: `python -m unittest` and `pytest`.
    - 调用对象 / Calls: two-digit diagnostic app service, CLI parser and checkpoint helpers.
    - 作用 / Purpose: 保护两位数规则诊断、强化策略、报告字段和恢复能力。
    - 变量 / Variables: `payload` 是小参数报告, `checkpoint_path` 是临时 JSONL。
    - 接入 / Integration: 测试文件放在 `tests/`, 报告文件仍写入 `reports/`。
    - 错误处理 / Error handling: 参数错误和报告字段缺失通过断言暴露。
    - 关键词 / Keywords:
      unittest|two_digit|diagnostic|learning_rate|checkpoint|resume|mhdsra2|reports|测试|两位数
    """

    def test_two_digit_only_spec_contains_only_two_digit_examples(self) -> None:
        """Validate the isolated two-digit dataset contains only two-digit examples.

        中文说明:
        - 调用方 / Called by: `unittest`.
        - 调用对象 / Calls: `build_two_digit_only_spec`,
          `ArithmeticRuleDatasetSpec.validate_training_scope`.
        - 作用 / Purpose: 确保 two-digit-only 诊断不混入无进位、进位或 headline/OOD。
        - 变量 / Variables: `equations` 是训练等式集合。
        - 接入 / Integration: 修改诊断数据集时必须保持本测试通过。
        - 错误处理 / Error handling: 泄漏或样例错误会触发断言。
        - 关键词 / Keywords:
          two_digit_only|dataset|leakage|headline|ood|mhdsra2|test|rules|数据集|两位数
        """
        spec = build_two_digit_only_spec()
        spec.validate_training_scope()
        equations = {example.equation for example in spec.training_examples}

        self.assertEqual(spec.name, TWO_DIGIT_ONLY)
        self.assertEqual(equations, {"10+10=20", "11+11=22", "12+12=24"})
        self.assertFalse(spec.curriculum_stages)
        self.assertEqual([stage.name for stage in spec.diagnostic_stages], [TWO_DIGIT_RULES_STAGE])
        self.assertNotIn("1+1=2", equations)
        self.assertNotIn("9+1=10", equations)
        self.assertNotIn("100+100=200", equations)

    def test_prereq_plus_two_digit_is_non_adaptive_but_keeps_diagnostics(self) -> None:
        """Validate prereq-plus-two-digit mixes stages without adaptive curriculum.

        中文说明:
        - 调用方 / Called by: `unittest`.
        - 调用对象 / Calls: `build_prereq_plus_two_digit_spec`.
        - 作用 / Purpose: 确保混合诊断训练集包含三个阶段样例但不启用 curriculum。
        - 变量 / Variables: `diagnostic_stage_names` 是诊断阶段名列表。
        - 接入 / Integration: 用于区分混合训练和 adaptive 推进问题。
        - 错误处理 / Error handling: 阶段边界错误通过断言暴露。
        - 关键词 / Keywords:
          prereq|two_digit|mixed|non_adaptive|diagnostic|mhdsra2|test|dataset|混合|阶段
        """
        spec = build_prereq_plus_two_digit_spec()
        diagnostic_stage_names = [stage.name for stage in spec.diagnostic_stages]
        equations = {example.equation for example in spec.training_examples}

        self.assertEqual(spec.name, PREREQ_PLUS_TWO_DIGIT)
        self.assertFalse(spec.curriculum_stages)
        self.assertEqual(
            diagnostic_stage_names,
            ["unit_no_carry", "unit_with_carry", "two_digit_rules"],
        )
        self.assertIn("1+1=2", equations)
        self.assertIn("9+1=10", equations)
        self.assertIn("10+10=20", equations)

    def test_two_digit_strategy_sampling_and_weighting_are_explicit(self) -> None:
        """Validate two-digit replay and weighted loss are strategy-controlled.

        中文说明:
        - 调用方 / Called by: `unittest`.
        - 调用对象 / Calls: `select_adaptive_curriculum_training_example`,
          `resolve_stage_loss_multiplier`.
        - 作用 / Purpose: 证明 two-digit replay 和加权 loss 不改变 baseline 行为。
        - 变量 / Variables: `two_digit_example` 是强化策略选出的样例。
        - 接入 / Integration: 调整采样策略时保持 baseline 兼容。
        - 错误处理 / Error handling: 非法策略由独立测试覆盖。
        - 关键词 / Keywords:
          two_digit_replay|weighted_loss|strategy|sampling|baseline|mhdsra2|test|策略|采样|两位数
        """
        spec = build_prereq_plus_two_digit_spec()
        curriculum_spec = spec.__class__(
            name=spec.name,
            training_examples=spec.training_examples,
            headline_example=spec.headline_example,
            ood_examples=spec.ood_examples,
            curriculum_stages=spec.diagnostic_stages,
        )
        baseline_example = select_adaptive_curriculum_training_example(
            curriculum_spec,
            active_stage_index=2,
            local_step=1,
            replay_ratio=0.75,
            training_strategy=BASELINE_TRAINING_STRATEGY,
        )
        two_digit_example = select_adaptive_curriculum_training_example(
            curriculum_spec,
            active_stage_index=2,
            local_step=1,
            replay_ratio=0.75,
            training_strategy=TWO_DIGIT_REPLAY_TRAINING_STRATEGY,
            carry_replay_ratio=1.0,
        )

        self.assertNotIn(baseline_example, curriculum_spec.curriculum_stages[2].examples)
        self.assertIn(two_digit_example, curriculum_spec.curriculum_stages[2].examples)
        self.assertEqual(
            resolve_stage_loss_multiplier(
                dataset_spec=curriculum_spec,
                example=two_digit_example,
                training_strategy=TWO_DIGIT_WEIGHTED_LOSS_TRAINING_STRATEGY,
                stage_loss_weights={TWO_DIGIT_RULES_STAGE: 3.0},
            ),
            3.0,
        )
        self.assertEqual(
            resolve_stage_loss_multiplier(
                dataset_spec=curriculum_spec,
                example=two_digit_example,
                training_strategy=TWO_DIGIT_REPLAY_TRAINING_STRATEGY,
                stage_loss_weights={TWO_DIGIT_RULES_STAGE: 3.0},
            ),
            1.0,
        )

    def test_invalid_two_digit_controls_raise(self) -> None:
        """Validate illegal two-digit diagnostic controls fail loudly.

        中文说明:
        - 调用方 / Called by: `unittest`.
        - 调用对象 / Calls: `validate_training_strategy`, `validate_stage_loss_weights`,
          `build_two_digit_diagnostic_grid_payload`.
        - 作用 / Purpose: 避免诊断实验静默使用错误配置。
        - 变量 / Variables: 无额外状态。
        - 接入 / Integration: 新增 CLI 参数时应同步覆盖非法输入。
        - 错误处理 / Error handling: 断言 `ValueError`。
        - 关键词 / Keywords:
          invalid|strategy|stage_loss_weights|empty_grid|device|mhdsra2|diagnostic|test|错误|两位数
        """
        with self.assertRaises(ValueError):
            validate_training_strategy("unknown")
        with self.assertRaises(ValueError):
            validate_stage_loss_weights({TWO_DIGIT_RULES_STAGE: 0.0})
        with self.assertRaises(ValueError):
            build_two_digit_diagnostic_grid_payload(layer_counts=())
        with self.assertRaises(ValueError):
            build_two_digit_diagnostic_grid_payload(device="gpu")
        with self.assertRaises(ValueError):
            select_two_digit_diagnostic_dataset_specs(())
        with self.assertRaises(ValueError):
            select_two_digit_diagnostic_dataset_specs(("unknown_dataset",))

    def test_two_digit_dataset_filter_limits_payload_scope(self) -> None:
        """Validate dataset filtering limits two-digit diagnostic payload rows.

        中文说明:
        - 调用方 / Called by: `unittest`.
        - 调用对象 / Calls: `build_two_digit_diagnostic_grid_payload`,
          `select_two_digit_diagnostic_dataset_specs`.
        - 作用 / Purpose: 保护 `--datasets two_digit_only` 中等网格只运行隔离两位数数据集。
        - 参数 / Parameters: 无外部参数; 测试内部使用小步数避免长训练。
        - 返回 / Returns: 无返回值; 通过断言表达预期。
        - 变量 / Variables: `payload` 是过滤后的报告数据, `dataset_names` 是输出数据集名称集合。
        - 接入 / Integration: CLI 解析 `--datasets` 后会把同一参数传入 payload builder。
        - 错误处理 / Error handling: 若过滤失效, 数据集名称和 run 明细断言会失败。
        - 副作用 / Side effects: 仅在内存中执行短训练, 不写报告文件。
        - 事务边界 / Transaction boundary: 不涉及 Unit of Work 或事务。
        - 并发与幂等 / Concurrency and idempotency: 固定 seed, 可重复运行。
        - 中文关键词: 数据集, 过滤, 两位数, 诊断, payload, 中等网格, 隔离, 训练, 测试, 校验

        English documentation:
        Function name:
            test_two_digit_dataset_filter_limits_payload_scope
        Purpose:
            Ensure the dataset filter restricts payload datasets and run rows.
        Called by:
            `unittest` and `pytest`.
        Calls:
            `build_two_digit_diagnostic_grid_payload`, `select_two_digit_diagnostic_dataset_specs`.
        Parameters:
            None; the test uses tiny in-memory training parameters.
        Returns:
            None; assertions define success.
        Internal variables:
            - payload: filtered diagnostic report data.
            - dataset_names: dataset names rendered in the payload.
        Integration:
            Mirrors CLI use after `--datasets` parsing.
        Error handling:
            Failed filtering is exposed by assertions.
        Side effects:
            In-memory short training only.
        Transaction boundary:
            No transaction or Unit of Work is used.
        Concurrency and idempotency:
            Deterministic with fixed seeds.
        English keywords:
            dataset, filter, two_digit, diagnostic, payload, medium_grid, isolated, training, test, validation
        """
        selected_specs = select_two_digit_diagnostic_dataset_specs((TWO_DIGIT_ONLY,))
        payload = build_two_digit_diagnostic_grid_payload(
            datasets=(TWO_DIGIT_ONLY,),
            layer_counts=(1,),
            max_steps_per_stage_values=(2,),
            learning_rates=(0.01,),
            training_strategies=(BASELINE_TRAINING_STRATEGY,),
            seeds=(101,),
            device="cpu",
        )
        dataset_names = {dataset["name"] for dataset in payload["datasets"]}
        run_dataset_names = {run["dataset_name"] for run in payload["runs"]}

        self.assertEqual(tuple(spec.name for spec in selected_specs), (TWO_DIGIT_ONLY,))
        self.assertEqual(dataset_names, {TWO_DIGIT_ONLY})
        self.assertEqual(run_dataset_names, {TWO_DIGIT_ONLY})
        self.assertEqual(payload["config"]["datasets"], [TWO_DIGIT_ONLY])

    def test_two_digit_payload_report_and_checkpoint_fields(self) -> None:
        """Validate a tiny two-digit diagnostic report and checkpoint round trip.

        中文说明:
        - 调用方 / Called by: `unittest`.
        - 调用对象 / Calls: `build_two_digit_diagnostic_grid_payload`,
          `build_two_digit_diagnostic_grid_markdown`, checkpoint helpers and report saver.
        - 作用 / Purpose: 用小参数保护 JSON/Markdown 字段与 resume checkpoint 结构。
        - 变量 / Variables: `payload` 是小网格报告, `row` 是首条运行明细。
        - 接入 / Integration: 修改报告 schema 时必须同步本测试。
        - 错误处理 / Error handling: 字段缺失或 checkpoint 读取失败会断言失败。
        - 关键词 / Keywords:
          payload|report|checkpoint|resume|two_digit_exact_match_mean|target_retention_rate|mhdsra2|test|报告|恢复
        """
        payload = build_two_digit_diagnostic_grid_payload(
            layer_counts=(1,),
            max_steps_per_stage_values=(2,),
            learning_rates=(0.01,),
            training_strategies=("baseline", TWO_DIGIT_REPLAY_TRAINING_STRATEGY),
            seeds=(101,),
            replay_ratio=0.75,
            stage_patience=1,
            two_digit_replay_ratio=0.5,
            stage_loss_weights={TWO_DIGIT_RULES_STAGE: 2.0},
            checkpoint_path="reports/test_mhdsra2_two_digit_diagnostic_grid.checkpoint.jsonl",
            device="cpu",
        )
        aggregate_row = payload["aggregates"][0]
        run_row = payload["runs"][0]
        markdown_text = "\n".join(build_two_digit_diagnostic_grid_markdown(payload))
        dataset_names = {dataset["name"] for dataset in payload["datasets"]}

        self.assertIn(TWO_DIGIT_ONLY, dataset_names)
        self.assertIn(PREREQ_PLUS_TWO_DIGIT, dataset_names)
        self.assertIn("curriculum_rule_set", dataset_names)
        self.assertEqual(
            set(payload["config"]["datasets"]),
            {"curriculum_rule_set", TWO_DIGIT_ONLY, PREREQ_PLUS_TWO_DIGIT},
        )
        self.assertIn("two_digit_exact_match_mean", aggregate_row)
        self.assertIn("target_retention_rate", aggregate_row)
        self.assertIn("checkpoint_path", payload["config"])
        self.assertTrue(payload["config"]["resume_supported"])
        self.assertIn("Two-Digit EM Mean", markdown_text)

        project_root = Path(__file__).resolve().parents[1]
        reports_dir = project_root / "reports"
        json_path, markdown_path = save_two_digit_diagnostic_grid_reports(payload, reports_dir)
        self.assertEqual(json_path.name, "mhdsra2_two_digit_diagnostic_grid.json")
        self.assertEqual(markdown_path.name, "mhdsra2_two_digit_diagnostic_grid.md")

        checkpoint_path = reports_dir / "test_mhdsra2_two_digit_diagnostic_checkpoint.jsonl"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        checkpoint_key = build_checkpoint_key(
            dataset_name=str(run_row["dataset_name"]),
            training_strategy=str(run_row["training_strategy"]),
            learning_rate=float(run_row["learning_rate"]),
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

    def test_run_point_records_two_digit_strategy(self) -> None:
        """Validate optimizer controls are visible in serialized two-digit runs.

        中文说明:
        - 调用方 / Called by: `unittest`.
        - 调用对象 / Calls: `run_one_two_digit_diagnostic_grid_point`,
          `serialize_two_digit_diagnostic_run`.
        - 作用 / Purpose: 确认 two-digit 策略和学习率进入 checkpoint/report。
        - 变量 / Variables: `row` 是序列化后的运行结果。
        - 接入 / Integration: 报告和 checkpoint 复用这些字段。
        - 错误处理 / Error handling: 字段缺失通过断言失败暴露。
        - 关键词 / Keywords:
          learning_rate|strategy|serialize|checkpoint|optimizer|two_digit|mhdsra2|test|学习率|策略
        """
        diagnostic_run = run_one_two_digit_diagnostic_grid_point(
            dataset_spec=build_two_digit_only_spec(),
            training_strategy=COMBINED_TRAINING_STRATEGY,
            learning_rate=0.01,
            max_steps_per_stage=2,
            num_layers=1,
            seed=101,
            replay_ratio=0.75,
            stage_patience=1,
            two_digit_replay_ratio=0.5,
            stage_loss_weights={TWO_DIGIT_RULES_STAGE: 2.0},
            device="cpu",
        )
        row = serialize_two_digit_diagnostic_run(diagnostic_run)

        self.assertEqual(row["learning_rate"], 0.01)
        self.assertEqual(row["training_strategy"], COMBINED_TRAINING_STRATEGY)
        self.assertEqual(row["run"]["learning_rate"], 0.01)
        self.assertEqual(row["run"]["training_strategy"], COMBINED_TRAINING_STRATEGY)

    def test_cli_parser_exposes_two_digit_grid_controls(self) -> None:
        """Validate the two-digit diagnostic CLI exposes scan dimensions.

        中文说明:
        - 调用方 / Called by: `unittest`.
        - 调用对象 / Calls: `build_parser`, argparse parsing helpers.
        - 作用 / Purpose: 保护 two-digit 网格命令所需参数和 `--resume` 开关。
        - 变量 / Variables: `args` 是解析后的命名空间。
        - 接入 / Integration: 回归命令直接依赖这些参数。
        - 错误处理 / Error handling: argparse 负责非法输入。
        - 关键词 / Keywords:
          cli|parser|resume|learning_rates|strategies|stage_loss_weights|two_digit|mhdsra2|参数|网格
        """
        args = build_parser().parse_args(
            [
                "--resume",
                "--layers",
                "4,8,16",
                "--max-steps-per-stage-values",
                "512,1024",
                "--learning-rates",
                "0.003,0.01",
                "--training-strategies",
                "baseline,two_digit_replay,two_digit_weighted_loss,combined",
                "--datasets",
                "two_digit_only",
                "--stage-loss-weights",
                "two_digit_rules=2.0",
                "--device",
                "cpu",
            ]
        )

        self.assertTrue(args.resume)
        self.assertEqual(args.layers, (4, 8, 16))
        self.assertEqual(args.max_steps_per_stage_values, (512, 1024))
        self.assertEqual(args.learning_rates, (0.003, 0.01))
        self.assertEqual(
            args.training_strategies,
            (
                "baseline",
                "two_digit_replay",
                "two_digit_weighted_loss",
                "combined",
            ),
        )
        self.assertEqual(args.datasets, (TWO_DIGIT_ONLY,))
        self.assertEqual(args.stage_loss_weights, {TWO_DIGIT_RULES_STAGE: 2.0})
        self.assertEqual(args.device, "cpu")


if __name__ == "__main__":
    unittest.main()
