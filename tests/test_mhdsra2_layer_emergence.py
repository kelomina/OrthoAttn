import unittest
from pathlib import Path

from scripts.mhdsra2_layer_emergence_report import build_parser, save_layer_emergence_reports
from src.dsra.application.arithmetic_emergence_service import (
    CURRICULUM_RULE_SET,
    DEFAULT_ARITHMETIC_EMERGENCE_DEVICE,
    DEFAULT_ARITHMETIC_EMERGENCE_LEARNING_RATE,
    DEFAULT_ARITHMETIC_EMERGENCE_MAX_STEPS_PER_STAGE,
    DEFAULT_ARITHMETIC_EMERGENCE_REPLAY_RATIO,
    DEFAULT_ARITHMETIC_EMERGENCE_STAGE_PATIENCE,
    ArithmeticStageMetric,
    DecimalArithmeticTokenizer,
    GeneratedArithmeticAnswer,
    build_default_arithmetic_spec,
    build_layer_emergence_payload,
    count_completed_curriculum_stages,
    is_exact_generated_answer,
    resolve_torch_device,
    select_adaptive_curriculum_training_example,
    select_curriculum_training_example,
    should_advance_curriculum_stage,
    should_advance_open_curriculum_stages,
)


class TestMHDSRA2LayerEmergence(unittest.TestCase):
    """Report-backed tests for decimal arithmetic emergence.

    中文说明:
    - 调用方 / Called by: `python -m unittest` and `pytest`.
    - 调用对象 / Calls: `build_layer_emergence_payload`,
      `save_layer_emergence_reports`, `DecimalArithmeticTokenizer`.
    - 作用 / Purpose: 验证 100+100 外推实验的数据泄漏约束、tokenizer 和报告结构。
    - 变量 / Variables:
      `payload` 为 JSON 报告数据, `spec` 为训练/OOD 数据集规约。
    - 接入 / Integration: 作为报告脚本的快速回归测试放在 `tests/`。
    - 错误处理 / Error handling: 断言暴露训练泄漏、生成解析或报告字段回归。
    - 关键词 / Keywords:
      unittest|mhdsra2|arithmetic|emergence|100+100|decimal|tokenizer|reports|regression|测试
    """

    def test_curriculum_rule_set_excludes_headline_and_hundreds(self) -> None:
        """Validate the low-value training set does not leak held-out arithmetic facts.

        中文说明:
        - 调用方 / Called by: `unittest`.
        - 调用对象 / Calls: `build_default_arithmetic_spec`,
          `ArithmeticRuleDatasetSpec.validate_training_scope`.
        - 作用 / Purpose: 确认训练集没有 `100+100`、百位 operand/result 或 OOD 样例。
        - 变量 / Variables:
          `training_equations` 是训练等式集合, `held_out_equations` 是 headline/OOD 集合。
        - 接入 / Integration: 修改默认训练规约时必须继续满足本测试。
        - 错误处理 / Error handling: 泄漏会触发断言失败或 `ValueError`。
        - 关键词 / Keywords:
          leakage|training_set|100+100|hundreds|ood|minimal_rule_set|arithmetic|mhdsra2|test|泄漏
        """
        spec = build_default_arithmetic_spec()
        spec.validate_training_scope()
        training_equations = {example.equation for example in spec.training_examples}
        held_out_equations = {
            spec.headline_example.equation,
            *[example.equation for example in spec.ood_examples],
        }

        self.assertNotIn("100+100=200", training_equations)
        self.assertTrue(all(example.max_term < 100 for example in spec.training_examples))
        self.assertTrue(training_equations.isdisjoint(held_out_equations))
        self.assertEqual(spec.name, CURRICULUM_RULE_SET)
        self.assertEqual(
            [stage.name for stage in spec.curriculum_stages],
            ["unit_no_carry", "unit_with_carry", "two_digit_rules"],
        )

    def test_curriculum_sampler_moves_through_ordered_stages(self) -> None:
        """Validate curriculum sampling order: no carry, carry, then two-digit rules.

        中文说明:
        - 调用方 / Called by: `unittest`.
        - 调用对象 / Calls: `build_default_arithmetic_spec`,
          `select_curriculum_training_example`.
        - 作用 / Purpose: 确认训练循环先学个位无进位、再学进位、最后学两位数。
        - 变量 / Variables: `selected_equations` 是每个关键 step 选中的等式。
        - 接入 / Integration: 修改课程切分逻辑时必须保持本测试通过。
        - 错误处理 / Error handling: 非法课程会由采样函数抛出 `ValueError`。
        - 关键词 / Keywords:
          curriculum|sampler|unit_no_carry|carry|two_digit|mhdsra2|test|order|课程|顺序
        """
        spec = build_default_arithmetic_spec()
        selected_equations = [
            select_curriculum_training_example(spec, step=step, training_steps=9).equation
            for step in (0, 3, 6)
        ]

        self.assertIn(
            selected_equations[0],
            {example.equation for example in spec.curriculum_stages[0].examples},
        )
        self.assertIn(
            selected_equations[1],
            {example.equation for example in spec.curriculum_stages[1].examples},
        )
        self.assertIn(
            selected_equations[2],
            {example.equation for example in spec.curriculum_stages[2].examples},
        )

    def test_adaptive_curriculum_advances_only_after_stage_threshold(self) -> None:
        """Validate adaptive curriculum stage promotion and active-stage sampling.

        中文说明:
        - 调用方 / Called by: `unittest`.
        - 调用对象 / Calls: `build_default_arithmetic_spec`,
          `select_adaptive_curriculum_training_example`, `should_advance_curriculum_stage`,
          `count_completed_curriculum_stages`.
        - 作用 / Purpose: 证明课程不再按固定步数切阶段, 而是阶段 EM 达标后推进。
        - 变量 / Variables: `low_metric/high_metric` 是未达标和达标阶段指标。
        - 接入 / Integration: 修改自适应课程推进规则时同步维护本测试。
        - 错误处理 / Error handling: 越界或空阶段由采样函数抛出 `ValueError`。
        - 关键词 / Keywords:
          adaptive|curriculum|advance|threshold|stage_em|sampling|mhdsra2|test|推进|达标
        """
        spec = build_default_arithmetic_spec()
        first_stage_example = select_adaptive_curriculum_training_example(
            spec,
            active_stage_index=0,
            local_step=0,
        )
        second_stage_example = select_adaptive_curriculum_training_example(
            spec,
            active_stage_index=1,
            local_step=0,
        )
        replay_example = select_adaptive_curriculum_training_example(
            spec,
            active_stage_index=1,
            local_step=1,
        )
        no_replay_example = select_adaptive_curriculum_training_example(
            spec,
            active_stage_index=1,
            local_step=1,
            replay_ratio=0.0,
        )
        low_metric = ArithmeticStageMetric(stage_name="unit_no_carry", exact_match=0.50)
        high_metric = ArithmeticStageMetric(stage_name="unit_no_carry", exact_match=1.00)

        self.assertIn(first_stage_example, spec.curriculum_stages[0].examples)
        self.assertIn(second_stage_example, spec.curriculum_stages[1].examples)
        self.assertIn(replay_example, spec.curriculum_stages[0].examples)
        self.assertIn(no_replay_example, spec.curriculum_stages[1].examples)
        self.assertFalse(should_advance_curriculum_stage(low_metric))
        self.assertTrue(should_advance_curriculum_stage(high_metric))
        self.assertFalse(
            should_advance_open_curriculum_stages(
                (
                    low_metric,
                    ArithmeticStageMetric(stage_name="unit_with_carry", exact_match=1.00),
                ),
                ("unit_no_carry", "unit_with_carry"),
            )
        )
        self.assertTrue(
            should_advance_open_curriculum_stages(
                (
                    high_metric,
                    ArithmeticStageMetric(stage_name="unit_with_carry", exact_match=1.00),
                ),
                ("unit_no_carry", "unit_with_carry"),
            )
        )
        self.assertEqual(
            count_completed_curriculum_stages(
                (
                    high_metric,
                    ArithmeticStageMetric(stage_name="unit_with_carry", exact_match=0.25),
                )
            ),
            1,
        )

    def test_cli_exposes_curriculum_stage_controls(self) -> None:
        """Validate CLI exposes max stage steps, eval interval and stage threshold.

        中文说明:
        - 调用方 / Called by: `unittest`.
        - 调用对象 / Calls: `build_parser`, `ArgumentParser.parse_args`.
        - 作用 / Purpose: 保护 `--max-steps-per-stage`, `--curriculum-eval-interval`,
          `--stage-threshold` 三个课程训练控制参数。
        - 变量 / Variables: `args` 是解析后的命令行参数对象。
        - 接入 / Integration: 报告脚本 CLI 变更时必须保持这些参数可用。
        - 错误处理 / Error handling: argparse 对非法输入负责报错退出。
        - 关键词 / Keywords:
          cli|max_steps_per_stage|curriculum_eval_interval|stage_threshold|argparse|mhdsra2|test|参数|课程|阈值
        """
        args = build_parser().parse_args(
            [
                "--max-steps-per-stage",
                "7",
                "--curriculum-eval-interval",
                "3",
                "--stage-threshold",
                "0.5",
                "--replay-ratio",
                "0.25",
                "--stage-patience",
                "3",
                "--learning-rate",
                "0.01",
                "--device",
                "cpu",
            ]
        )

        self.assertEqual(args.max_steps_per_stage, 7)
        self.assertEqual(args.curriculum_eval_interval, 3)
        self.assertEqual(args.stage_threshold, 0.5)
        self.assertEqual(args.replay_ratio, 0.25)
        self.assertEqual(args.stage_patience, 3)
        self.assertEqual(args.learning_rate, 0.01)
        self.assertEqual(args.device, "cpu")

    def test_cli_uses_optimized_main_experiment_defaults(self) -> None:
        """Validate the arithmetic emergence CLI defaults use the optimized main run.

        中文说明:
        - 调用方 / Called by: `unittest`.
        - 调用对象 / Calls: `build_parser`, `ArgumentParser.parse_args`.
        - 作用 / Purpose: 保护主实验默认参数，避免回退到早期不稳定训练预算。
        - 变量 / Variables: `args` 是空参数解析后的 CLI 默认值对象。
        - 接入 / Integration: 默认报告入口和 `scripts/main.py mhdsra2_layer_emergence` 共用这些默认值。
        - 错误处理 / Error handling: argparse 默认值缺失会触发断言失败。
        - 关键词 / Keywords:
          defaults|learning_rate|max_steps|replay_ratio|stage_patience|mhdsra2|cli|test|默认|主实验
        """
        args = build_parser().parse_args([])

        self.assertEqual(args.max_steps_per_stage, DEFAULT_ARITHMETIC_EMERGENCE_MAX_STEPS_PER_STAGE)
        self.assertEqual(args.replay_ratio, DEFAULT_ARITHMETIC_EMERGENCE_REPLAY_RATIO)
        self.assertEqual(args.stage_patience, DEFAULT_ARITHMETIC_EMERGENCE_STAGE_PATIENCE)
        self.assertEqual(args.learning_rate, DEFAULT_ARITHMETIC_EMERGENCE_LEARNING_RATE)
        self.assertEqual(args.device, DEFAULT_ARITHMETIC_EMERGENCE_DEVICE)

    def test_device_resolver_supports_auto_and_cpu(self) -> None:
        """Validate arithmetic emergence device resolution.

        中文说明:
        - 调用方 / Called by: `unittest`.
        - 调用对象 / Calls: `resolve_torch_device`.
        - 作用 / Purpose: 保护 `--device auto/cpu/cuda` 的应用层解析入口。
        - 变量 / Variables: `cpu_device` 与 `auto_device` 是解析后的 torch device。
        - 接入 / Integration: CLI 只传字符串, 训练服务负责解析为 `torch.device`。
        - 错误处理 / Error handling: 非法设备名必须抛出 `ValueError`。
        - 关键词 / Keywords:
          device|auto|cpu|cuda|resolve|torch|mhdsra2|test|设备|解析
        """
        cpu_device = resolve_torch_device("cpu")
        auto_device = resolve_torch_device("auto")

        self.assertEqual(cpu_device.type, "cpu")
        self.assertIn(auto_device.type, {"cpu", "cuda"})
        with self.assertRaises(ValueError):
            resolve_torch_device("gpu")

    def test_decimal_tokenizer_round_trips_headline_equation(self) -> None:
        """Validate character tokenizer round-trips `100+100=200`.

        中文说明:
        - 调用方 / Called by: `unittest`.
        - 调用对象 / Calls: `DecimalArithmeticTokenizer.encode_text`,
          `DecimalArithmeticTokenizer.decode_token_ids`.
        - 作用 / Purpose: 确认十进制字符表示不会把数字压成整数分类 token。
        - 变量 / Variables: `token_ids` 是编码结果, `decoded_text` 是解码文本。
        - 接入 / Integration: tokenizer 字符集变更时同步更新本测试。
        - 错误处理 / Error handling: 未知字符或 id 会由 tokenizer 抛出异常。
        - 关键词 / Keywords:
          tokenizer|round_trip|100+100=200|decimal|characters|bos|eos|mhdsra2|test|往返
        """
        tokenizer = DecimalArithmeticTokenizer()
        token_ids = tokenizer.encode_text("100+100=200", add_bos=True, add_eos=True)
        decoded_text = tokenizer.decode_token_ids(token_ids)

        self.assertEqual(decoded_text, "100+100=200")

    def test_exact_generation_requires_complete_eos_answer(self) -> None:
        """Validate exact-match parsing rejects incomplete greedy generations.

        中文说明:
        - 调用方 / Called by: `unittest`.
        - 调用对象 / Calls: `is_exact_generated_answer`, `GeneratedArithmeticAnswer`.
        - 作用 / Purpose: 确认成功标准要求生成 `200<eos>`, 不接受 teacher-forced 或截断结果。
        - 变量 / Variables: `complete/incomplete/wrong` 是三种生成结果。
        - 接入 / Integration: 修改生成判定时必须保持本约束。
        - 错误处理 / Error handling: 纯布尔断言, 不吞异常。
        - 关键词 / Keywords:
          exact_match|eos|required|greedy|generation|teacher_forcing|100+100|mhdsra2|test|完整
        """
        complete = GeneratedArithmeticAnswer("200", True, (1, 2, 3))
        incomplete = GeneratedArithmeticAnswer("200", False, (1, 2, 3))
        wrong = GeneratedArithmeticAnswer("201", True, (1, 2, 4))

        self.assertTrue(is_exact_generated_answer(complete, "200"))
        self.assertFalse(is_exact_generated_answer(incomplete, "200"))
        self.assertFalse(is_exact_generated_answer(wrong, "200"))

    def test_arithmetic_emergence_report_contains_required_fields(self) -> None:
        """Validate small arithmetic emergence report payload and artifacts.

        中文说明:
        - 调用方 / Called by: `unittest`.
        - 调用对象 / Calls: `build_layer_emergence_payload`, `save_layer_emergence_reports`.
        - 作用 / Purpose: 用小参数验证报告包含主实验、负控、headline/OOD 指标和最小层数字段。
        - 变量 / Variables: `payload` 是报告数据, `markdown_text` 是报告 Markdown。
        - 接入 / Integration: 保护 `reports/mhdsra2_layer_emergence_curve.*` 结构。
        - 错误处理 / Error handling: 文件缺失或字段缺失会断言失败。
        - 关键词 / Keywords:
          reports|json|markdown|minimal_rule_set|single_fact_only|headline|ood|mhdsra2|test|报告
        """
        payload = build_layer_emergence_payload(
            layer_counts=(1,),
            seeds=(101,),
            max_steps_per_stage=2,
            curriculum_eval_interval=1,
            stage_threshold=0.95,
            replay_ratio=0.25,
            stage_patience=1,
            learning_rate=0.01,
            device="cpu",
            include_standard_baseline=False,
        )
        dataset_names = {dataset["name"] for dataset in payload["datasets"]}
        aggregate_row = payload["aggregates"][0]

        self.assertIn("curriculum_rule_set", dataset_names)
        self.assertIn("single_fact_only", dataset_names)
        self.assertIn("minimum_arithmetic_emergent_layers", payload["summary"])
        self.assertIn("minimum_curriculum_mastery_layers", payload["summary"])
        self.assertIn("headline_exact_match_mean", aggregate_row)
        self.assertIn("ood_exact_match_mean", aggregate_row)
        self.assertEqual(payload["config"]["max_steps_per_stage"], 2)
        self.assertEqual(payload["config"]["curriculum_eval_interval"], 1)
        self.assertEqual(payload["config"]["curriculum_stage_exact_match_threshold"], 0.95)
        self.assertEqual(payload["config"]["replay_ratio"], 0.25)
        self.assertEqual(payload["config"]["stage_patience"], 1)
        self.assertEqual(payload["config"]["learning_rate"], 0.01)
        self.assertEqual(payload["config"]["device"], "cpu")
        self.assertIn("curriculum_stage_aggregates", payload)
        self.assertIn("pass_rate", payload["curriculum_stage_aggregates"][0])
        self.assertIn("advance_step_mean", payload["curriculum_stage_aggregates"][0])
        self.assertIn("curriculum_snapshots", payload["runs"][0])
        self.assertIn("stage_exact_matches", payload["runs"][0]["curriculum_snapshots"][0])
        self.assertLessEqual(payload["runs"][0]["training_steps_executed"], 6)
        self.assertIn("ever_passed_stage_count", payload["runs"][0])
        self.assertIn("retained_stage_count", payload["runs"][0])
        self.assertIn("stopped_reason", payload["runs"][0])

        project_root = Path(__file__).resolve().parents[1]
        reports_dir = project_root / "reports"
        json_path, markdown_path = save_layer_emergence_reports(payload, reports_dir)
        markdown_text = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(json_path.name, "mhdsra2_layer_emergence_curve.json")
        self.assertEqual(markdown_path.name, "mhdsra2_layer_emergence_curve.md")
        self.assertIn("# MHDSRA2 Decimal Arithmetic Emergence", markdown_text)
        self.assertIn("Max steps per curriculum stage", markdown_text)
        self.assertIn("Replay ratio", markdown_text)
        self.assertIn("Stage patience", markdown_text)
        self.assertIn("Learning rate", markdown_text)
        self.assertIn("Device", markdown_text)
        self.assertIn("Training Stop Summary", markdown_text)
        self.assertIn("Minimum curriculum mastery layers", markdown_text)
        self.assertIn("Ever Passed", markdown_text)
        self.assertIn("Retained", markdown_text)
        self.assertIn("unit_no_carry", markdown_text)
        self.assertIn("Curriculum Stage Aggregate", markdown_text)
        self.assertIn("Mean Pass Step", markdown_text)
        self.assertIn("single_fact_only", markdown_text)
        self.assertIn("Headline EM Mean", markdown_text)
        self.assertIn("OOD EM Mean", markdown_text)


if __name__ == "__main__":
    unittest.main()
