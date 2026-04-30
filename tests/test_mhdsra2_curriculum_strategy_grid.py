import unittest
from pathlib import Path

from scripts.mhdsra2_curriculum_strategy_grid_report import (
    build_parser,
    save_curriculum_strategy_grid_reports,
)
from src.dsra.application.arithmetic_emergence_service import (
    build_curriculum_strategy_grid_markdown,
    build_curriculum_strategy_grid_payload,
)


class TestMHDSRA2CurriculumStrategyGrid(unittest.TestCase):
    """Report-backed tests for MHDSRA2 curriculum strategy grid scans.

    中文说明:
    - 调用方 / Called by: `python -m unittest` and `pytest`.
    - 调用对象 / Calls: grid CLI parser, payload builder, Markdown builder and report saver.
    - 作用 / Purpose: 验证 replay ratio/stage patience 网格扫描的字段和报告产物。
    - 变量 / Variables: `payload` 是小参数网格报告, `markdown_text` 是渲染后的报告文本。
    - 接入 / Integration: 放在 `tests/` 中保护 `reports/mhdsra2_curriculum_strategy_grid.*`。
    - 错误处理 / Error handling: 字段缺失、类型错误或报告未写入都会通过断言失败暴露。
    - 关键词 / Keywords:
      unittest|strategy_grid|replay_ratio|stage_patience|curriculum|retention|mhdsra2|reports|测试|网格
    """

    def test_grid_cli_parser_exposes_scan_controls(self) -> None:
        """Validate the grid CLI exposes replay ratio and patience controls.

        中文说明:
        - 调用方 / Called by: `unittest`.
        - 调用对象 / Calls: `build_parser`, `ArgumentParser.parse_args`.
        - 作用 / Purpose: 保护 CLI 能接收用户要求的 3x3 网格参数。
        - 变量 / Variables: `args` 是解析后的命令行命名空间。
        - 接入 / Integration: 主入口和独立脚本共享这组 CLI 参数。
        - 错误处理 / Error handling: 非法输入由 argparse 处理, 本测试只覆盖合法路径。
        - 关键词 / Keywords:
          cli|parser|replay_ratios|stage_patiences|layers|seeds|mhdsra2|grid|参数|扫描
        """
        args = build_parser().parse_args(
            [
                "--replay-ratios",
                "0.25,0.5,0.75",
                "--stage-patiences",
                "1,2,3",
                "--layers",
                "4",
                "--seeds",
                "101,202",
                "--target-stage-count",
                "2",
                "--max-steps-per-stage-values",
                "128,256",
            ]
        )

        self.assertEqual(args.replay_ratios, (0.25, 0.5, 0.75))
        self.assertEqual(args.stage_patiences, (1, 2, 3))
        self.assertEqual(args.layers, (4,))
        self.assertEqual(args.seeds, (101, 202))
        self.assertEqual(args.target_stage_count, 2)
        self.assertEqual(args.max_steps_per_stage_values, (128, 256))

    def test_strategy_grid_payload_and_report_contain_required_fields(self) -> None:
        """Validate a tiny strategy grid run produces the report fields.

        中文说明:
        - 调用方 / Called by: `unittest`.
        - 调用对象 / Calls: `build_curriculum_strategy_grid_payload`,
          `build_curriculum_strategy_grid_markdown`, `save_curriculum_strategy_grid_reports`.
        - 作用 / Purpose: 用最小训练步数保护 JSON/Markdown 报告结构和目标阶段字段。
        - 变量 / Variables: `grid_row` 是单个策略聚合结果, `run_row` 是单 seed 明细。
        - 接入 / Integration: 修改应用层聚合或报告格式时必须保持本测试通过。
        - 错误处理 / Error handling: 训练或写文件失败直接让测试失败。
        - 关键词 / Keywords:
          payload|report|strategy_grid|target_stage_count|retention|json|markdown|mhdsra2|字段|报告
        """
        payload = build_curriculum_strategy_grid_payload(
            replay_ratios=(0.25,),
            stage_patiences=(1,),
            layer_counts=(1,),
            seeds=(101,),
            max_steps_per_stage=2,
            max_steps_per_stage_values=(2,),
            curriculum_eval_interval=1,
            stage_threshold=0.95,
            target_stage_count=2,
        )
        grid_row = payload["grid_results"][0]
        run_row = payload["grid_runs"][0]

        self.assertEqual(payload["config"]["replay_ratios"], [0.25])
        self.assertEqual(payload["config"]["stage_patiences"], [1])
        self.assertEqual(payload["config"]["max_steps_per_stage_values"], [2])
        self.assertEqual(payload["config"]["target_stage_count"], 2)
        self.assertEqual(
            payload["config"]["target_stages"],
            ["unit_no_carry", "unit_with_carry"],
        )
        self.assertIn("has_stable_target_strategy", payload["summary"])
        self.assertIn("stable_target_strategy_count", payload["summary"])
        self.assertIn("best_strategy", payload["summary"])
        self.assertIn("target_retention_rate", grid_row)
        self.assertIn("stable_target_retention", grid_row)
        self.assertIn("max_steps_per_stage", grid_row)
        self.assertIn("retained_stage_count_mean", grid_row)
        self.assertIn("ever_passed_stage_count_mean", grid_row)
        self.assertIn("replay_ratio", run_row)
        self.assertIn("stage_patience", run_row)
        self.assertIn("max_steps_per_stage", run_row)
        self.assertIn("retained_stage_count", run_row["run"])

        markdown_lines = build_curriculum_strategy_grid_markdown(payload)
        markdown_text = "\n".join(markdown_lines)
        project_root = Path(__file__).resolve().parents[1]
        reports_dir = project_root / "reports"
        json_path, markdown_path = save_curriculum_strategy_grid_reports(
            payload,
            reports_dir,
        )
        written_markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(json_path.name, "mhdsra2_curriculum_strategy_grid.json")
        self.assertEqual(markdown_path.name, "mhdsra2_curriculum_strategy_grid.md")
        self.assertIn("# MHDSRA2 Curriculum Strategy Grid", markdown_text)
        self.assertIn("Target Retention Rate", markdown_text)
        self.assertIn("Max Steps", markdown_text)
        self.assertIn("Stable target strategy count", written_markdown)
        self.assertIn("unit_no_carry", written_markdown)
        self.assertIn("unit_with_carry", written_markdown)


if __name__ == "__main__":
    unittest.main()
