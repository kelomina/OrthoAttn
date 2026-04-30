import unittest
from pathlib import Path

from scripts.compare_mhdsra2_vs_dsra import (
    build_benchmark_comparison_row,
    build_benchmark_payload,
    save_benchmark_reports,
)
from scripts.json_retrieval_test import build_retrieval_model
from scripts.next_round_benchmark_runner import build_parser


class TestNextRoundBenchmarkRunner(unittest.TestCase):
    """Regression tests for the dedicated next-round benchmark plumbing.

    中文说明:
    - 调用方 / Called by: `python -m unittest tests.test_next_round_benchmark_runner`
    - 调用对象 / Calls:
      `build_retrieval_model`, `build_benchmark_comparison_row`,
      `build_benchmark_payload`, `save_benchmark_reports`
    - 作用 / Purpose: 保护 MHDSRA2 benchmark 接入点与统一报告结构不发生回退
    - 变量 / Variables:
      `row/payload` 为标准化对比结果, `model` 为 JSON retrieval 任务模型
    - 接入 / Integration: 作为 next-round benchmark 的轻量回归测试放在 `tests/` 目录下
    - 错误处理 / Error handling: 使用 `assert*` 断言快速暴露结构变更与接口缺失
    - 关键词 / Keywords:
      benchmark|runner|report|payload|mhdsra2|json_retrieval|compare|tests|regression|统一结构
    """

    def test_build_retrieval_model_supports_mhdsra2(self):
        """Validate JSON retrieval factory now builds MHDSRA2-compatible model.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `build_retrieval_model`
        - 作用 / Purpose: 确认 `model_type="mhdsra2"` 已接入原 JSON retrieval 构建器
        - 变量 / Variables: `model` 为构造出的检索任务模型
        - 接入 / Integration: 保护 `next_round_benchmark_runner` 所依赖的模型工厂
        - 错误处理 / Error handling: 通过属性断言捕获接口缺失
        - 关键词 / Keywords:
          retrieval_model|mhdsra2|factory|forward_step|json|benchmark|compat|model_type|构建器|回归
        """
        model = build_retrieval_model(
            model_type="mhdsra2",
            vocab_size=259,
            dim=64,
            K=32,
            kr=8,
            chunk_size=64,
            local_context_size=4,
            local_context_mode="concat",
        )
        self.assertTrue(hasattr(model, "dsra"))
        self.assertTrue(hasattr(model.dsra, "forward_step"))
        self.assertEqual(model.chunk_size, 64)

    def test_benchmark_payload_and_reports_use_unified_rows(self):
        """Validate normalized rows aggregate and persist into unified reports.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls:
          `build_benchmark_comparison_row`, `build_benchmark_payload`, `save_benchmark_reports`
        - 作用 / Purpose: 校验统一对比表与汇总报告的核心字段稳定存在
        - 变量 / Variables:
          `rows` 标准化对比行, `payload` 汇总结果, `json_path/md_path` 报告路径
        - 接入 / Integration: 保护 compare 中新增的通用 benchmark 报告能力
        - 错误处理 / Error handling: 通过 winner/summary/文件存在性断言识别回归
        - 关键词 / Keywords:
          payload|summary|winner|reports|markdown|json|rows|benchmark|compare|统一报告
        """
        rows = [
            build_benchmark_comparison_row(
                suite="needle_in_haystack",
                task="seq_len=8192",
                split="overall",
                metric="best_accuracy",
                dsra_value=0.75,
                mhdsra2_value=0.80,
            ),
            build_benchmark_comparison_row(
                suite="json_retrieval_generalization",
                task="museum_artifact_generalization",
                split="test",
                metric="generation_exact_match_rate",
                dsra_value=0.25,
                mhdsra2_value=0.20,
            ),
        ]
        payload = build_benchmark_payload(
            config={"seed": 7},
            sections=[
                {
                    "title": "Synthetic",
                    "rows": rows,
                    "model_tables": [
                        {
                            "title": "Five-model diagnostic summary",
                            "columns": ["Model", "exact_match_rate"],
                            "rows": [
                                ["Archived DSRA alias / MHDSRA2", "0.2500"],
                                ["MH-DSRA-v2 (paged recall)", "1.0000"],
                            ],
                        }
                    ],
                }
            ],
        )

        self.assertEqual(payload["summary"]["overall"]["total_rows"], 2)
        self.assertEqual(payload["summary"]["overall"]["mhdsra2_wins"], 1)
        self.assertEqual(payload["summary"]["overall"]["dsra_wins"], 1)
        self.assertEqual(payload["rows"][0]["winner"], "mhdsra2")
        self.assertEqual(payload["rows"][1]["winner"], "dsra")

        project_root = Path(__file__).resolve().parents[1]
        reports_dir = project_root / "reports" / "test_next_round_benchmark" / "reports"
        json_path, md_path = save_benchmark_reports(payload, reports_dir)
        self.assertTrue(json_path.exists())
        self.assertTrue(md_path.exists())
        markdown_text = md_path.read_text(encoding="utf-8")
        self.assertIn("Five-model diagnostic summary", markdown_text)
        self.assertIn("MH-DSRA-v2 (paged recall)", markdown_text)

    def test_next_round_parser_accepts_diagnostic_retrieval_tau(self):
        """Validate retrieval tau is exposed through the benchmark CLI.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `build_parser`, `ArgumentParser.parse_args`
        - 作用 / Purpose: 校验 next-round benchmark CLI 能接收并解析 `--diagnostic-retrieval-tau`
        - 变量 / Variables:
          `parser` 为 benchmark 参数解析器, `args` 为解析后的命名空间,
          `diagnostic_retrieval_tau` 为 MHDSRA2 paged recall softmax 锐度参数
        - 接入 / Integration: 调优 retrieval attention 时可直接通过 CLI 传入 tau，无需改代码
        - 错误处理 / Error handling: 参数缺失或类型错误会由 argparse 或断言暴露
        - 关键词 / Keywords:
          retrieval_tau|diagnostic|cli|parser|benchmark|next_round|mhdsra2|paged_recall|softmax|参数
        """
        parser = build_parser()
        args = parser.parse_args(["--diagnostic-retrieval-tau", "10.0"])

        self.assertEqual(args.diagnostic_retrieval_tau, 10.0)


if __name__ == "__main__":
    unittest.main()
