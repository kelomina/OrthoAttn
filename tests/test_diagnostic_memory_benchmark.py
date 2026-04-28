import argparse
import unittest
from unittest import mock

from scripts import diagnostic_memory_benchmark as diagnostic_benchmark
from scripts.diagnostic_memory_benchmark import MODEL_ORDER, run_diagnostic_benchmarks


class TestDiagnosticMemoryBenchmark(unittest.TestCase):
    """Smoke tests for runnable A/B/C diagnostic benchmark sections.

    中文说明:
    - 调用方 / Called by: `python -m unittest tests.test_diagnostic_memory_benchmark`
    - 调用对象 / Calls: `run_diagnostic_benchmarks`
    - 作用 / Purpose: 保护 A/B/C 诊断脚本、五模型结果结构和报告 section 产物不回退
    - 变量 / Variables: `args` 为小规模 CPU smoke 配置，`sections` 为返回的报告分节
    - 接入 / Integration: 作为诊断脚本接入现有 runner 之前的最小回归保障
    - 错误处理 / Error handling: 使用断言快速暴露结构缺失或执行失败
    - 关键词 / Keywords:
      diagnostics|a_b_c|smoke|sections|models|runner|report|benchmark|tests|回归
    """

    def test_run_diagnostic_benchmarks_returns_three_sections(self):
        """Validate that A/B/C suites run and emit five-model tables.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `run_diagnostic_benchmarks`
        - 作用 / Purpose: 校验诊断实验可在小尺寸 CPU 配置下运行并返回统一 section 结构
        - 变量 / Variables: `sections` 为 A/B/C 三组诊断结果
        - 接入 / Integration: 保护 `next_round_benchmark_runner` 的诊断接入链路
        - 错误处理 / Error handling: 通过 section 数量、行数和表格结构断言识别回归
        - 关键词 / Keywords:
          diagnostics|sections|five_models|tables|cpu|smoke|benchmark|runner|payload|统一结构
        """
        args = argparse.Namespace(
            diagnostic_device="cpu",
            diagnostic_slots=4,
            diagnostic_key_count=8,
            diagnostic_value_count=8,
            diagnostic_chunk_size=16,
            diagnostic_page_size=8,
            diagnostic_retrieved_top_pages=2,
            diagnostic_retrieved_max_tokens=8,
            diagnostic_exact_seq_len=96,
            diagnostic_exact_fact_spacing=8,
            diagnostic_override_seq_len=80,
            diagnostic_override_gap_grid=[8, 16],
            diagnostic_fixation_seq_len=96,
            diagnostic_fixation_distractor_grid=[4, 8],
        )
        sections = run_diagnostic_benchmarks(args)
        self.assertEqual(len(sections), 3)
        for section in sections:
            self.assertGreaterEqual(len(section["rows"]), 2)
            self.assertEqual(len(section["model_tables"]), 2)
            self.assertEqual(len(section["diagnostic_cases"]), len(section["model_tables"][1]["rows"]))
            model_table = section["model_tables"][0]
            self.assertEqual(len(model_table["rows"]), len(MODEL_ORDER))

    def test_diagnostic_suite_converts_oom_to_missing_result(self):
        """Validate that one model OOM does not interrupt the whole diagnostic suite.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `run_diagnostic_benchmarks`, `mock.patch`
        - 作用 / Purpose: 校验五模型中单个模型出现 OOM 时，其余模型仍继续执行并产出 section
        - 变量 / Variables:
          `original_run_case` 为原始执行函数，`guarded_run_case` 为带 OOM 注入的包装函数
        - 接入 / Integration: 保护未来大长度真实 benchmark 运行时不会因单模型爆显存而中断
        - 错误处理 / Error handling: 使用模拟 `RuntimeError('CUDA out of memory')` 验证容错分支
        - 关键词 / Keywords:
          oom|guard|continue|missing_result|diagnostics|suite|mock|runtimeerror|benchmark|容错
        """
        args = argparse.Namespace(
            diagnostic_device="cpu",
            diagnostic_slots=4,
            diagnostic_key_count=8,
            diagnostic_value_count=8,
            diagnostic_chunk_size=16,
            diagnostic_page_size=8,
            diagnostic_retrieved_top_pages=2,
            diagnostic_retrieved_max_tokens=8,
            diagnostic_exact_seq_len=64,
            diagnostic_exact_fact_spacing=8,
            diagnostic_override_seq_len=64,
            diagnostic_override_gap_grid=[8],
            diagnostic_fixation_seq_len=64,
            diagnostic_fixation_distractor_grid=[4],
        )
        original_run_case = diagnostic_benchmark.run_case_for_model

        def guarded_run_case(case, model_name, **kwargs):
            if model_name == "sliding_window_attention":
                raise RuntimeError("CUDA out of memory while allocating tensor")
            return original_run_case(case, model_name, **kwargs)

        with mock.patch.object(diagnostic_benchmark, "run_case_for_model", side_effect=guarded_run_case):
            sections = run_diagnostic_benchmarks(args)

        self.assertEqual(len(sections), 3)
        first_case = sections[0]["diagnostic_cases"][0]
        self.assertEqual(first_case["models"]["sliding_window_attention"]["error"], "oom")
        self.assertIsNone(first_case["models"]["sliding_window_attention"]["is_correct"])
        self.assertEqual(sections[0]["model_tables"][1]["rows"][0][4], "OOM")


if __name__ == "__main__":
    unittest.main()
