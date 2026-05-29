import unittest
import tempfile
from pathlib import Path

import torch

from scripts.needle_in_haystack_test import (
    NIAH_DEPTHS,
    build_niah_verification_markdown,
    build_parser,
    compute_selected_logits_sample_metrics,
    extract_query_positions_and_targets,
    generate_haystack_with_needle,
    run_single_niah_capacity_test,
    save_niah_capacity_reports,
    save_niah_verification_report,
    summarize_niah_sample_metrics,
)
from src.dsra.dsra_model import MultiLayerMHDSRA2Model


class TestNIAHMemoryBoundedTraining(unittest.TestCase):
    """Regression tests for memory-bounded NIAH selected-logit training.

    中文说明:
    - 调用方 / Called by: `python -m pytest tests/test_niah_memory_bounded_training.py`
    - 调用对象 / Calls: `MultiLayerMHDSRA2Model.forward`,
      `MultiLayerMHDSRA2Model.forward_selected_logits`, NIAH data helpers
    - 作用 / Purpose: 保护 2M NIAH 显存优化所依赖的 selected logits 路径，
      避免未来回退到全序列 logits 训练
    - 变量 / Variables:
      `model` 是小规模 MHDSRA2；`tokens/positions` 构造等价性断言；
      `capacity_result` 验证训练入口结构
    - 接入 / Integration: 放在 `tests/` 下，作为 NIAH 训练入口的单元回归测试
    - 错误处理 / Error handling: 使用断言暴露 logits 不一致、target 提取错误或容量入口失败
    - 副作用 / Side effects: 只运行小规模 CPU 张量计算，不写文件、不访问网络
    - 事务边界 / Transaction boundary: 不涉及 Unit of Work 或持久化事务
    - 并发与幂等 / Concurrency and idempotency: 固定 `torch.manual_seed` 后可重复
    - 关键词 / Keywords:
      niah|selected_logits|2m|memory|train_step|mhdsra2|regression|capacity|oom|测试
    """

    def test_forward_selected_logits_matches_full_forward_at_positions(self):
        """Validate selected logits keep full-forward semantics for supervised tokens.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `MultiLayerMHDSRA2Model.forward`,
          `MultiLayerMHDSRA2Model.forward_selected_logits`
        - 作用 / Purpose: 确认新增显存优化路径只减少物化输出，不改变指定位置 logits 语义
        - 错误处理 / Error handling: 若数值不一致，`torch.testing.assert_close` 失败
        - 关键词 / Keywords:
          selected_logits|full_forward|equivalence|mhdsra2|niah|positions|memory|test|logits|等价
        """
        torch.manual_seed(123)
        model = MultiLayerMHDSRA2Model(
            vocab_size=32,
            dim=16,
            num_layers=1,
            K=8,
            kr=2,
            chunk_size=4,
        )
        tokens = torch.randint(4, 32, (2, 11), dtype=torch.long)
        positions = torch.tensor([3, 10], dtype=torch.long)

        full_logits = model(tokens)
        selected_logits = model.forward_selected_logits(tokens, positions)
        expected = full_logits[torch.arange(tokens.shape[0]), positions]

        torch.testing.assert_close(selected_logits, expected)

    def test_generate_haystack_extracts_query_without_gpu_sequence_copy(self):
        """Validate CPU query-position extraction for generated NIAH samples.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `generate_haystack_with_needle`,
          `extract_query_positions_and_targets`
        - 作用 / Purpose: 保护 2M 样本仍留在 CPU，只把小型 target tensor 搬到训练设备
        - 错误处理 / Error handling: 查询位置或 target 错误会触发断言失败
        - 关键词 / Keywords:
          haystack|query_position|variable_value|cpu|memory|niah|needle|extract|test|查询
        """
        torch.manual_seed(222)
        X, Y, _ = generate_haystack_with_needle(
            batch_size=8,
            seq_len=64,
            vocab_size=100,
            needle_depth_ratio=0.5,
        )

        positions, targets = extract_query_positions_and_targets(X, Y, torch.device("cpu"))
        center = int((64 - 5) * 0.5)
        half_window = max(1, (64 - 5) // 20)
        query_rows, query_cols = (X == 1).nonzero(as_tuple=True)

        self.assertEqual(query_rows.tolist(), list(range(8)))
        self.assertEqual(positions.tolist(), query_cols.tolist())
        self.assertTrue(any(position != 63 for position in positions.tolist()))
        needle_key_rows, needle_key_cols = (X == 2).nonzero(as_tuple=True)
        needle_key_positions_per_row = {}
        for r, c in zip(needle_key_rows.tolist(), needle_key_cols.tolist()):
            needle_key_positions_per_row.setdefault(r, []).append(c)
        for row_idx in range(8):
            needle_pos = min(needle_key_positions_per_row.get(row_idx, []))
            self.assertGreaterEqual(needle_pos, center - half_window)
            self.assertLessEqual(needle_pos, center + half_window)
            self.assertEqual(X[row_idx, needle_pos].item(), 2)
            self.assertEqual(targets[row_idx].item(), X[row_idx, needle_pos + 1].item())
            query_pos = positions[row_idx].item()
            self.assertGreater(query_pos, needle_pos + 1)
        self.assertEqual(targets.tolist(), Y[torch.arange(8), positions].tolist())
        self.assertTrue(torch.all(targets >= 4).item())
        self.assertEqual(X.device.type, "cpu")
        self.assertEqual(Y.device.type, "cpu")

    def test_capacity_train_step_uses_memory_bounded_path(self):
        """Validate the NIAH train-step capacity entry point runs on a tiny CPU case.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `run_single_niah_capacity_test`
        - 作用 / Purpose: 保护容量测试入口能完成一次训练步并返回结构化结果
        - 错误处理 / Error handling: 异常会直接使测试失败；结果字段缺失或非法会触发断言
        - 关键词 / Keywords:
          capacity|train_step|selected_logits|cpu|smoke|niah|mhdsra2|memory|test|容量
        """
        torch.manual_seed(321)
        result = run_single_niah_capacity_test(
            seq_len=64,
            device=torch.device("cpu"),
            mode="train_step",
            vocab_size=32,
            dim=16,
            num_layers=1,
            K=8,
            kr=2,
            chunk_size=8,
            batch_size=1,
            batches_per_depth=2,
        )

        self.assertEqual(result["status"], "ok")
        self.assertIn("accuracy", result)
        self.assertIn("peak_mem_mb", result)
        self.assertGreaterEqual(result["peak_mem_mb"], 0.0)
        self.assertGreaterEqual(result["accuracy"], 0.0)
        self.assertLessEqual(result["accuracy"], 1.0)
        self.assertEqual(result["batches_per_depth"], 2)
        self.assertEqual(result["total_samples"], len(NIAH_DEPTHS) * 2)
        self.assertTrue(all(row["samples"] == 2 for row in result["depth_results"]))

    def test_small_vocab_generation_warns_about_easy_answer_space(self):
        """Validate tiny NIAH vocabularies are flagged without changing compatibility.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `generate_haystack_with_needle`
        - 作用 / Purpose: 提醒小词表会让 NIAH answer space 过小，避免准确率被误读
        - 错误处理 / Error handling: 缺少 warning 或生成失败会触发测试失败
        - 关键词 / Keywords:
          vocab|warning|answer_space|niah|accuracy|saturation|test|small|词表|提示
        """
        with self.assertWarns(RuntimeWarning):
            X, Y, _ = generate_haystack_with_needle(
                batch_size=1,
                seq_len=16,
                vocab_size=6,
                needle_depth_ratio=0.5,
            )

        self.assertEqual(X.shape, (1, 16))
        self.assertEqual(Y.shape, (1, 16))

    def test_cli_parser_exposes_verify_and_larger_scale_benchmark(self):
        """Validate reproducible NIAH CLI subcommands and larger-scale defaults.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `build_parser`, `ArgumentParser.parse_args`
        - 作用 / Purpose: 保护 `verify-2m` 和 `benchmark-scale` CLI，确保更大参数规模默认值可见
        - 错误处理 / Error handling: argparse 解析失败或默认值回退会触发测试失败
        - 关键词 / Keywords:
          cli|verify_2m|benchmark_scale|parser|larger_scale|defaults|niah|mhdsra2|reports|命令
        """
        parser = build_parser()

        verify_args = parser.parse_args(["verify-2m"])
        benchmark_args = parser.parse_args(["benchmark-scale"])

        self.assertEqual(verify_args.command, "verify-2m")
        self.assertEqual(verify_args.seq_len, 2_097_152)
        self.assertEqual(verify_args.dim, 64)
        self.assertEqual(benchmark_args.command, "benchmark-scale")
        self.assertEqual(benchmark_args.dim, 128)
        self.assertEqual(benchmark_args.num_layers, 4)
        self.assertEqual(benchmark_args.slots, 128)
        self.assertEqual(benchmark_args.read_topk, 16)
        self.assertEqual(verify_args.eval_batches_per_depth, 1)
        self.assertIsNone(verify_args.robust_eval_interval)
        self.assertEqual(verify_args.robust_eval_batches_per_depth, 32)
        benchmark_perf_args = parser.parse_args(["benchmark-scale", "--cudnn-benchmark"])
        self.assertTrue(benchmark_perf_args.cudnn_benchmark)
        verify_swanlab_args = parser.parse_args(["verify-2m"])
        self.assertEqual(verify_swanlab_args.swanlab_mode, "cloud")
        disabled_args = parser.parse_args(["verify-2m", "--swanlab-mode", "disabled"])
        self.assertEqual(disabled_args.swanlab_mode, "disabled")

    def test_selected_logits_metrics_include_rank_confidence_and_sample_rows(self):
        """Validate selected-logit diagnostics expose richer NIAH capability signals.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `compute_selected_logits_sample_metrics`,
          `summarize_niah_sample_metrics`
        - 作用 / Purpose: 用固定 logits 验证 target rank、top-k、confidence、margin、entropy
          和 sample-level rows，避免评估报告退化为单一 accuracy/loss
        - 错误处理 / Error handling: rank、top-k 或 margin 语义变化会触发断言失败
        - 关键词 / Keywords:
          metrics|rank|topk|confidence|margin|entropy|sample|niah|logits|指标
        """
        logits = torch.tensor(
            [
                [0.0, 1.0, 0.5, 4.0, 2.0],
                [0.0, 5.0, 4.0, 3.0, 2.0],
            ]
        )
        targets = torch.tensor([3, 2], dtype=torch.long)
        query_positions = torch.tensor([4, 8], dtype=torch.long)

        rows = compute_selected_logits_sample_metrics(
            logits_target=logits,
            targets=targets,
            query_positions=query_positions,
            seq_len=10,
            depth=0.5,
            sample_index_start=7,
        )
        summary = summarize_niah_sample_metrics(rows)

        self.assertEqual([row["sample_index"] for row in rows], [7, 8])
        self.assertEqual(rows[0]["target_rank"], 1)
        self.assertEqual(rows[1]["target_rank"], 2)
        self.assertTrue(rows[0]["correct"])
        self.assertFalse(rows[1]["correct"])
        self.assertTrue(rows[1]["top3_correct"])
        self.assertGreater(rows[0]["logit_margin"], 0.0)
        self.assertLess(rows[1]["logit_margin"], 0.0)
        self.assertAlmostEqual(rows[0]["query_position_ratio"], 4 / 9)
        self.assertEqual(summary["total_samples"], 2)
        self.assertAlmostEqual(summary["top1_accuracy"], 0.5)
        self.assertAlmostEqual(summary["top3_accuracy"], 1.0)
        self.assertAlmostEqual(summary["median_target_rank"], 1.5)
        self.assertGreater(summary["mean_entropy"], 0.0)

    def test_markdown_report_contains_memory_and_accuracy(self):
        """Validate NIAH Markdown report includes reproducibility-critical fields.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `build_niah_verification_markdown`
        - 作用 / Purpose: 确认 reports/ Markdown 交付物包含准确率、显存和模型规模信息
        - 错误处理 / Error handling: 字段缺失会导致断言失败
        - 关键词 / Keywords:
          markdown|report|accuracy|memory|parameter_count|niah|reports|mhdsra2|test|报告
        """
        result = {
            "status": "success",
            "seq_len": 2_097_152,
            "best_accuracy": 1.0,
            "best_min_depth_accuracy": 1.0,
            "best_accuracy_step": 20,
            "best_accuracy_epoch": 20,
            "best_accuracy_loss": 0.01,
            "best_eval_source": "robust",
            "final_accuracy": 1.0,
            "final_min_depth_accuracy": 1.0,
            "final_eval_loss": 0.01,
            "final_eval_source": "robust",
            "re_eval_mean_accuracy": None,
            "re_eval_min_depth_accuracy": None,
            "best_step_train_accuracy": 1.0,
            "final_step_train_accuracy": 1.0,
            "final_train_loss": 0.01,
            "passed_accuracy": True,
            "passed_success_criteria": True,
            "final_loss": 0.01,
            "success_step": 20,
            "success_epoch": 20,
            "elapsed_sec": 1.25,
            "peak_memory_allocated_mb": 72.0,
            "peak_memory_reserved_mb": 86.0,
            "parameter_count": 123456,
            "device": "cuda:0",
            "cuda_device_name": "test-gpu",
            "torch_version": "2.x",
            "torch_cuda_version": "13.0",
            "config": {
                "vocab_size": 100,
                "dim": 64,
                "num_layers": 2,
                "slots": 64,
                "read_topk": 8,
                "chunk_size": 1024,
                "batch_size": 1,
                "epochs": 60,
                "optimizer_steps": 60,
                "learning_rate": 1e-3,
                "seed": 20260506,
                "target_accuracy": 1.0,
                "stop_loss": 0.1,
                "eval_interval": 20,
                "eval_batches_per_depth": 1,
                "light_eval_batches_per_depth": 1,
                "robust_eval_interval": 20,
                "robust_eval_batches_per_depth": 32,
                "eval_depths": [0.1, 0.5, 0.9],
                "cudnn_benchmark": False,
            },
            "steps_observed": [
                {
                    "optimizer_step": 0,
                    "epoch": 0,
                    "train_depth": 0.1,
                    "train_loss": 1.0,
                    "train_accuracy": 0.0,
                    "eval_mean_accuracy": 0.0,
                    "eval_min_depth_accuracy": 0.0,
                    "eval_mean_loss": 1.0,
                    "top3_accuracy": 0.0,
                    "mean_target_rank": 10.0,
                    "mean_entropy": 1.0,
                    "eval_depths": [],
                },
                {
                    "optimizer_step": 20,
                    "epoch": 20,
                    "train_depth": 0.9,
                    "train_loss": 0.01,
                    "train_accuracy": 1.0,
                    "eval_mean_accuracy": 1.0,
                    "eval_min_depth_accuracy": 1.0,
                    "eval_mean_loss": 0.01,
                    "top3_accuracy": 1.0,
                    "mean_target_rank": 1.0,
                    "mean_entropy": 0.1,
                    "eval_depths": [
                        {
                            "depth": 0.9,
                            "accuracy": 1.0,
                            "loss": 0.01,
                            "top3_accuracy": 1.0,
                            "mean_target_rank": 1.0,
                            "mean_target_prob": 0.9,
                            "mean_logit_margin": 2.0,
                            "mean_entropy": 0.1,
                            "samples": 1,
                        }
                    ],
                },
            ],
            "epochs_observed": [],
            "robust_evals_observed": [
                {
                    "optimizer_step": 20,
                    "eval_mean_accuracy": 1.0,
                    "eval_min_depth_accuracy": 1.0,
                    "eval_mean_loss": 0.01,
                    "top3_accuracy": 1.0,
                    "mean_target_rank": 1.0,
                    "mean_entropy": 0.1,
                    "total_samples": 96,
                    "eval_depths": [],
                }
            ],
            "sample_metrics": [],
        }

        markdown = "\n".join(build_niah_verification_markdown("Title", result))

        self.assertIn("final eval mean accuracy", markdown)
        self.assertIn("final eval min-depth accuracy", markdown)
        self.assertIn("best eval mean accuracy", markdown)
        self.assertIn("best accuracy step", markdown)
        self.assertIn("Robust Evaluations", markdown)
        self.assertIn("Latest Light Per-Depth Metrics", markdown)
        self.assertIn("passed success criteria", markdown)
        self.assertIn("peak allocated memory", markdown)
        self.assertIn("parameter count", markdown)
        self.assertIn("2.11", markdown.replace("2.x", "2.11"))

    def test_save_verification_report_writes_nonempty_files(self):
        """Validate report persistence verifies non-empty JSON and Markdown outputs.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `save_niah_verification_report`
        - 作用 / Purpose: 防止 reports/ 写入静默失败后仍返回成功路径
        - 错误处理 / Error handling: 文件缺失或为空会由被测函数抛出
        - 关键词 / Keywords:
          report|write|verify|json|markdown|niah|filesystem|test|nonempty|报告
        """
        result = {
            "status": "completed",
            "seq_len": 64,
            "best_accuracy": 0.0,
            "final_accuracy": 0.0,
            "final_min_depth_accuracy": 0.0,
            "re_eval_mean_accuracy": None,
            "re_eval_min_depth_accuracy": None,
            "best_step_train_accuracy": 0.0,
            "final_step_train_accuracy": 0.0,
            "passed_accuracy": False,
            "passed_success_criteria": False,
            "final_loss": 1.0,
            "elapsed_sec": 0.1,
            "peak_memory_allocated_mb": 0.0,
            "peak_memory_reserved_mb": 0.0,
            "parameter_count": 10,
            "device": "cpu",
            "cuda_device_name": None,
            "torch_version": "2.x",
            "torch_cuda_version": None,
            "config": {
                "vocab_size": 32,
                "dim": 16,
                "num_layers": 1,
                "slots": 8,
                "read_topk": 2,
                "chunk_size": 8,
                "batch_size": 1,
                "epochs": 1,
                "optimizer_steps": 1,
                "learning_rate": 1e-3,
                "seed": 123,
                "target_accuracy": 1.0,
                "stop_loss": 0.1,
                "eval_interval": 1,
                "eval_batches_per_depth": 1,
                "light_eval_batches_per_depth": 1,
                "robust_eval_interval": 1,
                "robust_eval_batches_per_depth": 32,
                "eval_depths": [0.1, 0.5, 0.9],
                "cudnn_benchmark": False,
            },
            "steps_observed": [
                {
                    "optimizer_step": 0,
                    "epoch": 0,
                    "train_depth": 0.1,
                    "train_loss": 1.0,
                    "train_accuracy": 0.0,
                    "eval_mean_accuracy": 0.0,
                    "eval_min_depth_accuracy": 0.0,
                    "eval_mean_loss": 1.0,
                    "eval_depths": [],
                }
            ],
            "epochs_observed": [],
            "robust_evals_observed": [],
            "sample_metrics": [],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = save_niah_verification_report(result, tmp_dir, "niah_test", "NIAH Test")

            self.assertGreater(Path(paths["json"]).stat().st_size, 0)
            self.assertGreater(Path(paths["markdown"]).stat().st_size, 0)

    def test_save_capacity_reports_writes_nonempty_files(self):
        """Validate capacity report persistence verifies non-empty outputs.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `save_niah_capacity_reports`
        - 作用 / Purpose: 防止 capacity sweep 报告写入静默失败后仍被当作可用证据
        - 错误处理 / Error handling: 文件缺失或为空会由被测函数抛出
        - 关键词 / Keywords:
          capacity|report|write|verify|json|markdown|niah|test|nonempty|容量
        """
        result = {
            64: {
                "status": "ok",
                "accuracy": 0.5,
                "min_depth_accuracy": 0.0,
                "peak_mem_mb": 0.0,
                "depth_results": [],
                "seed": 123,
            }
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = save_niah_capacity_reports(result, result, tmp_dir)

            self.assertGreater(Path(paths["json"]).stat().st_size, 0)
            self.assertGreater(Path(paths["markdown"]).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
