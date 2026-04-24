import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import scripts.attention_family_benchmark as benchmark_module
from attention_family_benchmark import benchmark_attention_family_complexity
from attention_family_benchmark import aggregate_seeded_task_runs, build_task_seed_bundle
from json_retrieval_test import VOCAB_SIZE, build_retrieval_model


class TestAttentionFamilyBenchmark(unittest.TestCase):
    def test_build_retrieval_model_supports_attention_family_variants(self):
        model_types = (
            "dsra",
            "sparse_attention",
            "sliding_window_attention",
            "linear_attention",
            "standard_attention",
        )
        x = torch.randint(0, VOCAB_SIZE, (1, 12), dtype=torch.long)

        for model_type in model_types:
            with self.subTest(model_type=model_type):
                model = build_retrieval_model(
                    model_type=model_type,
                    vocab_size=VOCAB_SIZE,
                    dim=8,
                    K=8,
                    kr=2,
                    chunk_size=4,
                    local_context_size=4,
                    local_context_mode="concat",
                )
                logits, hidden = model(x, return_hidden=True)
                self.assertEqual(logits.shape, (1, 12, VOCAB_SIZE))
                self.assertEqual(hidden.shape, (1, 12, 8))

    def test_benchmark_attention_family_complexity_smoke(self):
        results = benchmark_attention_family_complexity(
            model_types=("dsra", "linear_attention"),
            seq_lengths=[16],
            dim=8,
            K=8,
            kr=2,
            chunk_size=4,
            batch_size=1,
            warmup_runs=0,
            measured_runs=1,
            local_context_size=4,
            local_context_mode="concat",
        )

        self.assertEqual(results["seq_lengths"], [16])
        for model_type in ("dsra", "linear_attention"):
            model_result = results["models"][model_type]["length_results"][0]
            self.assertEqual(model_result["status"], "ok")
            self.assertGreaterEqual(model_result["mean_time_ms"], 0.0)

    def test_build_task_seed_bundle_offsets_are_stable(self):
        seed_bundle = build_task_seed_bundle(7)

        self.assertEqual(seed_bundle["train_dataset_seed"], 7)
        self.assertEqual(seed_bundle["validation_dataset_seed"], 108)
        self.assertEqual(seed_bundle["test_dataset_seed"], 209)
        self.assertEqual(seed_bundle["pair_split_seed"], 310)
        self.assertEqual(seed_bundle["torch_seed"], 411)
        self.assertEqual(seed_bundle["python_seed"], 512)

    def test_aggregate_seeded_task_runs_computes_mean_std_and_ignores_missing(self):
        seed_runs = [
            {
                "seed_root": 7,
                "metrics": {
                    "test_teacher_forced_seq_acc": 0.8,
                    "test_extract_exact": 0.5,
                    "test_evidence_window_acc": None,
                },
            },
            {
                "seed_root": 11,
                "metrics": {
                    "test_teacher_forced_seq_acc": 0.6,
                    "test_extract_exact": 0.25,
                    "test_evidence_window_acc": None,
                },
            },
        ]

        aggregated = aggregate_seeded_task_runs(seed_runs)

        self.assertAlmostEqual(aggregated["test_teacher_forced_seq_acc"]["mean"], 0.7)
        self.assertAlmostEqual(aggregated["test_extract_exact"]["mean"], 0.375)
        self.assertEqual(aggregated["test_teacher_forced_seq_acc"]["n"], 2)
        self.assertIsNone(aggregated["test_evidence_window_acc"])

    def test_run_attention_family_json_retrieval_benchmark_keeps_exact_report_root(self):
        original_run = benchmark_module.run_json_retrieval_generalization_test
        original_build = benchmark_module.build_retrieval_model

        class _StubModel:
            window_size = 4

        def _fake_run(**kwargs):
            reports_dir = Path(kwargs["reports_dir"])
            reports_dir.mkdir(parents=True, exist_ok=True)
            return {
                "config": {"model_type": kwargs["model_type"]},
                "validation_pool_evaluation": {
                    "generation_exact_match_rate": 0.0,
                    "generation_mean_sequence_accuracy": 0.5,
                    "teacher_forced_exact_match_rate": 0.0,
                    "teacher_forced_mean_sequence_accuracy": 0.75,
                    "evidence_window_accuracy": None,
                    "extract_then_compose_exact_match_rate": None,
                    "extract_then_compose_mean_sequence_accuracy": None,
                },
                "test_pool_evaluation": {
                    "generation_exact_match_rate": 0.0,
                    "generation_mean_sequence_accuracy": 0.4,
                    "teacher_forced_exact_match_rate": 0.0,
                    "teacher_forced_mean_sequence_accuracy": 0.7,
                    "evidence_window_accuracy": None,
                    "extract_then_compose_exact_match_rate": None,
                    "extract_then_compose_mean_sequence_accuracy": None,
                },
            }

        def _fake_build(**kwargs):
            return _StubModel()

        benchmark_module.run_json_retrieval_generalization_test = _fake_run
        benchmark_module.build_retrieval_model = _fake_build
        try:
            with TemporaryDirectory() as tmp_dir:
                output_root = Path(tmp_dir) / "attention_family_json_retrieval"
                results = benchmark_module.run_attention_family_json_retrieval_benchmark(
                    reports_dir=output_root,
                    model_types=("dsra",),
                    task_variants=("baseline",),
                    task_seed_roots=(7,),
                    generalization_kwargs={"dim": 8, "K": 8, "kr_grid": [2], "chunk_size_grid": [4]},
                )

                expected_seed_report = output_root / "baseline" / "dsra" / "seed_7" / "reports"
                self.assertTrue(expected_seed_report.exists())
                self.assertFalse((output_root / "reports").exists())
                self.assertEqual(
                    results["task_variants"]["baseline"]["model_results"][0]["seed_runs"][0]["report_dir"],
                    str(expected_seed_report),
                )
        finally:
            benchmark_module.run_json_retrieval_generalization_test = original_run
            benchmark_module.build_retrieval_model = original_build


if __name__ == "__main__":
    unittest.main()
