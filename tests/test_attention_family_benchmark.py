import unittest

import torch

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


if __name__ == "__main__":
    unittest.main()
