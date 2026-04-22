import random
import unittest

import torch

from json_retrieval_test import (
    ANSWER_START_TOKEN_ID,
    PAD_TOKEN_ID,
    QUESTION_TOKEN_ID,
    VOCAB_SIZE,
    build_answer_complexity_case,
    build_random_training_case_pool,
    build_answer_text_for_complexity,
    build_search_space,
    build_scheduled_sampling_inputs,
    build_scheduled_sampling_ratio_schedule,
    evaluate_teacher_forced,
    generate_random_json_retrieval_case,
    resolve_fixed_sample_ratio,
    resolve_answer_complexity_level,
)


class TestJsonRetrieval(unittest.TestCase):
    def test_evaluate_teacher_forced_ignores_special_tokens_in_predictions(self):
        case = {"expected_answer_bytes": b"AB"}
        y = torch.tensor([[PAD_TOKEN_ID, 65, 66]], dtype=torch.long)
        logits = torch.full((1, 3, VOCAB_SIZE), -1000.0)

        # The answer positions incorrectly rank special tokens highest overall.
        # Evaluation should still decode using the byte-only sub-vocabulary.
        logits[0, 1, ANSWER_START_TOKEN_ID] = 100.0
        logits[0, 1, 65] = 90.0
        logits[0, 2, QUESTION_TOKEN_ID] = 100.0
        logits[0, 2, 66] = 90.0

        result = evaluate_teacher_forced(logits, y, case)

        self.assertEqual(result["predicted_tokens"], [65, 66])
        self.assertEqual(result["predicted_text"], "AB")
        self.assertTrue(result["exact_byte_match"])

    def test_build_scheduled_sampling_inputs_replaces_only_answer_prefix_tokens(self):
        x = torch.tensor([[10, 11, ANSWER_START_TOKEN_ID, 65, 66]], dtype=torch.long)
        y = torch.tensor([[PAD_TOKEN_ID, PAD_TOKEN_ID, 65, 66, 67]], dtype=torch.long)
        predicted_tokens = torch.tensor([70, 71, 72], dtype=torch.long)
        sample_mask = torch.tensor([True, False], dtype=torch.bool)

        sampled_x, sampled_count = build_scheduled_sampling_inputs(
            x,
            y,
            predicted_tokens,
            sampling_ratio=0.5,
            sample_mask=sample_mask,
        )

        self.assertEqual(sampled_count, 1)
        self.assertEqual(sampled_x.tolist(), [[10, 11, ANSWER_START_TOKEN_ID, 70, 66]])

    def test_build_scheduled_sampling_inputs_aligns_candidate_token_dtype(self):
        x = torch.tensor([[10, 11, ANSWER_START_TOKEN_ID, 65, 66]], dtype=torch.long)
        y = torch.tensor([[PAD_TOKEN_ID, PAD_TOKEN_ID, 65, 66, 67]], dtype=torch.long)
        predicted_tokens = torch.tensor([70.0, 71.0, 72.0], dtype=torch.float32)
        sample_mask = torch.tensor([1, 0], dtype=torch.int64)

        sampled_x, sampled_count = build_scheduled_sampling_inputs(
            x,
            y,
            predicted_tokens,
            sampling_ratio=0.5,
            sample_mask=sample_mask,
        )

        self.assertEqual(sampled_count, 1)
        self.assertEqual(sampled_x.dtype, x.dtype)
        self.assertEqual(sampled_x.device, x.device)
        self.assertEqual(sampled_x.tolist(), [[10, 11, ANSWER_START_TOKEN_ID, 70, 66]])

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA 以验证跨设备对齐")
    def test_build_scheduled_sampling_inputs_aligns_predicted_tokens_and_mask_device(self):
        x = torch.tensor([[10, 11, ANSWER_START_TOKEN_ID, 65, 66]], dtype=torch.long, device="cpu")
        y = torch.tensor([[PAD_TOKEN_ID, PAD_TOKEN_ID, 65, 66, 67]], dtype=torch.long, device="cpu")
        predicted_tokens = torch.tensor([70, 71, 72], dtype=torch.long, device="cuda")
        sample_mask = torch.tensor([True, False], dtype=torch.bool, device="cuda")

        sampled_x, sampled_count = build_scheduled_sampling_inputs(
            x,
            y,
            predicted_tokens,
            sampling_ratio=0.5,
            sample_mask=sample_mask,
        )

        self.assertEqual(sampled_count, 1)
        self.assertEqual(sampled_x.device.type, "cpu")
        self.assertEqual(sampled_x.tolist(), [[10, 11, ANSWER_START_TOKEN_ID, 70, 66]])

    def test_build_scheduled_sampling_ratio_schedule_linearly_reaches_max_ratio(self):
        schedule = build_scheduled_sampling_ratio_schedule(total_steps=5, max_ratio=0.4)

        self.assertEqual(schedule(0), 0.0)
        self.assertAlmostEqual(schedule(2), 0.2)
        self.assertAlmostEqual(schedule(4), 0.4)
        self.assertAlmostEqual(schedule(99), 0.4)

    def test_generate_random_json_retrieval_case_produces_consistent_metadata(self):
        reference_case = {
            "sample_bytes": b"x" * 256,
            "metadata": {"target_total_bytes": 256},
        }

        generated_case = generate_random_json_retrieval_case(reference_case, rng=random.Random(3))

        insert_at = generated_case["metadata"]["insert_position_byte_index"]
        needle_bytes = generated_case["expected_answer_bytes"]
        needle_end = insert_at + len(needle_bytes)

        self.assertEqual(len(generated_case["sample_bytes"]), 256)
        self.assertEqual(generated_case["metadata"]["actual_total_bytes"], 256)
        self.assertEqual(generated_case["metadata"]["needle_bytes"], len(needle_bytes))
        self.assertEqual(generated_case["sample_bytes"][insert_at:needle_end], needle_bytes)
        self.assertEqual(
            generated_case["question_bytes"],
            generated_case["metadata"]["question"].encode("ascii"),
        )
        self.assertIn("question_template", generated_case["metadata"])
        self.assertIn("answer_template", generated_case["metadata"])

    def test_generate_random_json_retrieval_case_scales_answer_by_complexity(self):
        reference_case = {
            "sample_bytes": b"x" * 256,
            "metadata": {"target_total_bytes": 256},
        }

        short_case = generate_random_json_retrieval_case(
            reference_case,
            rng=random.Random(3),
            answer_complexity_level=1,
        )
        medium_case = generate_random_json_retrieval_case(
            reference_case,
            rng=random.Random(3),
            answer_complexity_level=2,
        )
        full_case = generate_random_json_retrieval_case(
            reference_case,
            rng=random.Random(3),
            answer_complexity_level=3,
        )

        self.assertLess(
            len(short_case["expected_answer_bytes"]),
            len(medium_case["expected_answer_bytes"]),
        )
        self.assertLess(
            len(medium_case["expected_answer_bytes"]),
            len(full_case["expected_answer_bytes"]),
        )
        self.assertEqual(short_case["metadata"]["answer_complexity_level"], 1)
        self.assertEqual(medium_case["metadata"]["answer_complexity_level"], 2)
        self.assertEqual(full_case["metadata"]["answer_complexity_level"], 3)

    def test_build_answer_text_for_complexity_returns_progressive_answer_lengths(self):
        full_answer = (
            "The most valuable exhibit in the Palace Museum is Along the River During "
            "the Qingming Festival painted by Zhang Zeduan of the Northern Song dynasty."
        )

        level1_answer = build_answer_text_for_complexity(
            full_answer_text=full_answer,
            answer_complexity_level=1,
            museum="Palace Museum",
            artifact="Along the River During the Qingming Festival",
            artist="Zhang Zeduan",
            dynasty="Northern Song",
        )
        level2_answer = build_answer_text_for_complexity(
            full_answer_text=full_answer,
            answer_complexity_level=2,
            museum="Palace Museum",
            artifact="Along the River During the Qingming Festival",
            artist="Zhang Zeduan",
            dynasty="Northern Song",
        )
        level3_answer = build_answer_text_for_complexity(
            full_answer_text=full_answer,
            answer_complexity_level=3,
            museum="Palace Museum",
            artifact="Along the River During the Qingming Festival",
            artist="Zhang Zeduan",
            dynasty="Northern Song",
        )

        self.assertEqual(level1_answer, "Along the River During the Qingming Festival")
        self.assertEqual(
            level2_answer,
            "Along the River During the Qingming Festival by Zhang Zeduan",
        )
        self.assertEqual(level3_answer, full_answer)

    def test_build_answer_complexity_case_preserves_reference_question_and_shortens_answer(self):
        reference_case = {
            "sample_bytes": b"x" * 256,
            "question_bytes": b"What is the most valuable exhibit in the Palace Museum?",
            "expected_answer_bytes": (
                b"The most valuable exhibit in the Palace Museum is Along the River During "
                b"the Qingming Festival painted by Zhang Zeduan of the Northern Song dynasty."
            ),
            "metadata": {
                "target_total_bytes": 256,
                "needle_position_pct": 0.4,
                "question": "What is the most valuable exhibit in the Palace Museum?",
                "expected_answer_text": (
                    "The most valuable exhibit in the Palace Museum is Along the River During "
                    "the Qingming Festival painted by Zhang Zeduan of the Northern Song dynasty."
                ),
                "museum": "Palace Museum",
                "artifact": "Along the River During the Qingming Festival",
                "artist": "Zhang Zeduan",
                "dynasty": "Northern Song",
            },
        }

        short_case = build_answer_complexity_case(reference_case, answer_complexity_level=1)

        self.assertEqual(
            short_case["metadata"]["question"],
            reference_case["metadata"]["question"],
        )
        self.assertEqual(
            short_case["metadata"]["expected_answer_text"],
            "Along the River During the Qingming Festival",
        )
        self.assertLess(
            len(short_case["expected_answer_bytes"]),
            len(reference_case["expected_answer_bytes"]),
        )

    def test_build_random_training_case_pool_creates_diverse_cases(self):
        reference_case = {
            "sample_bytes": b"x" * 256,
            "metadata": {"target_total_bytes": 256},
        }

        case_pool = build_random_training_case_pool(reference_case, dataset_size=6, seed=11)
        unique_answers = {case["metadata"]["expected_answer_text"] for case in case_pool}

        self.assertEqual(len(case_pool), 6)
        self.assertGreater(len(unique_answers), 1)

    def test_build_random_training_case_pool_varies_questions_and_answer_lengths(self):
        reference_case = {
            "sample_bytes": b"x" * 256,
            "metadata": {"target_total_bytes": 256},
        }

        case_pool = build_random_training_case_pool(reference_case, dataset_size=24, seed=21)
        unique_questions = {case["metadata"]["question"] for case in case_pool}
        unique_answer_lengths = {case["metadata"]["answer_length_bytes"] for case in case_pool}

        self.assertGreater(len(unique_questions), 1)
        self.assertGreater(len(unique_answer_lengths), 1)

    def test_build_random_training_case_pool_respects_answer_complexity_level(self):
        reference_case = {
            "sample_bytes": b"x" * 256,
            "question_bytes": b"What is the most valuable exhibit in the Palace Museum?",
            "expected_answer_bytes": (
                b"The most valuable exhibit in the Palace Museum is Along the River During "
                b"the Qingming Festival painted by Zhang Zeduan of the Northern Song dynasty."
            ),
            "metadata": {
                "target_total_bytes": 256,
                "needle_position_pct": 0.4,
                "question": "What is the most valuable exhibit in the Palace Museum?",
                "expected_answer_text": (
                    "The most valuable exhibit in the Palace Museum is Along the River During "
                    "the Qingming Festival painted by Zhang Zeduan of the Northern Song dynasty."
                ),
                "museum": "Palace Museum",
                "artifact": "Along the River During the Qingming Festival",
                "artist": "Zhang Zeduan",
                "dynasty": "Northern Song",
            },
        }

        case_pool = build_random_training_case_pool(
            reference_case,
            dataset_size=8,
            seed=21,
            fixed_ratio=0.5,
            answer_complexity_level=1,
        )

        self.assertEqual(len(case_pool), 8)
        self.assertTrue(
            all(case["metadata"]["answer_complexity_level"] == 1 for case in case_pool)
        )

    def test_resolve_answer_complexity_level_progresses_with_epoch(self):
        self.assertEqual(resolve_answer_complexity_level(1, 9), 1)
        self.assertEqual(resolve_answer_complexity_level(3, 9), 1)
        self.assertEqual(resolve_answer_complexity_level(4, 9), 2)
        self.assertEqual(resolve_answer_complexity_level(6, 9), 2)
        self.assertEqual(resolve_answer_complexity_level(7, 9), 3)
        self.assertEqual(resolve_answer_complexity_level(9, 9), 3)

    def test_resolve_fixed_sample_ratio_progresses_toward_target_ratio(self):
        self.assertEqual(resolve_fixed_sample_ratio(1, 9, 0.3), 0.8)
        self.assertEqual(resolve_fixed_sample_ratio(3, 9, 0.3), 0.8)
        self.assertEqual(resolve_fixed_sample_ratio(4, 9, 0.3), 0.5)
        self.assertEqual(resolve_fixed_sample_ratio(6, 9, 0.3), 0.5)
        self.assertEqual(resolve_fixed_sample_ratio(7, 9, 0.3), 0.3)
        self.assertEqual(resolve_fixed_sample_ratio(9, 9, 0.3), 0.3)

    def test_build_search_space_includes_train_dataset_size_dimension(self):
        search_space = build_search_space(
            kr_grid=[8, 16],
            chunk_size_grid=[256],
            lr_grid=[1e-3],
            warmup_ratio_grid=[0.1],
            scheduled_sampling_max_ratio_grid=[0.2, 0.3],
            train_dataset_size_grid=[32, 64],
        )

        self.assertEqual(len(search_space), 8)
        self.assertIn((8, 256, 1e-3, 0.1, 0.2, 32), search_space)
        self.assertIn((16, 256, 1e-3, 0.1, 0.3, 64), search_space)


if __name__ == "__main__":
    unittest.main()
