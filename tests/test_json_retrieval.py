import random
import unittest

import torch

from json_retrieval_test import (
    ANSWER_START_TOKEN_ID,
    PAD_TOKEN_ID,
    QUESTION_TOKEN_ID,
    VOCAB_SIZE,
    build_random_training_case_pool,
    build_scheduled_sampling_inputs,
    build_scheduled_sampling_ratio_schedule,
    evaluate_teacher_forced,
    generate_random_json_retrieval_case,
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
        self.assertIn(b"What is the most valuable exhibit in the ", generated_case["question_bytes"])

    def test_build_random_training_case_pool_creates_diverse_cases(self):
        reference_case = {
            "sample_bytes": b"x" * 256,
            "metadata": {"target_total_bytes": 256},
        }

        case_pool = build_random_training_case_pool(reference_case, dataset_size=6, seed=11)
        unique_answers = {case["metadata"]["expected_answer_text"] for case in case_pool}

        self.assertEqual(len(case_pool), 6)
        self.assertGreater(len(unique_answers), 1)


if __name__ == "__main__":
    unittest.main()
