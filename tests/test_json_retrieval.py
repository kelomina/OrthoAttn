import random
import unittest
from collections import deque

import torch

from json_retrieval_test import (
    ANSWER_START_TOKEN_ID,
    ALL_MUSEUM_ARTIFACT_PAIRS,
    PAD_TOKEN_ID,
    QUESTION_TOKEN_ID,
    VOCAB_SIZE,
    aggregate_case_pool_results,
    build_tail_error_analysis,
    build_case_signature,
    build_disjoint_case_pool,
    build_generation_prompt,
    build_random_training_case_pool,
    get_case_museum_artifact_pair,
    get_answer_entity_spans,
    build_scheduled_sampling_inputs,
    build_scheduled_sampling_ratio_schedule,
    compute_weighted_answer_loss,
    evaluate_teacher_forced,
    generate_random_json_retrieval_case,
    get_answer_start_index,
    rollout_generation_logits,
    score_generalization_result,
    select_case_batch,
    select_training_case,
    split_museum_artifact_pairs,
    get_evidence_window_target,
    get_answer_slot_spans,
    extract_slot_labels_from_window_bytes,
)
from toy_task_associative_recall import DSRAModel, StandardAttentionModel


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

    def test_build_disjoint_case_pool_avoids_cross_pool_signature_overlap(self):
        reference_case = {
            "sample_bytes": b"x" * 256,
            "metadata": {"target_total_bytes": 256},
        }

        train_pool, used_signatures = build_disjoint_case_pool(reference_case, dataset_size=4, seed=7)
        val_pool, used_signatures = build_disjoint_case_pool(
            reference_case,
            dataset_size=4,
            seed=17,
            used_signatures=used_signatures,
        )

        train_signatures = {build_case_signature(case) for case in train_pool}
        val_signatures = {build_case_signature(case) for case in val_pool}

        self.assertEqual(len(train_pool), 4)
        self.assertEqual(len(val_pool), 4)
        self.assertFalse(train_signatures & val_signatures)

    def test_split_museum_artifact_pairs_is_disjoint_and_complete(self):
        pair_split = split_museum_artifact_pairs(seed=29)
        train_pairs = set(pair_split["train"])
        validation_pairs = set(pair_split["validation"])
        test_pairs = set(pair_split["test"])

        self.assertFalse(train_pairs & validation_pairs)
        self.assertFalse(train_pairs & test_pairs)
        self.assertFalse(validation_pairs & test_pairs)
        self.assertEqual(train_pairs | validation_pairs | test_pairs, set(ALL_MUSEUM_ARTIFACT_PAIRS))

    def test_build_disjoint_case_pool_respects_allowed_museum_artifact_pairs(self):
        reference_case = {
            "sample_bytes": b"x" * 256,
            "metadata": {"target_total_bytes": 256},
        }
        allowed_pairs = {("Palace Museum", "Autumn Lantern Procession")}

        case_pool, _ = build_disjoint_case_pool(
            reference_case,
            dataset_size=3,
            seed=5,
            allowed_museum_artifact_pairs=allowed_pairs,
        )

        self.assertEqual(len(case_pool), 3)
        self.assertEqual(
            {get_case_museum_artifact_pair(case) for case in case_pool},
            allowed_pairs,
        )

    def test_select_training_case_can_force_reference_case(self):
        reference_case = {"name": "reference"}
        training_cases = [{"name": "pool-a"}, {"name": "pool-b"}]

        selected_case = select_training_case(
            training_cases=training_cases,
            reference_case=reference_case,
            sampler=random.Random(5),
            target_case_sampling_ratio=1.0,
        )

        self.assertIs(selected_case, reference_case)

    def test_select_training_case_uses_pool_when_ratio_is_zero(self):
        reference_case = {"name": "reference"}
        training_cases = [{"name": "pool-a"}, {"name": "pool-b"}]

        selected_case = select_training_case(
            training_cases=training_cases,
            reference_case=reference_case,
            sampler=random.Random(5),
            target_case_sampling_ratio=0.0,
        )

        self.assertIn(selected_case, training_cases)
        self.assertIsNot(selected_case, reference_case)

    def test_concat_local_context_matches_step_context(self):
        model = DSRAModel(
            vocab_size=VOCAB_SIZE,
            dim=8,
            K=8,
            kr=2,
            local_context_mode="concat",
            local_context_size=4,
        )
        raw_emb = torch.randn(1, 6, 8)
        full_context = model.build_causal_context(raw_emb)

        for end_idx in range(raw_emb.shape[1]):
            history = deque(
                [raw_emb[:, idx:idx + 1, :] for idx in range(max(0, end_idx - 3), end_idx + 1)],
                maxlen=model.local_context_size,
            )
            step_context = model.build_step_context(history)
            self.assertTrue(
                torch.allclose(
                    step_context,
                    full_context[:, end_idx:end_idx + 1, :],
                    atol=1e-6,
                    rtol=1e-5,
                )
            )

    def test_build_generation_prompt_appends_question_and_answer_start(self):
        case = {
            "sample_bytes": b"ctx",
            "question_bytes": b"q?",
            "expected_answer_bytes": b"AB",
            "metadata": {"needle_position_pct": 0.5, "insert_position_byte_index": 1, "needle_bytes": 2},
        }

        prompt_tokens, answer_tokens = build_generation_prompt(case)

        self.assertEqual(prompt_tokens, [99, 116, 120, QUESTION_TOKEN_ID, 113, 63, ANSWER_START_TOKEN_ID])
        self.assertEqual(answer_tokens, [65, 66])

    def test_get_answer_start_index_matches_prompt_length(self):
        case = {
            "sample_bytes": b"context",
            "question_bytes": b"q?",
            "expected_answer_bytes": b"AB",
            "metadata": {"needle_position_pct": 0.5, "insert_position_byte_index": 2, "needle_bytes": 2},
        }

        prompt_tokens, _ = build_generation_prompt(case)
        answer_start_index = get_answer_start_index(case)

        self.assertEqual(answer_start_index, len(prompt_tokens) - 1)

    def test_get_answer_entity_spans_locates_museum_and_artifact(self):
        expected = (
            b"The most valuable exhibit in the Palace Museum is Autumn Lantern Procession "
            b"painted by Lin Qiao of the Tang dynasty."
        )
        case = {
            "expected_answer_bytes": expected,
            "metadata": {
                "museum": "Palace Museum",
                "artifact": "Autumn Lantern Procession",
            },
        }

        spans = get_answer_entity_spans(case)

        self.assertEqual(spans["museum"]["label"], "Palace Museum")
        self.assertEqual(expected[spans["museum"]["start"]:spans["museum"]["end"]], b"Palace Museum")
        self.assertEqual(spans["artifact"]["label"], "Autumn Lantern Procession")
        self.assertEqual(
            expected[spans["artifact"]["start"]:spans["artifact"]["end"]],
            b"Autumn Lantern Procession",
        )

    def test_get_answer_slot_spans_locates_all_answer_slots(self):
        expected = (
            b"The most valuable exhibit in the Palace Museum is Autumn Lantern Procession "
            b"painted by Lin Qiao of the Tang dynasty."
        )
        case = {
            "expected_answer_bytes": expected,
            "metadata": {
                "museum": "Palace Museum",
                "artifact": "Autumn Lantern Procession",
                "artist": "Lin Qiao",
                "dynasty": "Tang",
            },
        }

        spans = get_answer_slot_spans(case)

        self.assertEqual(expected[spans["artist"]["start"]:spans["artist"]["end"]], b"Lin Qiao")
        self.assertEqual(expected[spans["dynasty"]["start"]:spans["dynasty"]["end"]], b"Tang")

    def test_get_evidence_window_target_tracks_curriculum_relative_position(self):
        case = {
            "sample_bytes": b"a" * 64 + b"NEEDLE" + b"b" * 57,
            "question_bytes": b"q?",
            "expected_answer_bytes": b"answer",
            "metadata": {
                "insert_position_byte_index": 64,
                "needle_bytes": 6,
                "needle_position_pct": 0.5,
            },
        }

        full_target = get_evidence_window_target(case, context_bytes=None, window_count=8)
        cropped_target = get_evidence_window_target(case, context_bytes=64, window_count=8)

        self.assertEqual(full_target, 4)
        self.assertIn(cropped_target, {3, 4})

    def test_extract_slot_labels_from_window_bytes_reads_answer_template_fields(self):
        window_bytes = (
            b"noise "
            b"The most valuable exhibit in the Palace Museum is Autumn Lantern Procession "
            b"painted by Lin Qiao of the Tang dynasty. "
            b"tail"
        )

        extracted_slots = extract_slot_labels_from_window_bytes(window_bytes)

        self.assertEqual(extracted_slots["museum"], "Palace Museum")
        self.assertEqual(extracted_slots["artifact"], "Autumn Lantern Procession")
        self.assertEqual(extracted_slots["artist"], "Lin Qiao")
        self.assertEqual(extracted_slots["dynasty"], "Tang")

    def test_compute_weighted_answer_loss_upweights_entity_span_errors(self):
        expected = (
            b"The most valuable exhibit in the Palace Museum is Autumn Lantern Procession "
            b"painted by Lin Qiao of the Tang dynasty."
        )
        case = {
            "expected_answer_bytes": expected,
            "metadata": {
                "museum": "Palace Museum",
                "artifact": "Autumn Lantern Procession",
            },
        }
        targets = torch.tensor(list(expected), dtype=torch.long)
        logits = torch.full((len(expected), VOCAB_SIZE), -8.0)
        logits[torch.arange(len(expected)), targets] = 8.0
        spans = get_answer_entity_spans(case)
        artifact_start = spans["artifact"]["start"]
        wrong_token = ord("X")
        logits[artifact_start, targets[artifact_start]] = 1.0
        logits[artifact_start, wrong_token] = 5.0

        baseline_loss = compute_weighted_answer_loss(logits, targets, case)
        weighted_loss = compute_weighted_answer_loss(
            logits,
            targets,
            case,
            artifact_span_loss_weight=4.0,
        )

        self.assertGreater(weighted_loss.item(), baseline_loss.item())

    def test_rollout_generation_logits_returns_one_logit_per_answer_token(self):
        case = {
            "sample_bytes": b"context",
            "question_bytes": b"q?",
            "expected_answer_bytes": b"ABC",
            "metadata": {"needle_position_pct": 0.5, "insert_position_byte_index": 2, "needle_bytes": 3},
        }
        prompt_tokens, answer_tokens = build_generation_prompt(case)
        model = DSRAModel(
            vocab_size=VOCAB_SIZE,
            dim=8,
            K=8,
            kr=2,
            local_context_mode="concat",
            local_context_size=4,
        )

        rollout_logits, sampled_count = rollout_generation_logits(
            model=model,
            case=case,
            prompt_tokens=prompt_tokens,
            answer_tokens=answer_tokens,
            device=torch.device("cpu"),
            self_feed_ratio=0.5,
        )

        self.assertEqual(rollout_logits.shape, (1, len(answer_tokens), VOCAB_SIZE))
        self.assertGreaterEqual(sampled_count, 0)
        self.assertLessEqual(sampled_count, len(answer_tokens) - 1)

    def test_standard_attention_model_matches_forward_interface(self):
        model = StandardAttentionModel(
            vocab_size=VOCAB_SIZE,
            dim=8,
            chunk_size=4,
            local_context_mode="concat",
            local_context_size=4,
        )
        x = torch.randint(0, VOCAB_SIZE, (1, 10), dtype=torch.long)

        logits, hidden = model(x, return_hidden=True)

        self.assertEqual(logits.shape, (1, 10, VOCAB_SIZE))
        self.assertEqual(hidden.shape, (1, 10, 8))

    def test_select_case_batch_samples_requested_episode_count(self):
        cases = [{"name": "a"}, {"name": "b"}]

        sampled_cases = select_case_batch(cases, sampler=random.Random(3), batch_size=5)

        self.assertEqual(len(sampled_cases), 5)
        self.assertTrue(all(sampled_case in cases for sampled_case in sampled_cases))

    def test_aggregate_case_pool_results_computes_exact_rates_and_means(self):
        case_results = [
            {
                "question": "q1",
                "museum": "Palace Museum",
                "artifact": "Autumn Lantern Procession",
                "answer_len": 4,
                "entity_auxiliary": {
                    "museum": {"available": True, "exact_match": True},
                    "artifact": {"available": True, "exact_match": False},
                },
                "slot_decoder": {
                    "available": True,
                    "all_slots_exact_match": False,
                    "museum": {"exact_match": True},
                    "artifact": {"exact_match": False},
                    "artist": {"exact_match": True},
                    "dynasty": {"exact_match": True},
                },
                "evidence_decoder": {
                    "available": True,
                    "exact_match": True,
                    "window_distance": 0,
                },
                "extract_then_compose": {
                    "available": True,
                    "exact_byte_match": True,
                    "sequence_accuracy": 1.0,
                    "prefix_match_length": 4,
                    "extracted_slots": {
                        "museum": {"exact_match": True},
                        "artifact": {"exact_match": True},
                        "artist": {"exact_match": True},
                        "dynasty": {"exact_match": True},
                    },
                },
                "teacher_forced": {
                    "exact_byte_match": True,
                    "sequence_accuracy": 1.0,
                    "prefix_match_length": 4,
                    "first_mismatch_index": None,
                    "tail_sequence_accuracy": 1.0,
                    "tail_exact_match": True,
                    "entity_span_metrics": {
                        "museum": {"exact_match": True, "sequence_accuracy": 1.0, "prefix_match_length": 2},
                        "artifact": {"exact_match": True, "sequence_accuracy": 1.0, "prefix_match_length": 2},
                    },
                },
                "generation": {
                    "exact_byte_match": False,
                    "sequence_accuracy": 0.5,
                    "prefix_match_length": 2,
                    "first_mismatch_index": 2,
                    "tail_sequence_accuracy": 0.5,
                    "tail_exact_match": False,
                    "entity_span_metrics": {
                        "museum": {"exact_match": False, "sequence_accuracy": 0.5, "prefix_match_length": 1},
                        "artifact": {"exact_match": False, "sequence_accuracy": 0.5, "prefix_match_length": 1},
                    },
                },
            },
            {
                "question": "q2",
                "museum": "Grand Archive Museum",
                "artifact": "Golden Crane Panorama",
                "answer_len": 4,
                "entity_auxiliary": {
                    "museum": {"available": True, "exact_match": False},
                    "artifact": {"available": True, "exact_match": True},
                },
                "slot_decoder": {
                    "available": True,
                    "all_slots_exact_match": False,
                    "museum": {"exact_match": False},
                    "artifact": {"exact_match": True},
                    "artist": {"exact_match": False},
                    "dynasty": {"exact_match": True},
                },
                "evidence_decoder": {
                    "available": True,
                    "exact_match": False,
                    "window_distance": 2,
                },
                "extract_then_compose": {
                    "available": True,
                    "exact_byte_match": False,
                    "sequence_accuracy": 0.5,
                    "prefix_match_length": 2,
                    "extracted_slots": {
                        "museum": {"exact_match": False},
                        "artifact": {"exact_match": True},
                        "artist": {"exact_match": False},
                        "dynasty": {"exact_match": True},
                    },
                },
                "teacher_forced": {
                    "exact_byte_match": False,
                    "sequence_accuracy": 0.25,
                    "prefix_match_length": 1,
                    "first_mismatch_index": 1,
                    "tail_sequence_accuracy": 0.25,
                    "tail_exact_match": False,
                    "entity_span_metrics": {
                        "museum": {"exact_match": False, "sequence_accuracy": 0.25, "prefix_match_length": 0},
                        "artifact": {"exact_match": True, "sequence_accuracy": 1.0, "prefix_match_length": 2},
                    },
                },
                "generation": {
                    "exact_byte_match": True,
                    "sequence_accuracy": 0.75,
                    "prefix_match_length": 3,
                    "first_mismatch_index": None,
                    "tail_sequence_accuracy": 0.75,
                    "tail_exact_match": True,
                    "entity_span_metrics": {
                        "museum": {"exact_match": True, "sequence_accuracy": 1.0, "prefix_match_length": 2},
                        "artifact": {"exact_match": False, "sequence_accuracy": 0.5, "prefix_match_length": 1},
                    },
                },
            },
        ]

        aggregated = aggregate_case_pool_results(case_results)

        self.assertEqual(aggregated["num_cases"], 2)
        self.assertAlmostEqual(aggregated["teacher_forced_exact_match_rate"], 0.5)
        self.assertAlmostEqual(aggregated["generation_exact_match_rate"], 0.5)
        self.assertAlmostEqual(aggregated["teacher_forced_mean_sequence_accuracy"], 0.625)
        self.assertAlmostEqual(aggregated["generation_mean_sequence_accuracy"], 0.625)
        self.assertAlmostEqual(aggregated["teacher_forced_mean_prefix_match_length"], 2.5)
        self.assertAlmostEqual(aggregated["generation_mean_prefix_match_length"], 2.5)
        self.assertAlmostEqual(aggregated["entity_aux_museum_accuracy"], 0.5)
        self.assertAlmostEqual(aggregated["entity_aux_artifact_accuracy"], 0.5)
        self.assertAlmostEqual(aggregated["slot_decoder_full_answer_accuracy"], 0.0)
        self.assertAlmostEqual(aggregated["slot_decoder_museum_accuracy"], 0.5)
        self.assertAlmostEqual(aggregated["slot_decoder_artifact_accuracy"], 0.5)
        self.assertAlmostEqual(aggregated["slot_decoder_artist_accuracy"], 0.5)
        self.assertAlmostEqual(aggregated["slot_decoder_dynasty_accuracy"], 1.0)
        self.assertAlmostEqual(aggregated["evidence_window_accuracy"], 0.5)
        self.assertAlmostEqual(aggregated["evidence_window_mean_distance"], 1.0)
        self.assertAlmostEqual(aggregated["extract_then_compose_exact_match_rate"], 0.5)
        self.assertAlmostEqual(aggregated["extract_then_compose_mean_sequence_accuracy"], 0.75)
        self.assertAlmostEqual(aggregated["extract_then_compose_mean_prefix_match_length"], 3.0)
        self.assertAlmostEqual(aggregated["extract_then_compose_museum_accuracy"], 0.5)
        self.assertAlmostEqual(aggregated["extract_then_compose_artifact_accuracy"], 1.0)
        self.assertAlmostEqual(aggregated["extract_then_compose_artist_accuracy"], 0.5)
        self.assertAlmostEqual(aggregated["extract_then_compose_dynasty_accuracy"], 1.0)
        self.assertAlmostEqual(aggregated["teacher_forced_museum_exact_match_rate"], 0.5)
        self.assertAlmostEqual(aggregated["teacher_forced_artifact_exact_match_rate"], 1.0)
        self.assertAlmostEqual(aggregated["generation_museum_exact_match_rate"], 0.5)
        self.assertAlmostEqual(aggregated["generation_artifact_mean_sequence_accuracy"], 0.5)

    def test_build_tail_error_analysis_summarizes_close_misses(self):
        pool_eval = {
            "case_results": [
                {
                    "question": "q1",
                    "museum": "Palace Museum",
                    "artifact": "Autumn Lantern Procession",
                    "answer_len": 8,
                    "teacher_forced": {
                        "exact_byte_match": False,
                        "sequence_accuracy": 0.75,
                        "prefix_match_length": 6,
                        "first_mismatch_index": 6,
                        "tail_sequence_accuracy": 0.5,
                        "tail_exact_match": False,
                    },
                    "generation": {
                        "exact_byte_match": False,
                        "sequence_accuracy": 0.5,
                        "prefix_match_length": 4,
                        "first_mismatch_index": 4,
                        "tail_sequence_accuracy": 0.25,
                        "tail_exact_match": False,
                    },
                },
                {
                    "question": "q2",
                    "museum": "Grand Archive Museum",
                    "artifact": "Golden Crane Panorama",
                    "answer_len": 8,
                    "teacher_forced": {
                        "exact_byte_match": True,
                        "sequence_accuracy": 1.0,
                        "prefix_match_length": 8,
                        "first_mismatch_index": None,
                        "tail_sequence_accuracy": 1.0,
                        "tail_exact_match": True,
                    },
                    "generation": {
                        "exact_byte_match": False,
                        "sequence_accuracy": 0.375,
                        "prefix_match_length": 3,
                        "first_mismatch_index": 3,
                        "tail_sequence_accuracy": 0.0,
                        "tail_exact_match": False,
                    },
                },
            ]
        }

        analysis = build_tail_error_analysis(pool_eval, tail_token_count=4, top_k=1)

        self.assertEqual(analysis["tail_token_count"], 4)
        self.assertAlmostEqual(analysis["teacher_forced"]["mean_tail_sequence_accuracy"], 0.75)
        self.assertAlmostEqual(analysis["generation"]["mean_tail_sequence_accuracy"], 0.125)
        self.assertEqual(len(analysis["teacher_forced"]["close_miss_cases"]), 1)
        self.assertEqual(analysis["teacher_forced"]["close_miss_cases"][0]["question"], "q1")
        self.assertAlmostEqual(analysis["teacher_forced"]["late_tail_failure_rate"], 1.0)

    def test_score_generalization_result_teacher_forced_mode_prefers_teacher_metrics(self):
        generation_better = {
            "validation_pool_evaluation": {
                "generation_exact_match_rate": 0.0,
                "teacher_forced_exact_match_rate": 0.0,
                "generation_mean_sequence_accuracy": 0.9,
                "teacher_forced_mean_sequence_accuracy": 0.5,
                "generation_mean_prefix_match_length": 20.0,
                "teacher_forced_mean_prefix_match_length": 10.0,
            },
            "test_pool_evaluation": {
                "generation_exact_match_rate": 0.0,
                "teacher_forced_exact_match_rate": 0.0,
                "generation_mean_sequence_accuracy": 0.8,
                "teacher_forced_mean_sequence_accuracy": 0.4,
                "generation_mean_prefix_match_length": 18.0,
                "teacher_forced_mean_prefix_match_length": 9.0,
            },
        }
        teacher_forced_better = {
            "validation_pool_evaluation": {
                "generation_exact_match_rate": 0.0,
                "teacher_forced_exact_match_rate": 0.0,
                "generation_mean_sequence_accuracy": 0.4,
                "teacher_forced_mean_sequence_accuracy": 0.8,
                "generation_mean_prefix_match_length": 8.0,
                "teacher_forced_mean_prefix_match_length": 25.0,
            },
            "test_pool_evaluation": {
                "generation_exact_match_rate": 0.0,
                "teacher_forced_exact_match_rate": 0.0,
                "generation_mean_sequence_accuracy": 0.3,
                "teacher_forced_mean_sequence_accuracy": 0.75,
                "generation_mean_prefix_match_length": 7.0,
                "teacher_forced_mean_prefix_match_length": 24.0,
            },
        }

        self.assertGreater(
            score_generalization_result(generation_better, score_mode="generation"),
            score_generalization_result(teacher_forced_better, score_mode="generation"),
        )
        self.assertGreater(
            score_generalization_result(teacher_forced_better, score_mode="teacher_forced"),
            score_generalization_result(generation_better, score_mode="teacher_forced"),
        )


if __name__ == "__main__":
    unittest.main()
