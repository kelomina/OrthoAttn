import unittest

import torch

from json_retrieval_test import (
    ANSWER_START_TOKEN_ID,
    PAD_TOKEN_ID,
    QUESTION_TOKEN_ID,
    VOCAB_SIZE,
    evaluate_teacher_forced,
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


if __name__ == "__main__":
    unittest.main()
