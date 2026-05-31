from __future__ import annotations

import math

import torch

from scripts.tiny_llama_mhdsra2 import evaluate_ppl
from scripts.tiny_llama_shared import PAD_ID, split_train_validation_text


class FixedLogitModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fixed_logits[: x.shape[0], : x.shape[1], :]


def test_evaluate_ppl_ignores_padding_tokens() -> None:
    """Validate padding-safe PPL uses only non-PAD targets.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `scripts.tiny_llama_mhdsra2.evaluate_ppl`.
    - 作用 / Purpose: 防止 padding token 被当成有效试题拉低或拉高验证 PPL。
    - 变量 / Variables: `batch_y` 中 0 是 PAD，其他 token 是有效目标。
    - 接入 / Integration: 保护 tiny LLaMA baseline 与 MHDSRA2 共用评估口径。
    - 错误处理 / Error handling: 断言失败直接暴露评估口径回归。
    - 副作用 / Side effects: 无。
    """
    batch_x = torch.tensor([[1, 1, 1]])
    batch_y = torch.tensor([[1, PAD_ID, 2]])
    logits = torch.tensor(
        [
            [
                [0.0, 2.0, 0.0, 0.0],
                [9.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    model = FixedLogitModel(logits)
    loader = [(batch_x, batch_y)]

    ppl = evaluate_ppl(model, loader, torch.device("cpu"))

    log_probs = torch.log_softmax(logits, dim=-1)
    expected_loss = -(log_probs[0, 0, 1] + log_probs[0, 2, 2]) / 2
    assert math.isclose(ppl, math.exp(float(expected_loss)), rel_tol=1e-6)


def test_evaluate_ppl_rejects_all_padding_loader() -> None:
    batch_x = torch.tensor([[1, 1]])
    batch_y = torch.tensor([[PAD_ID, PAD_ID]])
    logits = torch.zeros(1, 2, 4)
    model = FixedLogitModel(logits)

    try:
        evaluate_ppl(model, [(batch_x, batch_y)], torch.device("cpu"))
    except ValueError as exc:
        assert "no non-PAD tokens" in str(exc)
    else:
        raise AssertionError("evaluate_ppl should reject an all-PAD evaluation loader")


def test_split_train_validation_text_uses_tail_as_validation() -> None:
    text = "abcdefghij"

    train_text, valid_text = split_train_validation_text(text, validation_chars=3)

    assert train_text == "abcdefgh"
    assert valid_text == "ij"
