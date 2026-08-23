from __future__ import annotations

import math

import pytest
import torch

from scripts.tiny_llama_mhdsra2 import evaluate_ppl
from scripts.tiny_lm_practical_multiseed import (
    build_payload,
    build_parser as build_multiseed_parser,
    parse_compare_output,
)
from scripts.tiny_llama_shared import (
    PAD_ID,
    resolve_device,
    set_reproducible_seed,
    split_train_validation_text,
)
from scripts.tiny_llama_compare import build_parser


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


def test_tiny_llama_resolve_device_uses_cuda_zero_alias() -> None:
    """Validate tiny LLaMA device parsing follows the project cuda:0 policy.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `scripts.tiny_llama_shared.resolve_device`.
    - 作用 / Purpose: 防止 `auto/cuda` 路径返回裸 `cuda` 或接受非 0 号 GPU。
    - 错误处理 / Error handling: `cuda:1` 必须抛出 `ValueError`。
    - 副作用 / Side effects: 只构造 `torch.device`，不分配 CUDA 张量。
    """
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("cuda") == torch.device("cuda:0")
    with pytest.raises(ValueError):
        resolve_device("cuda:1")


def test_tiny_llama_compare_parser_exposes_mhdsra2_chunk_size() -> None:
    """Validate the tiny LM comparison can reproduce tuned MHDSRA2 chunk sizes.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `scripts.tiny_llama_compare.build_parser`.
    - 作用 / Purpose: 将 MHDSRA2 语言模型的流式 chunk 尺寸暴露为正式 CLI 参数，
      避免只能靠临时 Python 片段复现实战 PPL 对照。
    - 错误处理 / Error handling: argparse 解析异常会让测试失败。
    - 副作用 / Side effects: 无。
    """
    args = build_parser().parse_args(["--mhdsra2-chunk-size", "1024"])

    assert args.mhdsra2_chunk_size == 1024


def test_tiny_llama_set_reproducible_seed_resets_torch_rng() -> None:
    """Validate tiny LM seed helper makes Torch draws reproducible.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `scripts.tiny_llama_shared.set_reproducible_seed`.
    - 作用 / Purpose: 保护 tiny LM 实战 PPL 对照的随机种子入口，确保同一 seed
      下模型初始化和数据顺序可复现。
    - 错误处理 / Error handling: 随机数不一致时断言失败。
    - 副作用 / Side effects: 修改测试进程的随机数状态。
    """
    set_reproducible_seed(777)
    first = torch.rand(4)
    set_reproducible_seed(777)
    second = torch.rand(4)

    assert torch.equal(first, second)


def test_tiny_lm_practical_multiseed_parses_compare_output() -> None:
    """Validate the resumable practical runner parses public CLI output.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `scripts.tiny_lm_practical_multiseed.parse_compare_output`.
    - 作用 / Purpose: 确保多 seed runner 不依赖人工读取终端文本，可以稳定提取 PPL 与耗时。
    - 错误处理 / Error handling: 输出格式变化会触发解析失败。
    - 副作用 / Side effects: 无。
    """
    stdout = """
  Standard Attention Validation PPL: 12.67
  MHDSRA2 Validation PPL:            12.61
  Ratio:                  0.996x
  Training Time Std:      21s
  Training Time MHDSRA2:  52s
"""

    row = parse_compare_output(1234, stdout, ["python", "scripts/tiny_llama_compare.py"])

    assert row["status"] == "completed"
    assert row["standard_validation_ppl"] == 12.67
    assert row["mhdsra2_validation_ppl"] == 12.61
    assert row["time_ratio_mhdsra2_over_standard"] == 52 / 21


def test_tiny_lm_practical_multiseed_payload_summarizes_completed_rows() -> None:
    """Validate multi-seed payload summarizes only completed seed rows.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `scripts.tiny_lm_practical_multiseed.build_payload`.
    - 作用 / Purpose: 防止失败 seed 污染均值，同时保留失败计数供报告解释。
    - 错误处理 / Error handling: 汇总字段不符合预期时断言失败。
    - 副作用 / Side effects: 无。
    """
    args = build_multiseed_parser().parse_args(
        ["--seeds", "1,2", "--report-name", "unit_practical"]
    )
    rows = [
        {
            "seed": 1,
            "status": "completed",
            "standard_validation_ppl": 10.0,
            "mhdsra2_validation_ppl": 11.0,
            "ppl_ratio_mhdsra2_over_standard": 1.1,
            "standard_time_s": 5.0,
            "mhdsra2_time_s": 10.0,
            "time_ratio_mhdsra2_over_standard": 2.0,
        },
        {"seed": 2, "status": "failed", "returncode": 1},
    ]

    payload = build_payload(args, rows)

    assert payload["summary"]["completed_count"] == 1
    assert payload["summary"]["failed_count"] == 1
    assert payload["summary"]["mhdsra2_validation_ppl"]["mean"] == 11.0


def test_tiny_lm_practical_multiseed_payload_counts_timeout_rows() -> None:
    """Validate timeout rows are preserved but excluded from metric means.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `scripts.tiny_lm_practical_multiseed.build_payload`.
    - 作用 / Purpose: 确认某个 seed 超时后不会污染 PPL 均值，也不会从报告中消失。
    - 错误处理 / Error handling: timeout 计数不正确时断言失败。
    - 副作用 / Side effects: 无。
    """
    args = build_multiseed_parser().parse_args(["--seeds", "1"])

    payload = build_payload(args, [{"seed": 1, "status": "timeout", "timeout_sec": 1}])

    assert payload["summary"]["completed_count"] == 0
    assert payload["summary"]["failed_count"] == 1
    assert payload["summary"]["standard_validation_ppl"]["mean"] is None
