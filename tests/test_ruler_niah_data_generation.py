# -*- coding: utf-8 -*-
"""RULER-NIAH 数据生成器单元与边界测试套件.

中文说明:
- 被测对象 / Targets: `src.dsra.domain.ruler_niah` 的词表、分词器、批量生成与评测函数
- 调用方 / Called by: `pytest tests/test_ruler_niah_data_generation.py`
- 覆盖 / Coverage:
  1. 词表封闭性与 PAD=0 保留;
  2. 配置校验(非法变体/数量关系/批大小);
  3. 批量生成形状、确定性(同种子逐位复现)、设备参数;
  4. 规范对齐: 噪声句/针句文本与 NVIDIA/RULER 官方模板逐字一致(小写后);
  5. 因果与泄漏防御: 答案数字串在上下文中恰好出现一次(被查询针句内),
     非答案数字位监督为 0, 标签按 next-token 挂载;
  6. 多针(MQ 场景)下 hit/answer 元数据一致性;
  7. exact-match 评测函数的命中/未命中行为。
"""

import pytest
import torch

from src.dsra.domain.ruler_niah import (
    ADJECTIVES,
    ID2TOKEN,
    NOUNS,
    NOISE_SENTENCE,
    RulerNiahConfig,
    VOCAB,
    _digits_to_ids,
    _tokenize,
    evaluate_ruler_niah_exact_match,
    generate_ruler_niah_batch,
)
from typing import List


def test_vocab_closed_and_pad_reserved() -> None:
    """词表必须封闭覆盖所有素材词; PAD=0 不分配给任何真实 token."""
    assert VOCAB["<pad>"] == 0
    for w in ("the", "grass", "special", "magic", "numbers", "for", "is", ":"):
        assert w in VOCAB, w
    for a in ADJECTIVES[:8]:
        assert a in VOCAB
    for n in NOUNS[:8]:
        assert n in VOCAB
    for d in "0123456789":
        assert d in VOCAB


def test_config_validation_errors() -> None:
    """非法配置必须精确抛出 ValueError."""
    with pytest.raises(ValueError):
        RulerNiahConfig(variant="mq")
    with pytest.raises(ValueError):
        RulerNiahConfig(num_haystack=0)
    with pytest.raises(ValueError):
        RulerNiahConfig(num_needle_k=0)
    with pytest.raises(ValueError):
        RulerNiahConfig(num_needle_k=2, num_needle_q=3)
    with pytest.raises(ValueError):
        RulerNiahConfig(batch_size=0)


def test_batch_shapes_determinism_and_device() -> None:
    """同种子逐位可复现; 形状一致; 设备参数生效."""
    cfg = RulerNiahConfig(
        num_haystack=64, num_needle_k=2, num_needle_q=1,
        batch_size=4, seed=1234,
    )
    X1, Y1, m1 = generate_ruler_niah_batch(cfg)
    X2, Y2, m2 = generate_ruler_niah_batch(cfg)
    assert torch.equal(X1, X2)
    assert torch.equal(Y1, Y2)
    assert X1.shape[0] == Y1.shape[0] == 4
    assert X1.dtype == Y1.dtype == torch.long
    assert [x["answers"] for x in m1] == [x["answers"] for x in m2]
    Xc, _, _ = generate_ruler_niah_batch(
        RulerNiahConfig(num_haystack=16, batch_size=2, seed=7, device="cpu")
    )
    assert Xc.device.type == "cpu"


def test_spec_alignment_verbatim_sentences() -> None:
    """噪声句与针句须与 NVIDIA/RULER 官方模板逐字一致(经同一分词管线比对)."""
    cfg = RulerNiahConfig(num_haystack=32, num_needle_k=1, num_needle_q=1, seed=5)
    X, Y, metas = generate_ruler_niah_batch(cfg)
    toks = [ID2TOKEN[i] for i in X[0].tolist()]
    text = " ".join(t for t in toks if t != "<pad>")
    # 官方噪声句(小写)经同一分词管线后应完整出现
    noise_lower = " ".join(_tokenize(NOISE_SENTENCE))
    assert noise_lower in text
    # 官方针句模板(小写)经同一分词管线后应完整出现
    ans = metas[0]["answers"][0]
    key_txt = str(metas[0]["query_keys"])
    needle_tokens = _tokenize(
        f"One of the special magic numbers for {key_txt} is: {''.join(ans)}."
    )
    needle_lower = " ".join(needle_tokens)
    assert needle_lower in text


def test_answer_supervision_positions_and_no_leak() -> None:
    """答案数字串在上下文恰好出现一次; 监督位置严格落在答案前缀尾部之后."""
    cfg = RulerNiahConfig(num_haystack=48, num_needle_k=1, num_needle_q=1, seed=11)
    X, Y, metas = generate_ruler_niah_batch(cfg)
    meta = metas[0]
    answer = meta["answers"][0]
    digit_ids = _digits_to_ids(answer)

    # 上下文部分(前 context_len 个 token)中答案数字串恰好出现一次
    ctx = X[0, : int(meta["context_len"])].tolist()
    occurrences = sum(
        1
        for s in range(len(ctx) - len(digit_ids) + 1)
        if ctx[s : s + len(digit_ids)] == digit_ids
    )
    assert occurrences == 1

    # 监督位置: 值依次等于答案数字位+句号, 且末位标签位于倒数第二位(预测句号后无监督)
    nonzero = (Y[0] != 0).nonzero().flatten().tolist()
    expect_labels = digit_ids + [VOCAB["."]]
    assert Y[0, nonzero].tolist() == expect_labels
    assert nonzero[-1] == X.shape[1] - 2
    assert all(p >= int(meta["context_len"]) - 1 for p in nonzero)


def test_multi_needle_metadata_consistency() -> None:
    """多针(K>1,Q>1)时元数据与监督内容保持一致."""
    cfg = RulerNiahConfig(
        num_haystack=64, num_needle_k=3, num_needle_q=2, batch_size=2, seed=21
    )
    X, Y, metas = generate_ruler_niah_batch(cfg)
    for b in range(2):
        answers = metas[b]["answers"]
        assert len(answers) == 2
        digit_ids: List[int] = []
        for a in answers:
            digit_ids.extend(_digits_to_ids(a))
        nonzero = (Y[b] != 0).nonzero().flatten().tolist()
        labels = Y[b, nonzero].tolist()
        assert labels[:-1] == digit_ids and labels[-1] == VOCAB["."]


def test_exact_match_evaluator() -> None:
    """精确匹配评测: 全对计 1, 缺位/错位计 0."""
    assert evaluate_ruler_niah_exact_match([[1, 2, 3, 4, 5, 6, 7]], [["1234567"]]) == 1.0
    assert evaluate_ruler_niah_exact_match([[1, 2, 3, 4, 5, 6, 8]], [["1234567"]]) == 0.0
    assert evaluate_ruler_niah_exact_match([[1, 2, 3]], [["1234567"]]) == 0.0
