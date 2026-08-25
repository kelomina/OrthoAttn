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
from typing import List

from src.dsra.domain.ruler_niah import (
    ADJECTIVES,
    ID2TOKEN,
    NOUNS,
    NOISE_SENTENCE,
    RulerNiahConfig,
    VOCAB,
    _digits_to_ids,
    _tokenize,
    generate_ruler_niah_batch,
)


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


def test_spec_alignment_hardcoded_literal() -> None:
    """硬编码字面量断言: 打破"用被测分词器验证被测分词器"的循环性.

    中文说明:
    - 直接以字面量 token 序列断言一个已知样本的上下文包含完整官方针句
      (seed=3 时 key=quiet-opal, value=4781964), 不经过 _tokenize/VOCAB 构造
      期望值。若未来分词器或词表发生破坏性变更, 此用例将率先失败。
    """
    cfg = RulerNiahConfig(num_haystack=64, num_needle_k=1, num_needle_q=1, seed=3)
    X, _, _ = generate_ruler_niah_batch(cfg)
    toks = [ID2TOKEN[i] for i in X[0].tolist() if ID2TOKEN[i] != "<pad>"]
    joined = " ".join(toks)
    literal = (
        "one of the special magic numbers for quiet - opal is: "
        "4 7 8 1 9 6 4 ."
    )
    assert literal in joined


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


def test_per_step_seed_varies_batches() -> None:
    """逐步重播种必须产生不同批次(防止冻结在单批上死记硬背).

    中文说明:
    - 回归背景: scripts/benchmark_ruler_niah.py 曾因每步传入同一固定 seed 导致
      训练冻结在单批 8 样本上——loss 归零但泛化为 0(第一次幻觉复查未覆盖训练循环)。
      本测试锁定"不同 seed ⇒ 不同批次"这一跨文件契约。
    """
    import dataclasses

    base = RulerNiahConfig(num_haystack=64, num_needle_k=1, num_needle_q=1, batch_size=4, seed=100)
    X1, _, _ = generate_ruler_niah_batch(base)
    # 同种子复现
    X1b, _, _ = generate_ruler_niah_batch(base)
    assert torch.equal(X1, X1b)
    # 换种子必不同批次(不同针句/位置)
    X2, _, _ = generate_ruler_niah_batch(dataclasses.replace(base, seed=101))
    assert not torch.equal(X1, X2)


def test_score_prediction_excludes_period_and_reports_first_digit() -> None:
    """打分函数必须排除句号监督位; 首位数字正确率独立上报.

    中文说明:
    - 回归背景: 历史版本把句号预测计入长度比对导致 EM 恒为 0(训练成功也被
      判零)。本测试锁定该 off-by-one 不再复发。
    """
    from scripts.benchmark_ruler_niah import score_prediction

    ans = ["1234567"]
    digit_ids = [_digits_to_ids(ans[0])]
    period = VOCAB["."]
    # 数字全对 + 句号对 → EM=1, 首位=1
    em, first = score_prediction(digit_ids[0] + [period], ans)
    assert em is True and first is True
    # 数字全对但句号预测错误 → EM 仍应为 1(句号不参与 EM)
    em, first = score_prediction(digit_ids[0] + [VOCAB["0"]], ans)
    assert em is True and first is True
    # 仅首位数字错 → EM=0 且 first=False
    wrong_first = digit_ids[0].copy()
    wrong_first[0] = VOCAB["9"] if ans[0][0] != "9" else VOCAB["8"]
    em, first = score_prediction(wrong_first + [period], ans)
    assert em is False and first is False
    # 首位对、中间错 → EM=0 但 first=True
    mid_wrong = digit_ids[0].copy()
    mid_wrong[3] = VOCAB["0"] if ans[0][3] != "0" else VOCAB["1"]
    em, first = score_prediction(mid_wrong + [period], ans)
    assert em is False and first is True
