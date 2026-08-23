"""verify_technical_report 独立指标的 Oracle 已知答案回归测试.

中文说明:
- 调用方 / Called by: pytest (Phase 0 门4; oracle 不通过禁止进入重跑阶段)
- 被测对象: scripts.verify_technical_report 的四个独立指标函数与表3闭合解
- 作用: 用手工构造的已知答案用例锁定指标口径, 防止复核工具自身成为幻觉源
"""

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_technical_report import (  # noqa: E402
    REPORTED_TABLE3_WEIGHTS,
    CASE_STUDIES_TABLE3_EXACT,
    _closed_form_exact_weight,
    independent_niah_accuracy,
    independent_ppl,
    judge_rel,
)


def test_accuracy_oracle_5_of_8():
    """8 样本 5 对 3 错 -> 0.625."""
    acc, n = independent_niah_accuracy(
        [10, 20, 30, 40, 50, 60, 70, 80],
        [10, 20, 99, 40, 99, 60, 99, 80],
    )
    assert n == 8
    assert abs(acc - 0.625) < 1e-12


def test_accuracy_masked_samples_excluded_from_denominator():
    """padding/无效样本不计入分母 (3 有效全对 -> 1.0)."""
    acc, n = independent_niah_accuracy(
        [7, 8, 9, 111],
        [7, 8, 9, 222],
        valid_mask=[True, True, True, False],
    )
    assert n == 3
    assert acc == 1.0


def test_accuracy_empty_returns_none():
    acc, n = independent_niah_accuracy([], [])
    assert acc is None and n == 0


def test_accuracy_shape_mismatch_raises():
    try:
        independent_niah_accuracy([1, 2], [1])
    except ValueError:
        return
    raise AssertionError("expected ValueError on length mismatch")


def test_ppl_oracle_known_nll():
    """NLL=[ln2,ln2,ln4] -> exp(ln(2*2*4)/3) = 16^(1/3) = 2.5198."""
    expected = math.exp((math.log(2) + math.log(2) + math.log(4)) / 3)
    got = independent_ppl(math.log(2) + math.log(2) + math.log(4), 3)
    assert abs(got - expected) < 1e-12
    assert abs(got - 16 ** (1 / 3)) < 1e-9  # 手工数值锚点: 2.5198


def test_ppl_zero_tokens_raises():
    try:
        independent_ppl(1.0, 0)
    except ValueError:
        return
    raise AssertionError("expected ValueError on zero tokens")


def test_peak_memory_probe_measures_known_allocation():
    """分配 64MB 张量后峰值必须 >= 64MB (CUDA 不可用则跳过)."""
    import torch

    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA not available; memory probe oracle requires GPU")
    from scripts.verify_technical_report import independent_peak_memory_probe

    def alloc():
        torch.zeros(64 * 1024 * 1024 // 4, device="cuda:0")  # 64MiB float32

    peak_mb = independent_peak_memory_probe(alloc)
    assert peak_mb is not None
    assert peak_mb >= 64.0 - 0.5, f"peak {peak_mb}MB below 64MB allocation"


def test_timing_positive_and_repeatable():
    from scripts.verify_technical_report import independent_timing

    t1 = independent_timing(lambda: None, warmup=1, repeats=3)
    t2 = independent_timing(lambda: None, warmup=1, repeats=3)
    assert t1 >= 0.0 and t2 >= 0.0
    assert abs(t1 - t2) < 1000.0  # 空循环两次测量应在同一量级


def test_closed_form_matches_reported_table3():
    """表3 闭合解必须与报告/case studies 数值一致 (相对差 < 1%)."""
    params = {"none": 128, "top32": 32, "top16": 16, "top8": 8}
    for label, n in params.items():
        closed = _closed_form_exact_weight(8.0, 8, n)
        assert abs(closed - CASE_STUDIES_TABLE3_EXACT[label]) / CASE_STUDIES_TABLE3_EXACT[label] < 0.01
        assert abs(closed - REPORTED_TABLE3_WEIGHTS[label]) / REPORTED_TABLE3_WEIGHTS[label] < 0.01


def test_judge_rel_thresholds():
    assert judge_rel(1.0, 1.0, 0.01)[0] == "confirmed"
    assert judge_rel(0.995, 1.0, 0.01)[0] == "confirmed"
    assert judge_rel(0.95, 1.0, 0.01)[0] == "deviation"
    assert judge_rel(0.5, 1.0, 0.01)[0] == "refuted"
    assert judge_rel(None, 1.0, 0.01)[0] == "no_source"
