import unittest

from src.dsra.application.arithmetic_emergence_service import (
    ArithmeticStageMetric,
    ArithmeticCurriculumSnapshot,
    compute_forgetting_gap,
    is_catastrophic_forgetting,
    count_completed_curriculum_stages,
    count_ever_passed_curriculum_stages,
)


def _make_metric(stage_name: str, em: float) -> ArithmeticStageMetric:
    return ArithmeticStageMetric(stage_name=stage_name, exact_match=em)


def _make_snapshot(
    step: int,
    metrics: tuple[ArithmeticStageMetric, ...],
) -> ArithmeticCurriculumSnapshot:
    return ArithmeticCurriculumSnapshot(
        dataset_name="test",
        model_name="mhdsra2",
        seed=101,
        num_layers=1,
        step=step,
        active_stage_name="test",
        advanced_to_stage_name=None,
        stage_exact_matches=metrics,
    )


class TestForgettingGap(unittest.TestCase):
    """Unit tests for forgetting gap computation.

    中文说明:
    - 调用方 / Called by: ``python -m unittest``, ``pytest``.
    - 调用对象 / Calls: ``compute_forgetting_gap``, ``is_catastrophic_forgetting``.
    - 作用 / Purpose: 验证遗忘量 F 和灾难性遗忘判定的纯函数逻辑。
    - 变量 / Variables: 所有输入均为简单整数。
    - 接入 / Integration: 作为遗忘曲线报告的底层函数保护。
    - 错误处理 / Error handling: 非法输入通过断言暴露。
    - 关键词 / Keywords:
      forgetting_gap|catastrophic|forgetting|unit_test|curriculum|mhdsra2|retained|ever_passed|遗忘量|单元测试

    English documentation:
    Function name:
        TestForgettingGap
    Purpose:
        Unit tests for forgetting gap and catastrophic forgetting detection.
    Called by:
        ``python -m unittest``, ``pytest``.
    Calls:
        ``compute_forgetting_gap``, ``is_catastrophic_forgetting``.
    Parameters:
        None; each test is self-contained.
    Returns:
        None; assertions define success.
    Integration:
        Protects the underlying pure functions used by the forgetting curve report.
    Error handling:
        Illegal inputs exposed via assertions.
    English keywords:
        forgetting_gap, catastrophic, forgetting, unit_test, curriculum
    """

    def test_no_forgetting_when_all_retained(self) -> None:
        """F=0 when retained == ever_passed == 3."""
        gap = compute_forgetting_gap(retained=3, ever_passed=3)
        self.assertEqual(gap, 0)
        self.assertFalse(is_catastrophic_forgetting(retained=3))

    def test_curriculum_forgetting_mild(self) -> None:
        """F=1 when 1 stage is forgotten (retained=2, ever_passed=3)."""
        gap = compute_forgetting_gap(retained=2, ever_passed=3)
        self.assertEqual(gap, 1)
        self.assertFalse(is_catastrophic_forgetting(retained=2))

    def test_curriculum_forgetting_severe(self) -> None:
        """F=2 when 2 stages forgotten (retained=1, ever_passed=3)."""
        gap = compute_forgetting_gap(retained=1, ever_passed=3)
        self.assertEqual(gap, 2)
        self.assertFalse(is_catastrophic_forgetting(retained=1))

    def test_catastrophic_forgetting_detected(self) -> None:
        """Catastrophic when retained=0."""
        gap = compute_forgetting_gap(retained=0, ever_passed=3)
        self.assertEqual(gap, 3)
        self.assertTrue(is_catastrophic_forgetting(retained=0))

    def test_catastrophic_when_retained_zero_and_ever_passed_zero(self) -> None:
        """Catastrophic when both retained=0 and ever_passed=0."""
        gap = compute_forgetting_gap(retained=0, ever_passed=0)
        self.assertEqual(gap, 0)
        self.assertTrue(is_catastrophic_forgetting(retained=0))

    def test_invalid_input_raises(self) -> None:
        """retained > ever_passed raises ValueError."""
        with self.assertRaises(ValueError):
            compute_forgetting_gap(retained=3, ever_passed=2)

    def test_no_forgetting_partial_stages(self) -> None:
        """F=0 when only 2 stages exist and both retained."""
        gap = compute_forgetting_gap(retained=2, ever_passed=2)
        self.assertEqual(gap, 0)
        self.assertFalse(is_catastrophic_forgetting(retained=2))


class TestForgettingWithDataclass(unittest.TestCase):
    """Integration tests: forgetting metrics computed from real ArithmeticEmergenceRun fields.

    中文说明:
    - 调用方 / Called by: ``python -m unittest``, ``pytest``.
    - 调用对象 / Calls: ``compute_forgetting_gap``, ``is_catastrophic_forgetting``,
      ``count_completed_curriculum_stages``, ``count_ever_passed_curriculum_stages``.
    - 作用 / Purpose: 验证遗忘指标可以从 ArithmeticEmergenceRun 的字段正确导出。
    - 变量 / Variables: 使用构造的 dataclass 模拟不同遗忘场景。
    - 接入 / Integration: 保护遗忘曲线报告从 run 对象提取遗忘指标的正确性。
    - 关键词 / Keywords:
      forgetting|retained|ever_passed|dataclass|test|curriculum|mhdsra2|arithmetic_emergence|run|集成测试

    English documentation:
    Function name:
        TestForgettingWithDataclass
    Purpose:
        Verify forgetting metrics correctly derived from ArithmeticEmergenceRun fields.
    Called by:
        ``python -m unittest``, ``pytest``.
    Calls:
        ``compute_forgetting_gap``, ``is_catastrophic_forgetting``,
        ``count_completed_curriculum_stages``, ``count_ever_passed_curriculum_stages``.
    Parameters:
        None; each test constructs dataclass instances.
    Returns:
        None; assertions define success.
    English keywords:
        forgetting, retained, ever_passed, dataclass, test, curriculum, emergence, run
    """

    def test_run_with_all_retained(self) -> None:
        """ArithmeticEmergenceRun with all 3 stages retained -> F=0, not catastrophic."""
        final_metrics = (
            _make_metric("unit_no_carry", 1.0),
            _make_metric("unit_with_carry", 1.0),
            _make_metric("two_digit_rules", 1.0),
        )
        retained = count_completed_curriculum_stages(final_metrics)
        gap = compute_forgetting_gap(retained=retained, ever_passed=3)
        self.assertEqual(gap, 0)
        self.assertFalse(is_catastrophic_forgetting(retained=retained))

    def test_run_with_curriculum_forgetting(self) -> None:
        """2 retained, 3 ever_passed -> F=1, not catastrophic."""
        final_metrics = (
            _make_metric("unit_no_carry", 1.0),
            _make_metric("unit_with_carry", 1.0),
            _make_metric("two_digit_rules", 0.5),
        )
        retained = count_completed_curriculum_stages(final_metrics)
        gap = compute_forgetting_gap(retained=retained, ever_passed=3)
        self.assertEqual(gap, 1)
        self.assertFalse(is_catastrophic_forgetting(retained=retained))

    def test_run_with_catastrophic_forgetting(self) -> None:
        """0 retained, 3 ever_passed -> F=3, catastrophic."""
        final_metrics = (
            _make_metric("unit_no_carry", 0.0),
            _make_metric("unit_with_carry", 0.0),
            _make_metric("two_digit_rules", 0.0),
        )
        retained = count_completed_curriculum_stages(final_metrics)
        gap = compute_forgetting_gap(retained=retained, ever_passed=3)
        self.assertEqual(gap, 3)
        self.assertTrue(is_catastrophic_forgetting(retained=retained))

    def test_ever_passed_accounting(self) -> None:
        """Snapshots where a stage once passed but later failed should count as ever_passed."""
        snapshots = (
            _make_snapshot(
                step=10,
                metrics=(
                    _make_metric("unit_no_carry", 1.0),
                    _make_metric("unit_with_carry", 0.3),
                ),
            ),
            _make_snapshot(
                step=20,
                metrics=(
                    _make_metric("unit_no_carry", 0.4),
                    _make_metric("unit_with_carry", 1.0),
                ),
            ),
        )
        ever_passed = count_ever_passed_curriculum_stages(
            snapshots,
            ("unit_no_carry", "unit_with_carry"),
        )
        self.assertEqual(ever_passed, 2)


if __name__ == "__main__":
    unittest.main()
