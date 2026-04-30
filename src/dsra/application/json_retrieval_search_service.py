"""Application service for JSON retrieval search orchestration decisions."""

from __future__ import annotations

from collections.abc import Callable

SearchScore = Callable[[dict], tuple]
GeneralizationScore = Callable[[dict], float]


class JsonRetrievalSearchService:
    """Select and order JSON retrieval experiment results.

    中文说明:
    - 调用方 / Called by: `scripts.json_retrieval_test.run_json_retrieval_test`,
      `scripts.json_retrieval_test.run_json_retrieval_generalization_test`
    - 调用对象 / Calls: scorer callback functions, `list.sort`
    - 作用 / Purpose: 把搜索结果选择与排序决策从脚本循环中抽到应用层
    - 变量 / Variables:
      `candidate/current_best` 是训练结果, `scorer` 是脚本提供的评分函数,
      `summaries` 是报告用搜索摘要列表
    - 接入 / Integration: 脚本保留训练细节，调用本服务选择最佳结果
    - 错误处理 / Error handling: 空最佳值会接受第一个候选；评分函数异常向上抛出
    - 关键词 / Keywords:
      search|service|json_retrieval|application|best_result|sort|score|use_case|mhdsra2|搜索
    """

    def choose_best(self, current_best: dict | None, candidate: dict, scorer: SearchScore) -> dict:
        """Return the better single-case search result.

        中文说明:
        - 调用方 / Called by: `run_json_retrieval_test`
        - 调用对象 / Calls: `scorer`
        - 作用 / Purpose: 统一单用例搜索最佳结果选择规则
        - 变量 / Variables:
          `current_best` 是当前最佳结果, `candidate` 是新候选, `scorer` 返回可比较分数
        - 接入 / Integration: 每个 search trial 结束后调用
        - 错误处理 / Error handling: `current_best=None` 时直接返回候选
        - 关键词 / Keywords:
          choose_best|single_case|search|json|score|candidate|application|retrieval|result|选择
        """
        if current_best is None:
            return candidate
        if scorer(candidate) > scorer(current_best):
            return candidate
        return current_best

    def choose_best_generalization(
        self,
        current_best: dict | None,
        candidate: dict,
        scorer: GeneralizationScore,
    ) -> dict:
        """Return the better generalization search result.

        中文说明:
        - 调用方 / Called by: `run_json_retrieval_generalization_test`
        - 调用对象 / Calls: `scorer`
        - 作用 / Purpose: 统一泛化搜索最佳结果选择规则
        - 变量 / Variables:
          `current_best` 是当前最佳泛化结果, `candidate` 是新候选, `scorer` 返回浮点分数
        - 接入 / Integration: 每个泛化 search trial 完成验证/测试评估后调用
        - 错误处理 / Error handling: `current_best=None` 时直接返回候选
        - 关键词 / Keywords:
          choose_best|generalization|search|json|score|candidate|application|retrieval|result|泛化
        """
        if current_best is None:
            return candidate
        if scorer(candidate) > scorer(current_best):
            return candidate
        return current_best

    def sort_single_case_summaries(self, summaries: list[dict]) -> list[dict]:
        """Sort single-case search summaries in report order.

        中文说明:
        - 调用方 / Called by: `run_json_retrieval_test`
        - 调用对象 / Calls: `list.sort`
        - 作用 / Purpose: 集中维护单用例报告搜索摘要排序规则
        - 变量 / Variables: `summaries` 是 `summarize_search_result` 返回的字典列表
        - 接入 / Integration: 生成报告前调用并使用返回列表
        - 错误处理 / Error handling: 缺失必要键会由字典访问抛出 `KeyError`
        - 关键词 / Keywords:
          sort|single_case|summaries|report|json_retrieval|application|generation|teacher_forced|search|排序
        """
        summaries.sort(
            key=lambda item: (
                int(item["generation_exact_byte_match"]),
                int(item["teacher_forced_exact_byte_match"]),
                item["generation_prefix_match_length"],
                item["generation_sequence_accuracy"],
                item["teacher_forced_prefix_match_length"],
                item["teacher_forced_sequence_accuracy"],
            ),
            reverse=True,
        )
        return summaries

    def sort_generalization_summaries(
        self,
        summaries: list[dict],
        scorer: GeneralizationScore,
    ) -> list[dict]:
        """Sort generalization search summaries in report order.

        中文说明:
        - 调用方 / Called by: `run_json_retrieval_generalization_test`
        - 调用对象 / Calls: `list.sort`, `scorer`
        - 作用 / Purpose: 集中维护泛化报告搜索摘要排序规则
        - 变量 / Variables: `summaries` 是泛化搜索摘要列表, `scorer` 是摘要评分函数
        - 接入 / Integration: 生成泛化报告前调用并使用返回列表
        - 错误处理 / Error handling: 评分函数异常向上抛出
        - 关键词 / Keywords:
          sort|generalization|summaries|report|json_retrieval|application|score|search|mhdsra2|排序
        """
        summaries.sort(key=scorer, reverse=True)
        return summaries
