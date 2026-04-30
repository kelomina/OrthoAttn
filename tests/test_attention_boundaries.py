import unittest
from pathlib import Path

import torch

from src.dsra.application import JsonRetrievalSearchService, StreamingAttentionUnitOfWork
from src.dsra.domain import AttentionLayerSpec, RetrievalModelSpec, normalize_model_type
from src.dsra.infrastructure import JsonRetrievalReportRepository, PagedMemoryRepository
from src.dsra.mhdsra2.improved_dsra_mha import MHDSRA2Config, MultiHeadDSRA2


class TestAttentionBoundaries(unittest.TestCase):
    """Validate the first DDD and unit-of-work boundaries.

    中文说明:
    - 调用方 / Called by: `python -m unittest tests.test_attention_boundaries`
    - 调用对象 / Calls:
      `AttentionLayerSpec`, `PagedMemoryRepository`, `StreamingAttentionUnitOfWork`
    - 作用 / Purpose: 保护领域层、基础设施层、应用层工作单元的最小可用边界
    - 变量 / Variables: 测试方法内分别构造规格、分页记忆仓储和工作单元
    - 接入 / Integration: 放在 `tests/` 下，由 `pytest` 与 `unittest` 自动发现
    - 错误处理 / Error handling: 通过断言和 `assertRaises` 暴露边界回归
    - 关键词 / Keywords:
      ddd|domain|application|infrastructure|unit_of_work|memory|spec|tests|boundary|边界
    """

    def test_attention_layer_spec_rejects_invalid_values(self):
        """Validate domain specification errors are explicit.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `AttentionLayerSpec`
        - 作用 / Purpose: 确认领域层在非法配置进入模型层前抛出错误
        - 变量 / Variables: `valid_spec` 是合法规格，其余构造触发异常
        - 接入 / Integration: 新增领域配置字段时扩展本测试
        - 错误处理 / Error handling: 使用 `assertRaises(ValueError)` 校验错误路径
        - 关键词 / Keywords:
          domain|spec|validation|ValueError|pe_mode|slots|topk|config|test|校验
        """
        valid_spec = AttentionLayerSpec(
            dim=64,
            slots=16,
            read_topk=4,
            write_topk=4,
            local_window=32,
            pe_mode="none",
        )

        self.assertEqual(valid_spec.dim, 64)
        with self.assertRaises(ValueError):
            AttentionLayerSpec(0, 16, 4, 4, 32)
        with self.assertRaises(ValueError):
            AttentionLayerSpec(64, 16, 4, 4, 32, pe_mode="unsupported")

    def test_paged_memory_repository_retrieves_batch_one_candidates(self):
        """Validate infrastructure repository wraps paged exact memory.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `PagedMemoryRepository.append`, `PagedMemoryRepository.retrieve`
        - 作用 / Purpose: 确认基础设施层可以写入并召回 batch=1 的 head-space K/V
        - 变量 / Variables: `key/value/query` 是分页记忆使用的 K/V 与查询张量
        - 接入 / Integration: DSRA 兼容层可依赖该仓储执行 paged recall
        - 错误处理 / Error handling: 空召回或形状错误会触发断言失败
        - 关键词 / Keywords:
          infrastructure|repository|paged_memory|append|retrieve|batch_one|kv|mhdsra2|test|召回
        """
        repository = PagedMemoryRepository(
            enabled=True,
            page_size=4,
            dtype=torch.float32,
            top_pages=1,
            max_tokens=2,
        )
        key = torch.randn(1, 2, 6, 8)
        value = torch.randn(1, 2, 6, 8)
        query = key[:, :, 2:3, :]

        repository.append(key, value)
        retrieved_k, retrieved_v = repository.retrieve(query, torch.device("cpu"))

        self.assertIsNotNone(retrieved_k)
        self.assertIsNotNone(retrieved_v)
        assert retrieved_k is not None
        assert retrieved_v is not None
        self.assertEqual(retrieved_k.shape[:2], (1, 2))
        self.assertEqual(retrieved_v.shape[:2], (1, 2))

    def test_streaming_attention_unit_of_work_commits_forward_state(self):
        """Validate application unit-of-work state commit semantics.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls:
          `MultiHeadDSRA2.init_state`, `StreamingAttentionUnitOfWork.commit_forward`
        - 作用 / Purpose: 确认应用层工作单元成功前向后才更新状态和缓存字段
        - 变量 / Variables: `state` 是初始核心状态, `cache` 是兼容局部缓存
        - 接入 / Integration: DSRA 兼容层通过本工作单元管理一次前向边界
        - 错误处理 / Error handling: 字段未提交或提交错误会触发断言失败
        - 关键词 / Keywords:
          application|unit_of_work|commit|state|cache|time_state|mhdsra2|boundary|test|提交
        """
        layer = MultiHeadDSRA2(MHDSRA2Config(dim=16, heads=2, slots=4, use_retrieval=False))
        state = layer.init_state(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
        cache = (torch.randn(1, 2, 16), torch.randn(1, 2, 16))
        time_state = torch.zeros(1, 4)
        repository = PagedMemoryRepository(enabled=False)

        with StreamingAttentionUnitOfWork(
            state=None,
            kv_cache=None,
            time_state=None,
            memory_repository=repository,
        ) as unit_of_work:
            unit_of_work.commit_forward(state=state, kv_cache=cache, time_state=time_state)

        self.assertIs(unit_of_work.state, state)
        self.assertIs(unit_of_work.kv_cache, cache)
        self.assertIs(unit_of_work.time_state, time_state)

    def test_retrieval_model_spec_archives_dsra_alias_to_mhdsra2(self):
        """Validate archived `dsra` model names normalize to MHDSRA2.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `RetrievalModelSpec`, `normalize_model_type`
        - 作用 / Purpose: 确认领域层把历史 `dsra` 名称归档为当前 `mhdsra2` 架构
        - 变量 / Variables: `spec` 是 JSON retrieval 模型构建规格
        - 接入 / Integration: 保护 `scripts.json_retrieval_test.build_retrieval_model`
        - 错误处理 / Error handling: 归一化错误或非法规格会触发断言失败
        - 关键词 / Keywords:
          retrieval|model_spec|archive|dsra|mhdsra2|alias|domain|normalize|test|归档
        """
        spec = RetrievalModelSpec(
            requested_model_type="dsra",
            vocab_size=259,
            dim=16,
            slots=8,
            topk=2,
            chunk_size=4,
            local_context_size=2,
            local_context_mode="concat",
        )

        self.assertEqual(normalize_model_type("dsra"), "mhdsra2")
        self.assertEqual(spec.model_type, "mhdsra2")

    def test_json_retrieval_search_service_orders_results(self):
        """Validate application search service selects and sorts summaries.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls:
          `JsonRetrievalSearchService.choose_best`, `sort_single_case_summaries`
        - 作用 / Purpose: 保护 JSON retrieval 用例运行中的最佳结果选择与报告排序边界
        - 变量 / Variables: `candidate_a/candidate_b/summaries` 是模拟搜索结果
        - 接入 / Integration: 脚本层训练循环通过本服务做应用层决策
        - 错误处理 / Error handling: 排序或选择规则回归会触发断言失败
        - 关键词 / Keywords:
          search_service|choose_best|sort|json_retrieval|application|summary|score|test|use_case|排序
        """
        service = JsonRetrievalSearchService()
        candidate_a = {"score": (0, 2)}
        candidate_b = {"score": (1, 1)}
        summaries = [
            {
                "generation_exact_byte_match": False,
                "teacher_forced_exact_byte_match": True,
                "generation_prefix_match_length": 2,
                "generation_sequence_accuracy": 0.5,
                "teacher_forced_prefix_match_length": 3,
                "teacher_forced_sequence_accuracy": 0.75,
            },
            {
                "generation_exact_byte_match": True,
                "teacher_forced_exact_byte_match": False,
                "generation_prefix_match_length": 1,
                "generation_sequence_accuracy": 0.25,
                "teacher_forced_prefix_match_length": 1,
                "teacher_forced_sequence_accuracy": 0.25,
            },
        ]

        best = service.choose_best(candidate_a, candidate_b, lambda item: item["score"])
        ordered = service.sort_single_case_summaries(summaries)

        self.assertIs(best, candidate_b)
        self.assertTrue(ordered[0]["generation_exact_byte_match"])

    def test_json_retrieval_report_repository_writes_artifacts(self):
        """Validate report output is handled by the infrastructure repository.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `JsonRetrievalReportRepository.write_report`
        - 作用 / Purpose: 确认 JSON retrieval 报告 JSON/Markdown 双产物由基础设施层写入
        - 变量 / Variables: `repository/json_path/markdown_path` 是报告仓储与产物路径
        - 接入 / Integration: `save_json_retrieval_reports` 复用该仓储输出报告
        - 错误处理 / Error handling: 文件缺失或内容缺失会触发断言失败
        - 关键词 / Keywords:
          report_repository|json|markdown|artifact|infrastructure|write|reports|test|persist|报告
        """
        project_root = Path(__file__).resolve().parents[1]
        reports_dir = project_root / "reports" / "test_json_retrieval_repository" / "reports"
        repository = JsonRetrievalReportRepository(reports_dir)
        json_path, markdown_path = repository.write_report(
            json_filename="sample.json",
            markdown_filename="sample.md",
            payload={"status": "ok"},
            markdown_lines=["# Sample", "", "ok"],
        )

        self.assertTrue(json_path.exists())
        self.assertTrue(markdown_path.exists())
        self.assertIn("Sample", markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
