import unittest

import torch

from src.dsra.mhdsra2.improved_dsra_mha import MHDSRA2Config, MultiHeadDSRA2
from src.dsra.mhdsra2.paged_exact_memory import PagedExactMemory


class TestMHDSRA2Core(unittest.TestCase):
    """MHDSRA2 core behavior tests.

    中文说明:
    - 调用方 / Called by: `python -m unittest tests.test_mhdsra2_core`、`python main.py unit`
    - 调用对象 / Calls: `MHDSRA2Config`, `MultiHeadDSRA2.forward`, `PagedExactMemory.append`,
      `PagedExactMemory.retrieve`
    - 作用 / Purpose: 验证 MHDSRA2 的张量形状、状态更新和分页检索行为
    - 变量 / Variables:
      `cfg` 配置, `layer` 注意力层, `state` 流式状态, `rk/rv` 检索键值, `mem` 分页记忆
    - 接入 / Integration: 放置在 `tests/` 下，可由 `unittest discover` 自动发现
    - 错误处理 / Error handling: 通过 `assert*` 断言在回归时快速失败
    - 关键词 / Keywords:
      MHDSRA2|MultiHeadDSRA2|PagedExactMemory|slot|local|retrieval|streaming|state|shape|memory
    """

    def setUp(self):
        """Prepare deterministic MHDSRA2 test fixtures.

        中文说明:
        - 调用方 / Called by: `unittest` 在每个测试方法执行前自动调用
        - 调用对象 / Calls: `torch.manual_seed`, `MHDSRA2Config`, `MultiHeadDSRA2`
        - 作用 / Purpose: 构造稳定的配置、模型和测试维度，避免随机性导致误报
        - 变量 / Variables:
          `dim` 特征维度, `heads` 头数, `slots` 全局槽位数, `local_window` 局部窗口大小
        - 接入 / Integration: 无需手动调用，新增测试方法会自动复用
        - 错误处理 / Error handling: 依赖配置构造中的 ValueError 进行参数校验
        - 关键词 / Keywords:
          setUp|fixture|seed|config|layer|dimension|heads|slots|window|deterministic
        """
        torch.manual_seed(0)
        self.dim = 32
        self.heads = 4
        self.d_head = self.dim // self.heads
        self.slots = 8
        self.local_window = 6
        self.cfg = MHDSRA2Config(
            dim=self.dim,
            heads=self.heads,
            slots=self.slots,
            read_topk=3,
            write_topk=2,
            local_window=self.local_window,
            use_local=True,
            use_retrieval=True,
            detach_state=True,
        )
        self.layer = MultiHeadDSRA2(self.cfg)

    def test_forward_returns_expected_shapes_and_aux(self):
        """Validate forward output, state and fusion aux tensor shapes.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `MultiHeadDSRA2.forward`
        - 作用 / Purpose: 校验三路融合前向传播返回的输出、状态和辅助统计形状是否正确
        - 变量 / Variables:
          `x` 输入块, `rk/rv` 检索记忆, `y` 输出, `state` 下一状态, `aux` 门控统计
        - 接入 / Integration: 为后续修改 attention 路由逻辑提供基础回归保护
        - 错误处理 / Error handling: 通过形状和有限值断言捕获回归
        - 关键词 / Keywords:
          forward|aux|shape|gates|slot_usage|confidence|retrieval|local|output|tensor
        """
        batch = 2
        tokens = 5
        retrieved = 4
        x = torch.randn(batch, tokens, self.dim)
        rk = torch.randn(batch, self.heads, retrieved, self.d_head)
        rv = torch.randn(batch, self.heads, retrieved, self.d_head)

        y, state, aux = self.layer(x, retrieved_k=rk, retrieved_v=rv, return_aux=True)

        self.assertEqual(y.shape, (batch, tokens, self.dim))
        self.assertEqual(state.slot_k.shape, (batch, self.heads, self.slots, self.d_head))
        self.assertEqual(state.slot_v.shape, (batch, self.heads, self.slots, self.d_head))
        self.assertEqual(state.usage.shape, (batch, self.heads, self.slots))
        self.assertEqual(state.confidence.shape, (batch, self.heads, self.slots))
        self.assertIsNotNone(state.local_k)
        self.assertIsNotNone(state.local_v)
        self.assertEqual(state.local_k.shape, (batch, self.heads, tokens, self.d_head))
        self.assertEqual(state.local_v.shape, (batch, self.heads, tokens, self.d_head))
        self.assertEqual(aux["gates_mean"].shape, (self.heads, 3))
        self.assertEqual(aux["read_mass"].shape, (batch, self.heads, self.slots))
        self.assertEqual(aux["slot_usage"].shape, (batch, self.heads, self.slots))
        self.assertEqual(aux["slot_confidence"].shape, (batch, self.heads, self.slots))
        self.assertTrue(torch.isfinite(y).all())

    def test_state_updates_across_chunks(self):
        """Validate that MHDSRA2 state advances and changes across chunks.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `MultiHeadDSRA2.forward`
        - 作用 / Purpose: 校验流式分块时 position、slot usage 与 local cache 会持续更新
        - 变量 / Variables:
          `x1/x2` 两段输入块, `state1/state2` 连续状态
        - 接入 / Integration: 保护后续对流式推理和状态写入逻辑的重构
        - 错误处理 / Error handling: 用位置、缓存长度和状态变化断言识别回归
        - 关键词 / Keywords:
          state|chunk|streaming|position|usage|local_cache|update|slot|sequence|regression
        """
        x1 = torch.randn(2, 4, self.dim)
        x2 = torch.randn(2, 5, self.dim)

        _, state1 = self.layer(x1)
        _, state2 = self.layer(x2, state=state1)

        self.assertEqual(state1.position, 4)
        self.assertEqual(state2.position, 9)
        self.assertIsNotNone(state2.local_k)
        self.assertLessEqual(state2.local_k.shape[2], self.local_window)
        self.assertTrue(torch.isfinite(state2.slot_k).all())
        self.assertTrue(torch.isfinite(state2.slot_v).all())
        self.assertTrue((state2.usage >= 0).all())
        self.assertFalse(torch.allclose(state1.slot_k, state2.slot_k))
        self.assertGreater(state2.usage.sum().item(), state1.usage.sum().item())

    def test_forward_step_returns_single_token_output_and_kv_cache(self):
        """Validate autoregressive one-step decoding compatibility outputs.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `MultiHeadDSRA2.forward_step`
        - 作用 / Purpose: 校验 MHDSRA2 已提供与 JSON generation 评测兼容的逐步解码接口
        - 变量 / Variables:
          `step_input_1/2` 两个连续 token hidden, `state1/state2` 连续流式状态,
          `kv_cache1/2` 返回的局部缓存
        - 接入 / Integration: 保护 `scripts.json_retrieval_test` 所依赖的 `forward_step` 接口
        - 错误处理 / Error handling: 通过形状、缓存存在性和 position 递增断言发现回归
        - 关键词 / Keywords:
          forward_step|generation|decode|single_token|kv_cache|state|mhdsra2|compat|autoregressive|回归
        """
        step_input_1 = torch.randn(2, 1, self.dim)
        step_input_2 = torch.randn(2, 1, self.dim)

        out1, state1, kv_cache1 = self.layer.forward_step(step_input_1)
        out2, state2, kv_cache2 = self.layer.forward_step(
            step_input_2,
            S_prev=state1,
            kv_cache=kv_cache1,
        )

        self.assertEqual(out1.shape, (2, 1, self.dim))
        self.assertEqual(out2.shape, (2, 1, self.dim))
        self.assertIsNotNone(kv_cache1)
        self.assertIsNotNone(kv_cache2)
        self.assertIsNotNone(kv_cache1[0])
        self.assertIsNotNone(kv_cache1[1])
        self.assertIsNotNone(kv_cache2[0])
        self.assertIsNotNone(kv_cache2[1])
        self.assertEqual(state1.position, 1)
        self.assertEqual(state2.position, 2)
        self.assertLessEqual(kv_cache2[0].shape[2], self.local_window)
        self.assertTrue(torch.isfinite(out2).all())

    def test_paged_exact_memory_retrieve_returns_expected_page(self):
        """Validate page retrieval shape and selected token positions.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `PagedExactMemory.append`, `PagedExactMemory.retrieve`
        - 作用 / Purpose: 校验分页记忆会优先召回与 query 最相似页面中的 token
        - 变量 / Variables:
          `mem` 分页记忆, `key/value` 历史键值, `query` 查询, `rk/rv/pos` 召回结果
        - 接入 / Integration: 为未来替换 ANN/分页策略时保留参考行为基线
        - 错误处理 / Error handling: 通过位置范围和张量形状断言检查召回结果
        - 关键词 / Keywords:
          paged|retrieve|page|token|positions|memory|query|append|recall|shape
        """
        mem = PagedExactMemory(page_size=4, dtype=torch.float32)
        key = torch.zeros(1, self.heads, 8, self.d_head)
        value = torch.zeros(1, self.heads, 8, self.d_head)

        key[:, :, :4, 0] = 1.0
        value[:, :, :4, 0] = 10.0
        key[:, :, 4:, 1] = 1.0
        value[:, :, 4:, 1] = 20.0
        mem.append(key, value)

        query = torch.zeros(1, self.heads, 2, self.d_head)
        query[:, :, :, 1] = 1.0
        rk, rv, pos = mem.retrieve(query, top_pages=1, max_tokens=3)

        self.assertIsNotNone(rk)
        self.assertIsNotNone(rv)
        self.assertIsNotNone(pos)
        self.assertEqual(rk.shape, (1, self.heads, 3, self.d_head))
        self.assertEqual(rv.shape, (1, self.heads, 3, self.d_head))
        self.assertEqual(pos.shape, (3,))
        self.assertTrue(torch.all((pos >= 4) & (pos < 8)))
        self.assertTrue(torch.allclose(rv[..., 1], torch.full_like(rv[..., 1], 20.0)))

    def test_paged_exact_memory_retrieve_prefers_latest_position_on_tie(self):
        """Validate latest-wins ordering for equal-score external memory matches.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `PagedExactMemory.append`, `PagedExactMemory.retrieve`
        - 作用 / Purpose: 同一 key 多次写入且相似度相同的情况下，检索应返回最新 position 的 value
        - 变量 / Variables:
          `old_position/new_position` 为同 key 的两次写入位置, `pos` 为召回结果位置,
          `rv` 为召回 value，用于确认最新写入的 value 胜出
        - 接入 / Integration: 保护 external memory 的 latest-wins 语义，避免 Diagnostic B 回退到旧事实
        - 错误处理 / Error handling: 如果返回空结果、旧位置或旧 value，断言立即失败
        - 关键词 / Keywords:
          latest_wins|external_memory|tie_break|position|retrieve|paged|same_key|value|regression|最新优先
        """
        mem = PagedExactMemory(page_size=4, dtype=torch.float32)
        key = torch.zeros(1, self.heads, 8, self.d_head)
        value = torch.zeros(1, self.heads, 8, self.d_head)
        old_position = 1
        new_position = 6
        key[:, :, old_position, 2] = 1.0
        key[:, :, new_position, 2] = 1.0
        value[:, :, old_position, 0] = 1.0
        value[:, :, new_position, 1] = 1.0
        mem.append(key, value)

        query = torch.zeros(1, self.heads, 1, self.d_head)
        query[:, :, :, 2] = 1.0
        _, rv, pos = mem.retrieve(query, top_pages=2, max_tokens=1)

        self.assertIsNotNone(rv)
        self.assertIsNotNone(pos)
        self.assertEqual(int(pos[0].item()), new_position)
        self.assertTrue(torch.all(rv[0, :, 0, 1] > rv[0, :, 0, 0]))

    def test_gates_degrade_when_local_disabled(self):
        """Validate fusion gates when local branch is disabled.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `MultiHeadDSRA2.forward`
        - 作用 / Purpose: 校验 use_local=False 时 gate 的 local 分量被强制置零并重新归一化
        - 变量 / Variables:
          `cfg` 关闭 local 的配置, `aux["gates_mean"]` 门控统计
        - 接入 / Integration: 用于保护分支开关逻辑不被重构破坏
        - 错误处理 / Error handling: 通过精确零值断言捕获回归
        - 关键词 / Keywords:
          gates|local|disable|degenerate|normalize|branch|fusion|mhdsra2|aux|regression
        """
        cfg = MHDSRA2Config(
            dim=self.dim,
            heads=self.heads,
            slots=self.slots,
            read_topk=3,
            write_topk=2,
            local_window=self.local_window,
            use_local=False,
            use_retrieval=True,
            detach_state=True,
        )
        layer = MultiHeadDSRA2(cfg)
        x = torch.randn(2, 5, self.dim)
        rk = torch.randn(2, self.heads, 4, self.d_head)
        rv = torch.randn(2, self.heads, 4, self.d_head)
        _, _, aux = layer(x, retrieved_k=rk, retrieved_v=rv, return_aux=True)
        self.assertTrue((aux["gates_mean"][:, 1] == 0).all().item())
        self.assertTrue(torch.isfinite(aux["gates_mean"]).all().item())

    def test_retrieval_attention_prefers_exact_match_over_near_distractors(self):
        """Validate retrieval attention is sharp enough for rare exact matches.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `MHDSRA2Config`, `MultiHeadDSRA2._retrieval_attention`
        - 作用 / Purpose: 构造 1 个 exact right token 和 3 个 near wrong token，
          验证 retrieval softmax 不会让多个近似错误项用数量压过最高分精确匹配
        - 变量 / Variables:
          `query` 为单步查询, `retrieved_k` 包含一个最高分 key 和三个次高分 key,
          `retrieved_v` 的第 0/1 维分别代表 wrong/right value 分数
        - 接入 / Integration: 保护 Diagnostic C 的 paged recall 分支，防止 anti-fixation 回退到 majority trap
        - 错误处理 / Error handling: 如果 right value 不高于 wrong value，断言失败
        - 关键词 / Keywords:
          retrieval_attention|exact_match|near_distractor|softmax|retrieval_tau|anti_fixation|paged_recall|mhdsra2|test|精确匹配
        """
        cfg = MHDSRA2Config(
            dim=4,
            heads=1,
            slots=2,
            local_window=0,
            use_local=False,
            use_retrieval=True,
            retrieval_tau=8.0,
        )
        layer = MultiHeadDSRA2(cfg)
        query = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
        retrieved_k = torch.tensor(
            [[[[4.0, 0.0, 0.0, 0.0], [3.6, 0.0, 0.0, 0.0], [3.6, 0.0, 0.0, 0.0], [3.6, 0.0, 0.0, 0.0]]]]
        )
        retrieved_v = torch.tensor(
            [[[[0.0, 3.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]]]]
        )

        output = layer._retrieval_attention(query, retrieved_k, retrieved_v)

        self.assertGreater(output[0, 0, 0, 1].item(), output[0, 0, 0, 0].item())

    def test_gates_degrade_when_retrieval_disabled(self):
        """Validate fusion gates when retrieval branch is disabled.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `MultiHeadDSRA2.forward`
        - 作用 / Purpose: 校验 use_retrieval=False 时 gate 的 retrieval 分量被强制置零并重新归一化
        - 变量 / Variables:
          `cfg` 关闭 retrieval 的配置, `aux["gates_mean"]` 门控统计
        - 接入 / Integration: 用于保护分支开关逻辑不被重构破坏
        - 错误处理 / Error handling: 通过精确零值断言捕获回归
        - 关键词 / Keywords:
          gates|retrieval|disable|degenerate|normalize|branch|fusion|mhdsra2|aux|regression
        """
        cfg = MHDSRA2Config(
            dim=self.dim,
            heads=self.heads,
            slots=self.slots,
            read_topk=3,
            write_topk=2,
            local_window=self.local_window,
            use_local=True,
            use_retrieval=False,
            detach_state=True,
        )
        layer = MultiHeadDSRA2(cfg)
        x = torch.randn(2, 5, self.dim)
        rk = torch.randn(2, self.heads, 4, self.d_head)
        rv = torch.randn(2, self.heads, 4, self.d_head)
        _, _, aux = layer(x, retrieved_k=rk, retrieved_v=rv, return_aux=True)
        self.assertTrue((aux["gates_mean"][:, 2] == 0).all().item())
        self.assertTrue(torch.isfinite(aux["gates_mean"]).all().item())

    def test_gates_degrade_when_local_and_retrieval_disabled(self):
        """Validate fusion gates when only slot branch remains.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `MultiHeadDSRA2.forward`
        - 作用 / Purpose: 校验 use_local=False 且 use_retrieval=False 时 gates 退化为 [1,0,0]
        - 变量 / Variables:
          `aux["gates_mean"]` 门控统计
        - 接入 / Integration: 用于保护三路融合在极端开关组合下的稳定行为
        - 错误处理 / Error handling: 通过退化形态断言捕获回归
        - 关键词 / Keywords:
          gates|slot|local|retrieval|disable|degenerate|normalize|mhdsra2|aux|regression
        """
        cfg = MHDSRA2Config(
            dim=self.dim,
            heads=self.heads,
            slots=self.slots,
            read_topk=3,
            write_topk=2,
            local_window=self.local_window,
            use_local=False,
            use_retrieval=False,
            detach_state=True,
        )
        layer = MultiHeadDSRA2(cfg)
        x = torch.randn(2, 5, self.dim)
        rk = torch.randn(2, self.heads, 4, self.d_head)
        rv = torch.randn(2, self.heads, 4, self.d_head)
        _, _, aux = layer(x, retrieved_k=rk, retrieved_v=rv, return_aux=True)
        self.assertTrue(torch.allclose(aux["gates_mean"][:, 0], torch.ones(self.heads)))
        self.assertTrue((aux["gates_mean"][:, 1] == 0).all().item())
        self.assertTrue((aux["gates_mean"][:, 2] == 0).all().item())

    def test_paged_exact_memory_invalidate_before_excludes_old_pages(self):
        """Validate invalidate_before prevents retrieval from old pages.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `PagedExactMemory.append`, `PagedExactMemory.invalidate_before`, `PagedExactMemory.retrieve`
        - 作用 / Purpose: 校验旧页失效后，retrieve 不会返回失效页内 token 的 position
        - 变量 / Variables:
          `position` 失效阈值, `pos` 召回位置
        - 接入 / Integration: 用于保护内存截断/滑动窗口式回收策略
        - 错误处理 / Error handling: 通过位置范围断言捕获回归
        - 关键词 / Keywords:
          invalidate_before|paged|memory|retrieve|position|forget|page|valid|regression|mhdsra2
        """
        mem = PagedExactMemory(page_size=4, dtype=torch.float32)
        key = torch.zeros(1, self.heads, 8, self.d_head)
        value = torch.zeros(1, self.heads, 8, self.d_head)
        key[:, :, :4, 0] = 1.0
        value[:, :, :4, 0] = 10.0
        key[:, :, 4:, 1] = 1.0
        value[:, :, 4:, 1] = 20.0
        mem.append(key, value)

        query = torch.zeros(1, self.heads, 2, self.d_head)
        query[:, :, :, 0] = 1.0
        _, _, pos_before = mem.retrieve(query, top_pages=1, max_tokens=3)
        self.assertIsNotNone(pos_before)
        self.assertTrue(torch.all((pos_before >= 0) & (pos_before < 4)))

        mem.invalidate_before(4)
        _, _, pos_after = mem.retrieve(query, top_pages=1, max_tokens=3)
        self.assertIsNotNone(pos_after)
        self.assertTrue(torch.all((pos_after >= 4) & (pos_after < 8)))

    def test_paged_exact_memory_invalidate_before_all_pages_returns_none(self):
        """Validate invalidating all pages makes retrieve return None.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `PagedExactMemory.append`, `PagedExactMemory.invalidate_before`, `PagedExactMemory.retrieve`
        - 作用 / Purpose: 校验全部页面失效时 retrieve 返回 (None, None, None)
        - 变量 / Variables:
          `rk/rv/pos` 召回结果
        - 接入 / Integration: 用于保护记忆清空/回收极端场景
        - 错误处理 / Error handling: 通过 None 断言捕获回归
        - 关键词 / Keywords:
          invalidate_before|paged|memory|retrieve|none|forget|page|valid|regression|edge
        """
        mem = PagedExactMemory(page_size=4, dtype=torch.float32)
        key = torch.randn(1, self.heads, 8, self.d_head)
        value = torch.randn(1, self.heads, 8, self.d_head)
        mem.append(key, value)
        mem.invalidate_before(10_000)

        rk, rv, pos = mem.retrieve(key[:, :, :2, :], top_pages=1, max_tokens=3)
        self.assertIsNone(rk)
        self.assertIsNone(rv)
        self.assertIsNone(pos)


if __name__ == "__main__":
    unittest.main()
