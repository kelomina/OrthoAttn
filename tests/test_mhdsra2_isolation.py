import argparse
import unittest

import torch
import torch.nn.functional as F

from scripts.diagnostic_memory_benchmark import (
    _build_mhdsra2_layer,
    _decode_prediction,
    _generate_chunk,
    build_error_override_cases,
    build_token_vector,
)
from src.dsra.mhdsra2.paged_exact_memory import PagedExactMemory


class TestMHDSRA2Isolation(unittest.TestCase):
    """Minimal isolation probes for retrieval, fusion and overwrite diagnostics.

    中文说明:
    - 调用方 / Called by: `python -m unittest tests.test_mhdsra2_isolation -v`
    - 调用对象 / Calls:
      `PagedExactMemory.append`, `PagedExactMemory.retrieve`,
      `build_error_override_cases`, `_build_mhdsra2_layer`, `_decode_prediction`
    - 作用 / Purpose: 按用户要求执行 4 个最小隔离验证，定位 paged retrieval、
      retrieval-only forward、correction overwrite trace 与 external memory latest-wins
    - 变量 / Variables:
      `args` 为小规模诊断配置, `layer` 为确定性 MHDSRA2 层, `state` 为流式状态,
      `metrics` 为 overwrite 追踪指标
    - 接入 / Integration: 放在 `tests/` 下，可由 `unittest` 单独运行，不影响现有主测试入口
    - 错误处理 / Error handling: 通过断言保障 probe 至少完成可解释的最小执行；关键诊断值通过打印输出
    - 关键词 / Keywords:
      isolation|retrieval_only|oracle_hit|overwrite_trace|latest_wins|paged_memory|fusion_gate|mhdsra2|diagnostic|隔离验证
    """

    def setUp(self):
        """Prepare deterministic CPU fixtures shared by all isolation probes.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `torch.manual_seed`, `argparse.Namespace`
        - 作用 / Purpose: 固定随机种子并创建与诊断脚本一致的小规模配置，保证 probe 可复现
        - 变量 / Variables:
          `slots/key_count/value_count` 分别是路由槽位数、键空间大小和值空间大小,
          `dim` 为共享 token probing 维度
        - 接入 / Integration: 每个测试方法自动复用，无需手动调用
        - 错误处理 / Error handling: 若配置非法，后续构造函数会直接抛出异常
        - 关键词 / Keywords:
          setUp|fixture|seed|cpu|args|slots|key_count|value_count|dim|deterministic
        """
        torch.manual_seed(0)
        self.args = argparse.Namespace(
            diagnostic_device="cpu",
            diagnostic_slots=4,
            diagnostic_key_count=8,
            diagnostic_value_count=8,
            diagnostic_chunk_size=1,
            diagnostic_page_size=4,
            diagnostic_retrieved_top_pages=1,
            diagnostic_retrieved_max_tokens=4,
            diagnostic_exact_seq_len=32,
            diagnostic_exact_fact_spacing=4,
            diagnostic_override_seq_len=32,
            diagnostic_override_gap_grid=[8],
            diagnostic_fixation_seq_len=32,
            diagnostic_fixation_distractor_grid=[4],
        )
        self.device = torch.device("cpu")
        self.slots = self.args.diagnostic_slots
        self.key_count = self.args.diagnostic_key_count
        self.value_count = self.args.diagnostic_value_count
        self.dim = self.slots + self.key_count + self.value_count

    def _make_query_heads(self, layer, token_vector: torch.Tensor) -> torch.Tensor:
        """Convert one probing token into MHDSRA2 head-space query tensor.

        中文说明:
        - 调用方 / Called by:
          `test_retrieval_only_forward_prefers_retrieved_value`,
          `test_correction_slot_overwrite_trace_records_required_metrics`
        - 调用对象 / Calls: `layer.qkv`, `layer._to_heads`
        - 作用 / Purpose: 将单个 probe token 通过当前层的 q 投影转换为 `[B,H,T,d]` 查询表示
        - 变量 / Variables: `token_vector` 为单 token 向量, `layer` 为待诊断的 MHDSRA2 层
        - 接入 / Integration: 构造 retrieval-only/query-only 诊断时复用
        - 错误处理 / Error handling: 依赖底层投影层做形状校验；异常直接向上抛出
        - 关键词 / Keywords:
          query_heads|qkv|to_heads|token_vector|mhdsra2|query|probe|tensor|helper|查询构造
        """
        chunk = token_vector.unsqueeze(0).unsqueeze(0).to(self.device)
        q, _, _ = layer.qkv(chunk).chunk(3, dim=-1)
        return layer._to_heads(q)

    def _make_value_heads(self, layer, token_vector: torch.Tensor) -> torch.Tensor:
        """Convert one probing token into MHDSRA2 head-space value tensor.

        中文说明:
        - 调用方 / Called by: `test_retrieval_only_forward_prefers_retrieved_value`
        - 调用对象 / Calls: `layer.qkv`, `layer._to_heads`
        - 作用 / Purpose: 将单个 probe token 转换为 `[B,H,T,d]` value 表示，用于 retrieval-only forward
        - 变量 / Variables: `token_vector` 为待作为 retrieved value 的单 token 向量
        - 接入 / Integration: 与 `_make_query_heads` 配套，供 retrieval-only 诊断复用
        - 错误处理 / Error handling: 依赖底层线性层与张量形状检查
        - 关键词 / Keywords:
          value_heads|qkv|to_heads|retrieved_v|mhdsra2|probe|tensor|helper|value|值构造
        """
        chunk = token_vector.unsqueeze(0).unsqueeze(0).to(self.device)
        _, _, v = layer.qkv(chunk).chunk(3, dim=-1)
        return layer._to_heads(v)

    def test_paged_recall_oracle_hit_records_target_position_and_value(self):
        """Probe 1: paged recall oracle hit without model output path.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `PagedExactMemory.append`, `PagedExactMemory.retrieve`
        - 作用 / Purpose: 不经过模型输出头，只验证 retrieval 是否能把目标 fact 的位置和值 token 召回
        - 变量 / Variables:
          `target_position` 为 oracle fact 位置, `target_value` 为目标值 token,
          `retrieved_positions` 为召回位置列表
        - 接入 / Integration: 对 memory retrieval 的最小隔离验证
        - 错误处理 / Error handling: 若未命中目标位置或目标值 token，则断言失败
        - 关键词 / Keywords:
          oracle_hit|paged_recall|retrieval|position|value_token|memory|probe|token|query|最小验证
        """
        mem = PagedExactMemory(page_size=4, dtype=torch.float32)
        key = torch.zeros(1, 1, 12, 8)
        value = torch.zeros(1, 1, 12, 8)

        key[:, :, 0:4, 0] = 1.0
        key[:, :, 4:8, 1] = 0.25
        key[:, :, 8:12, 3] = 1.0
        key[:, :, 8:12, 4] = 0.20
        target_position = 10
        key[:, :, target_position, 3] = 1.0
        key[:, :, target_position, 4] = 0.0

        target_value = torch.tensor([0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        value[:, :, 8:12, 0] = 1.0
        value[:, :, target_position, :] = target_value
        mem.append(key, value)

        query = torch.zeros(1, 1, 1, 8)
        query[:, :, :, 3] = 1.0
        rk, rv, pos = mem.retrieve(query, top_pages=1, max_tokens=3)

        self.assertIsNotNone(rk)
        self.assertIsNotNone(rv)
        self.assertIsNotNone(pos)
        retrieved_positions = pos.tolist()
        retrieved_value_rows = rv[0, 0]
        value_hit = any(torch.allclose(row, target_value, atol=1e-6) for row in retrieved_value_rows)

        print(
            f"Test 1 - retrieved_positions={retrieved_positions}, "
            f"target_position_hit={target_position in retrieved_positions}, value_token_hit={value_hit}"
        )

        self.assertIn(target_position, retrieved_positions)
        self.assertTrue(value_hit)

    def test_retrieval_only_forward_prefers_retrieved_value(self):
        """Probe 2: retrieval-only forward with slot/local gates suppressed.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `_build_mhdsra2_layer`, `build_token_vector`, `_decode_prediction`
        - 作用 / Purpose: 强制 slot/local 近似为 0、retrieval 近似为 1，隔离验证 retrieval content 与 output head
        - 变量 / Variables:
          `query_token` 为仅含 key 的查询 token, `retrieved_v` 为带目标值的检索结果,
          `predicted_value` 为解码出的输出值编号
        - 接入 / Integration: 可用于判断 retrieval-only 成功但正常前向失败时是否为 fusion gate 问题
        - 错误处理 / Error handling: 若 retrieval-only 不能恢复目标值或门控未偏向 retrieval，则断言失败
        - 关键词 / Keywords:
          retrieval_only|forward|gate|slot_gate|local_gate|retrieval_gate|output_head|fusion|probe|隔离
        """
        layer = _build_mhdsra2_layer(
            self.dim,
            self.slots,
            use_retrieval=True,
            key_count=self.key_count,
        ).to(self.device)
        with torch.no_grad():
            layer.fuse_gate.weight.zero_()
            layer.fuse_gate.bias[:] = torch.tensor(
                [-20.0, -20.0, 20.0],
                dtype=layer.fuse_gate.bias.dtype,
            )

        key_id = self.slots
        target_value_id = 1
        query_token = build_token_vector(
            key_id,
            None,
            slots=self.slots,
            key_count=self.key_count,
            value_count=self.value_count,
            route_scale=6.0,
            key_scale=2.0,
            value_scale=0.0,
        )
        retrieved_token = build_token_vector(
            key_id,
            target_value_id,
            slots=self.slots,
            key_count=self.key_count,
            value_count=self.value_count,
            route_scale=6.0,
            key_scale=2.0,
            value_scale=3.0,
        )

        query_chunk = query_token.unsqueeze(0).unsqueeze(0).to(self.device)
        retrieved_k = self._make_query_heads(layer, query_token)
        retrieved_v = self._make_value_heads(layer, retrieved_token)

        output, _, aux = layer(
            query_chunk,
            state=None,
            retrieved_k=retrieved_k,
            retrieved_v=retrieved_v,
            return_aux=True,
        )
        predicted_value, _ = _decode_prediction(
            output[0, -1].cpu(),
            tuple(range(self.value_count)),
            self.slots,
            self.key_count,
            self.value_count,
        )
        gates = aux["gates_mean"][0].detach().cpu().tolist()

        print(
            "Test 2 - "
            f"gates(slot/local/retrieval)={[round(x, 6) for x in gates]}, "
            f"predicted_value={predicted_value}, target_value={target_value_id}"
        )

        self.assertLess(gates[0], 1e-6)
        self.assertEqual(gates[1], 0.0)
        self.assertGreater(gates[2], 0.999999)
        self.assertEqual(predicted_value, target_value_id)

    def test_correction_slot_overwrite_trace_records_required_metrics(self):
        """Probe 3: record correction overwrite trace around error override chunk.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls:
          `build_error_override_cases`, `_generate_chunk`, `_build_mhdsra2_layer`,
          `MultiHeadDSRA2._slot_read`, `torch.nn.functional.cosine_similarity`
        - 作用 / Purpose: 在 error override 场景中记录 correction chunk 前后的 overwrite trace 指标
        - 变量 / Variables:
          `stale_slot_id` 为 correction 前 query 最偏向的旧槽位,
          `corrected_slot_id` 为 correction chunk 写入质量最大的槽位,
          `metrics` 为需要输出的诊断值
        - 接入 / Integration: 该 probe 输出的指标可直接用于判断 forget gate 是否激活、输出是否仍偏向 stale fact
        - 错误处理 / Error handling: 若关键统计缺失、非有限值或写入质量为 0，则断言失败
        - 关键词 / Keywords:
          correction|overwrite|trace|forget_gate|write_mass|read_mass|stale_slot|corrected_slot|probe|诊断
        """
        cases = build_error_override_cases(self.args, self.dim)
        case = cases[0]
        layer = _build_mhdsra2_layer(
            self.dim,
            self.slots,
            use_retrieval=False,
            key_count=self.key_count,
        ).to(self.device)
        filler_vector = build_token_vector(
            None,
            None,
            slots=self.slots,
            key_count=self.key_count,
            value_count=self.value_count,
            route_scale=0.0,
            key_scale=0.0,
            value_scale=0.0,
        )

        correction_position = max(position for position in case.special_tokens if position != case.query_position)
        state = None
        for position in range(correction_position):
            chunk = _generate_chunk(case, position, position + 1, filler_vector, self.device)
            _, state, _ = layer(chunk, state=state, return_aux=True)

        pre_slot_k = state.slot_k.detach().clone()
        pre_slot_v = state.slot_v.detach().clone()
        query_token = case.special_tokens[case.query_position]
        query_heads = self._make_query_heads(layer, query_token)
        _, pre_read_aux = layer._slot_read(query_heads, state)
        stale_slot_id = int(pre_read_aux["read_mass"][0, 0].argmax().item())

        correction_chunk = _generate_chunk(
            case,
            correction_position,
            correction_position + 1,
            filler_vector,
            self.device,
        )
        _, corrected_state, correction_aux = layer(correction_chunk, state=state, return_aux=True)
        write_stats = correction_aux["write_stats"]
        corrected_slot_id = int(write_stats["write_mass"][0, 0].argmax().item())

        _, post_read_aux = layer._slot_read(query_heads, corrected_state)
        post_read_slot_id = int(post_read_aux["read_mass"][0, 0].argmax().item())
        slot_k_before = pre_slot_k[0, 0, corrected_slot_id]
        slot_k_after = corrected_state.slot_k[0, 0, corrected_slot_id]
        slot_v_before = pre_slot_v[0, 0, corrected_slot_id]
        slot_v_after = corrected_state.slot_v[0, 0, corrected_slot_id]
        value_offset = self.slots + self.key_count
        old_value_id = int(case.metadata["old_value_id"])
        new_value_id = int(case.metadata["new_value_id"])
        corrected_value_scores = corrected_state.slot_v[
            0,
            0,
            corrected_slot_id,
            value_offset : value_offset + self.value_count,
        ]
        cosine_after = F.cosine_similarity(
            slot_k_before.unsqueeze(0),
            slot_k_after.unsqueeze(0),
            dim=-1,
            eps=1e-6,
        ).item()
        metrics = {
            "stale_slot_id": stale_slot_id,
            "corrected_slot_id": corrected_slot_id,
            "slot_k_cosine_change": float(1.0 - cosine_after),
            "slot_v_norm_change": float(abs(slot_v_after.norm().item() - slot_v_before.norm().item())),
            "write_mass": float(write_stats["write_mass"][0, 0, corrected_slot_id].item()),
            "forget_gate_max": float(write_stats["forget_gate_max"].item()),
            "forget_gate_mean": float(write_stats["forget_gate_mean"].item()),
            "read_mass_stale_slot": float(post_read_aux["read_mass"][0, 0, stale_slot_id].item()),
            "read_mass_corrected_slot": float(post_read_aux["read_mass"][0, 0, corrected_slot_id].item()),
            "post_read_slot_id": post_read_slot_id,
            "old_value_score_after": float(corrected_value_scores[old_value_id].item()),
            "new_value_score_after": float(corrected_value_scores[new_value_id].item()),
        }

        print(f"Test 3 - overwrite_trace={metrics}")

        self.assertGreaterEqual(stale_slot_id, 0)
        self.assertGreaterEqual(corrected_slot_id, 0)
        self.assertLess(stale_slot_id, self.slots)
        self.assertLess(corrected_slot_id, self.slots)
        self.assertGreater(metrics["write_mass"], 0.0)
        self.assertEqual(stale_slot_id, corrected_slot_id)
        self.assertEqual(post_read_slot_id, corrected_slot_id)
        self.assertGreater(metrics["new_value_score_after"], metrics["old_value_score_after"])
        for metric_name, metric_value in metrics.items():
            if metric_name.endswith("_id"):
                continue
            self.assertTrue(torch.isfinite(torch.tensor(metric_value)), msg=f"{metric_name} must be finite")

    def test_latest_wins_external_memory_probe_reports_observed_value(self):
        """Probe 4: check whether external memory returns latest value for same key.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `PagedExactMemory.append`, `PagedExactMemory.retrieve`
        - 作用 / Purpose: 对同一 key 连续写入两次，直接观察外部 memory 是否具备 latest-wins 语义
        - 变量 / Variables:
          `old_position/new_position` 为两次写入位置, `returned_position` 为检索返回的位置,
          `latest_wins` 表示是否返回最新值
        - 接入 / Integration: 该 probe 主要用于诊断，不把 latest-wins 作为当前实现的硬性通过条件
        - 错误处理 / Error handling: 只要求 memory 至少返回匹配 key 的一个结果；latest-wins 通过打印报告
        - 关键词 / Keywords:
          latest_wins|external_memory|symbolic|same_key|latest_value|retrieve|paged|probe|diagnostic|外部记忆
        """
        mem = PagedExactMemory(page_size=4, dtype=torch.float32)
        key = torch.zeros(1, 1, 8, 6)
        value = torch.zeros(1, 1, 8, 6)
        old_position = 1
        new_position = 6
        key[:, :, old_position, 2] = 1.0
        key[:, :, new_position, 2] = 1.0
        value[:, :, old_position, 0] = 1.0
        value[:, :, new_position, 1] = 1.0
        mem.append(key, value)

        query = torch.zeros(1, 1, 1, 6)
        query[:, :, :, 2] = 1.0
        _, rv, pos = mem.retrieve(query, top_pages=2, max_tokens=1)

        self.assertIsNotNone(rv)
        self.assertIsNotNone(pos)
        returned_position = int(pos[0].item())
        returned_value = rv[0, 0, 0]
        latest_wins = returned_position == new_position and returned_value[1].item() > returned_value[0].item()

        print(
            "Test 4 - "
            f"returned_position={returned_position}, old_position={old_position}, new_position={new_position}, "
            f"latest_wins={latest_wins}"
        )

        self.assertEqual(returned_position, new_position)
        self.assertTrue(latest_wins)


if __name__ == "__main__":
    unittest.main()
