import unittest
import torch
from dsra_layer import DSRA_Chunk_Layer
from src.dsra.mhdsra2.improved_dsra_mha import MHDSRA2State

class TestDSRAMath(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dim = 64
        self.K = 16
        self.kr = 4
        self.layer = DSRA_Chunk_Layer(self.dim, K=self.K, kr=self.kr).to(self.device)

    def test_forward_dimensions(self):
        """Validate DSRA compatibility output and MHDSRA2 state dimensions.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `DSRA_Chunk_Layer.forward`
        - 作用 / Purpose: 确认旧 DSRA 入口已返回 MHDSRA2 状态且输出/缓存形状兼容
        - 变量 / Variables: `x` 输入 chunk, `S_next` 新状态, `bypass_kv` 旧接口缓存
        - 接入 / Integration: 保护 `dsra_layer.py` 的兼容前向接口
        - 错误处理 / Error handling: 通过断言暴露状态形状或缓存形状回归
        - 关键词 / Keywords:
          forward|dimensions|mhdsra2|state|cache|compat|dsra|shape|test|维度
        """
        B, T = 2, 128
        x = torch.randn(B, T, self.dim).to(self.device)
        
        out, S_next, bypass_kv, _ = self.layer(x)
        
        self.assertEqual(out.shape, (B, T, self.dim))
        self.assertIsInstance(S_next, MHDSRA2State)
        self.assertEqual(S_next.slot_v.shape[0], B)
        self.assertEqual(S_next.slot_v.shape[2], self.K)
        self.assertEqual(S_next.slot_v.shape[1] * S_next.slot_v.shape[3], self.dim)
        expected_cache_tokens = min(T, self.layer.spec.local_window)
        self.assertEqual(bypass_kv[0].shape, (B, expected_cache_tokens, self.dim))
        self.assertEqual(bypass_kv[1].shape, (B, expected_cache_tokens, self.dim))

    def test_mhdsra2_state_update_proxy_is_finite(self):
        """Validate MHDSRA2 slot update diagnostics after replacing orthogonal writes.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `DSRA_Chunk_Layer.forward`, `torch.isfinite`
        - 作用 / Purpose: 新机制不再使用旧正交逆矩阵，改为检查 slot 更新代理量稳定存在
        - 变量 / Variables: `V_proxy` 是 `last_V_orth` 兼容诊断代理, `S_next` 是新状态
        - 接入 / Integration: 保护饱和度脚本继续读取 `last_V_orth`
        - 错误处理 / Error handling: NaN/Inf 或形状错误会触发断言失败
        - 关键词 / Keywords:
          mhdsra2|state_update|proxy|last_V_orth|finite|slot|diagnostic|compat|test|更新
        """
        B, T = 2, 128
        x = torch.randn(B, T, self.dim).to(self.device)
        
        _, S_next, _, _ = self.layer(x)
        V_proxy = self.layer.last_V_orth

        self.assertIsInstance(S_next, MHDSRA2State)
        self.assertEqual(V_proxy.shape, (B, self.K, self.dim))
        self.assertTrue(torch.isfinite(V_proxy).all())
        self.assertTrue(torch.isfinite(S_next.slot_v).all())

    def test_gradient_flow_and_stability(self):
        B, T = 2, 64
        # Create tensors directly on device with requires_grad=True to make them leaf nodes
        x1 = torch.randn(B, T, self.dim, device=self.device, requires_grad=True)
        x2 = torch.randn(B, T, self.dim, device=self.device, requires_grad=True)
        
        # Chunk 1
        out1, S_next1, kv1, _ = self.layer(x1)
        # Chunk 2
        out2, S_next2, kv2, _ = self.layer(x2, S_prev=S_next1, bypass_kv=kv1)
        
        # Dummy loss
        loss = out2.sum()
        loss.backward()
        
        # Check gradients
        self.assertIsNotNone(x1.grad, "Gradient did not flow back to x1")
        self.assertIsNotNone(x2.grad, "Gradient did not flow back to x2")
        
        # Check for NaNs
        self.assertFalse(torch.isnan(x1.grad).any(), "NaN found in x1.grad")
        self.assertFalse(torch.isnan(x2.grad).any(), "NaN found in x2.grad")
        
        # Ensure S_init got gradients
        self.assertIsNotNone(self.layer.S_init.grad)
        self.assertFalse(torch.isnan(self.layer.S_init.grad).any())

    def test_extreme_k_and_kr(self):
        """Validate extreme slot counts under the MHDSRA2 compatibility layer.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `DSRA_Chunk_Layer.forward`
        - 作用 / Purpose: 确认很小和很大的 slot/top-k 配置仍能生成有效 MHDSRA2 状态
        - 变量 / Variables: `layer_small/layer_large` 是不同容量的兼容层
        - 接入 / Integration: 保护 `K/kr` 参数继续从旧入口映射到新核心
        - 错误处理 / Error handling: NaN 或状态槽位数错误会触发断言失败
        - 关键词 / Keywords:
          extreme|slots|topk|mhdsra2|compat|state|finite|K|kr|边界
        """
        B, T = 2, 32
        x = torch.randn(B, T, self.dim).to(self.device)
        
        # Extreme small K
        layer_small = DSRA_Chunk_Layer(self.dim, K=2, kr=1).to(self.device)
        out, S_next, _, _ = layer_small(x)
        self.assertEqual(S_next.slot_v.shape[2], 2)
        self.assertFalse(torch.isnan(out).any())
        
        # Extreme large K
        layer_large = DSRA_Chunk_Layer(self.dim, K=1024, kr=1024).to(self.device)
        out, S_next, _, _ = layer_large(x)
        self.assertEqual(S_next.slot_v.shape[2], 1024)
        self.assertFalse(torch.isnan(out).any())

    def test_extreme_chunk_size(self):
        B = 2
        # Extremely small chunk size (T < K)
        T_small = 4
        x_small = torch.randn(B, T_small, self.dim).to(self.device)
        out, _, _, _ = self.layer(x_small)
        self.assertEqual(out.shape, (B, T_small, self.dim))
        
        # T=1
        T_one = 1
        x_one = torch.randn(B, T_one, self.dim).to(self.device)
        out, _, _, _ = self.layer(x_one)
        self.assertEqual(out.shape, (B, T_one, self.dim))

    def test_repeated_input_keeps_mhdsra2_state_finite(self):
        """Validate repeated inputs keep the replacement MHDSRA2 state stable.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `DSRA_Chunk_Layer.forward`
        - 作用 / Purpose: 替代旧 zero-novelty 正交投影断言，确认重复信息不会破坏新状态
        - 变量 / Variables: `x` 是重复输入, `S_next` 是第一次状态, `S_next_2` 是第二次状态
        - 接入 / Integration: 保护 MHDSRA2 写入、遗忘和局部缓存组合的数值稳定性
        - 错误处理 / Error handling: NaN/Inf 会触发断言失败
        - 关键词 / Keywords:
          repeated|input|state|finite|mhdsra2|novelty|compat|stability|test|稳定
        """
        B, T = 2, 64
        x = torch.randn(B, T, self.dim).to(self.device)

        _, S_next, bypass_kv, _ = self.layer(x)
        out, S_next_2, _, _ = self.layer(x, S_prev=S_next, bypass_kv=bypass_kv)

        self.assertFalse(torch.isnan(out).any())
        self.assertTrue(torch.isfinite(S_next_2.slot_k).all())
        self.assertTrue(torch.isfinite(S_next_2.slot_v).all())

    def test_pe_modes(self):
        B, T = 2, 32
        x = torch.randn(B, T, self.dim).to(self.device)
        
        for pe_mode in ['none', 'rope', 'alibi', 'timestamps']:
            layer_pe = DSRA_Chunk_Layer(self.dim, K=16, kr=4, pe_mode=pe_mode).to(self.device)
            # Test forward without previous state
            out1, S_next1, kv1, S_time_next1 = layer_pe(x)
            self.assertEqual(out1.shape, (B, T, self.dim))
            self.assertFalse(torch.isnan(out1).any())
            
            # Test forward with previous state
            out2, S_next2, kv2, S_time_next2 = layer_pe(x, S_prev=S_next1, bypass_kv=kv1, S_time_prev=S_time_next1, chunk_idx=1)
            self.assertEqual(out2.shape, (B, T, self.dim))
            self.assertFalse(torch.isnan(out2).any())

    def test_alibi_mask_shape(self):
        from dsra_layer import get_alibi_mask
        seq_len_q, seq_len_k = 16, 32
        mask = get_alibi_mask(seq_len_q, seq_len_k, is_causal=False, device=self.device, dtype=torch.float32)
        # Expected shape: [1, 1, 16, 32]
        self.assertEqual(mask.shape, (1, 1, seq_len_q, seq_len_k))
        
    def test_timestamps_shape(self):
        B, T = 2, 16
        x = torch.randn(B, T, self.dim).to(self.device)
        layer_pe = DSRA_Chunk_Layer(self.dim, K=8, kr=4, pe_mode='timestamps').to(self.device)
        
        # Manually provide S_time_prev to check shape tracking
        S_time_prev = torch.randn(B, 8).to(self.device)
        out, S_next, _, S_time_next = layer_pe(x, S_time_prev=S_time_prev, chunk_idx=1)
        
        self.assertEqual(S_time_next.shape, (B, 8))
        self.assertFalse(torch.isnan(S_time_next).any())

if __name__ == '__main__':
    unittest.main()
