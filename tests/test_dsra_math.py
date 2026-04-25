import unittest
import torch
import torch.nn as nn
from dsra_layer import DSRA_Chunk_Layer

class TestDSRAMath(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dim = 64
        self.K = 16
        self.kr = 4
        self.layer = DSRA_Chunk_Layer(self.dim, K=self.K, kr=self.kr).to(self.device)

    def test_forward_dimensions(self):
        B, T = 2, 128
        x = torch.randn(B, T, self.dim).to(self.device)
        
        out, S_next, bypass_kv, _ = self.layer(x)
        
        self.assertEqual(out.shape, (B, T, self.dim))
        self.assertEqual(S_next.shape, (B, self.K, self.dim))
        self.assertEqual(bypass_kv[0].shape, (B, T, self.dim))
        self.assertEqual(bypass_kv[1].shape, (B, T, self.dim))

    def test_orthogonality(self):
        B, T = 2, 128
        x = torch.randn(B, T, self.dim).to(self.device)
        
        out, S_next, _, _ = self.layer(x)
        V_orth = self.layer.last_V_orth # [B, K, dim]
        S_prev = self.layer.S_init.unsqueeze(0).expand(B, -1, -1) # [B, K, dim]
        
        # Test if V_orth is orthogonal to S_prev
        # V_orth @ S_prev.T should be close to 0
        dot_product = torch.bmm(V_orth, S_prev.transpose(1, 2)) # [B, K, K]
        
        # Mean absolute dot product should be very small
        mean_abs_dot = dot_product.abs().mean().item()
        self.assertTrue(mean_abs_dot < 1e-4, f"Orthogonality failed, mean abs dot product: {mean_abs_dot}")

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
        B, T = 2, 32
        x = torch.randn(B, T, self.dim).to(self.device)
        
        # Extreme small K
        layer_small = DSRA_Chunk_Layer(self.dim, K=2, kr=1).to(self.device)
        out, S_next, _, _ = layer_small(x)
        self.assertEqual(S_next.shape, (B, 2, self.dim))
        self.assertFalse(torch.isnan(out).any())
        
        # Extreme large K
        layer_large = DSRA_Chunk_Layer(self.dim, K=1024, kr=1024).to(self.device)
        out, S_next, _, _ = layer_large(x)
        self.assertEqual(S_next.shape, (B, 1024, self.dim))
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

    def test_zero_novelty(self):
        B, T = 2, 64
        x = torch.randn(B, T, self.dim).to(self.device)
        
        # Overwrite W_v so V is exactly equal to the first T elements of S_init
        # We need to simulate the case where V is identical to existing S
        # A simple way is to make V composed entirely of the first row of S
        const_vec = self.layer.S_init[0].unsqueeze(0).unsqueeze(0).expand(B, T, -1) # [B, T, dim]
        
        # Override forward to pass this const_vec as V
        class MockWV(nn.Module):
            def __init__(self, const_vec):
                super().__init__()
                self.const_vec = const_vec
            def forward(self, x):
                return self.const_vec

        original_wv = self.layer.W_v
        self.layer.W_v = MockWV(const_vec)
        
        out, S_next, _, _ = self.layer(x)
        
        # Restore W_v
        self.layer.W_v = original_wv
        
        # If V is just a vector from S_prev's span, it should be entirely projected away
        V_orth = self.layer.last_V_orth
        self.assertTrue(V_orth.abs().mean().item() < 1e-4, f"V_orth should be close to 0 when V is in span of S, got {V_orth.abs().mean().item()}")

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
