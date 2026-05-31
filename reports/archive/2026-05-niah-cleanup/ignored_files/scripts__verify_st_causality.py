import sys
sys.path.insert(0, r'e:\Project\python\DSRA')

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(42)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

from src.dsra.mhdsra2.improved_dsra_mha import MultiHeadDSRA2, MHDSRA2Config

class HybridLanguageModel(nn.Module):
    def __init__(self, vocab_size, dim=256, n_layers=2, n_heads=4,
                 slots=64, local_window=128, chunk_size=128):
        super().__init__()
        self.chunk_size = chunk_size
        self.dim = dim
        self.n_layers = n_layers
        self._causal_mask_cache = {}

        self.tok_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Embedding(chunk_size, dim)

        mh_cfg = MHDSRA2Config(
            dim=dim, heads=n_heads,
            local_window=local_window,
            slot_pe="rope", slots=slots,
            tau_init=8.0, tau_write_init=4.0,
            read_topk=8, write_topk=4,
            use_retrieval=False,
            forget_base=0.001,
            usage_decay=0.995,
            conf_decay=0.999,
        )
        self.mh_layers = nn.ModuleList([
            MultiHeadDSRA2(mh_cfg) for _ in range(n_layers)
        ])

        self.st_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=n_heads, dim_feedforward=dim*4,
                batch_first=True, activation='gelu'
            ) for _ in range(n_layers)
        ])
        self.st_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(n_layers)
        ])

        self.fuse_gates = nn.ModuleList([
            nn.Linear(dim * 2, 2) for _ in range(n_layers)
        ])
        for gate in self.fuse_gates:
            nn.init.zeros_(gate.bias)
            nn.init.normal_(gate.weight, mean=0.0, std=0.01)

        self.norm = nn.LayerNorm(dim)
        self.to(device)

    def _init_states(self, batch_size, device, dtype):
        return [layer.init_state(batch_size, device=device, dtype=dtype)
                for layer in self.mh_layers]

    def _get_causal_mask(self, seq_len, device):
        key = (seq_len, device)
        if key not in self._causal_mask_cache:
            self._causal_mask_cache[key] = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=device
            )
        return self._causal_mask_cache[key]

    def forward(self, x, states=None, return_gate_info=False):
        seq_len = x.shape[1]
        bsz = x.shape[0]

        if states is None:
            states = self._init_states(bsz, x.device, x.dtype)

        all_h = []
        gate_info = {} if return_gate_info else None
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            chunk = x[:, start:end]
            chunk_len = end - start

            positions = torch.arange(chunk_len, device=x.device)
            h = self.tok_embedding(chunk) + self.pos_embedding(positions)

            for i, (mh_layer, st_layer, st_proj, fuse_gate) in enumerate(
                zip(self.mh_layers, self.st_layers, self.st_projs, self.fuse_gates)
            ):
                causal_mask = self._get_causal_mask(chunk_len, x.device)
                h_st = st_layer(h, src_mask=causal_mask, is_causal=True)
                h_st = st_proj(h_st)

                mh_result = mh_layer(h, state=states[i], return_aux=True)
                if len(mh_result) == 3:
                    h_mh, states[i], _aux = mh_result
                else:
                    h_mh, states[i] = mh_result

                gate_input = torch.cat([h_st, h_mh], dim=-1)
                gate_logits = fuse_gate(gate_input)
                gate_weights = F.softmax(gate_logits, dim=-1)

                if return_gate_info:
                    with torch.no_grad():
                        gate_info[f"layer{i}_st_weight"] = gate_weights[..., 0].mean().item()
                        gate_info[f"layer{i}_mh_weight"] = gate_weights[..., 1].mean().item()

                h = gate_weights[..., 0:1] * h_st + gate_weights[..., 1:2] * h_mh

            all_h.append(h)

        h = torch.cat(all_h, dim=1)
        h = self.norm(h)
        logits = F.linear(h, self.tok_embedding.weight)

        if return_gate_info:
            return logits, states, gate_info
        return logits, states

vocab_size = 1000
model = HybridLanguageModel(vocab_size=vocab_size, dim=256, n_layers=2, n_heads=4,
                            slots=64, local_window=128, chunk_size=128)
model.eval()

# ===== 测试1：修改未来token不影响当前位置输出 =====
print("\n===== 测试1：因果性验证 - 修改未来token不应影响当前位置输出 =====")
seq_len = 128
x1 = torch.randint(0, vocab_size, (2, seq_len), device=device)
x2 = x1.clone()
# 修改最后10个token（位置118-127）
x2[:, -10:] = (x1[:, -10:] + 1) % vocab_size

with torch.no_grad():
    logits1, _ = model(x1)
    logits2, _ = model(x2)

# 只检查未被修改的位置（0, 50, 117），这些位置的输入完全相同
# 因果掩码下，这些位置不应看到位置118+的任何信息
diff_pos0 = (logits1[:, 0, :] - logits2[:, 0, :]).abs().max().item()
diff_pos50 = (logits1[:, 50, :] - logits2[:, 50, :]).abs().max().item()
diff_pos117 = (logits1[:, 117, :] - logits2[:, 117, :]).abs().max().item()
# 位置127本身被修改了，其输出变化是正常的（自身输入变了），不用于因果性判断
diff_pos127 = (logits1[:, 127, :] - logits2[:, 127, :]).abs().max().item()

print(f"  修改最后10个token（位置118-127）后:")
print(f"  位置0的logits最大差异:   {diff_pos0:.8f} (应≈0，输入未变且看不到未来)")
print(f"  位置50的logits最大差异:  {diff_pos50:.8f} (应≈0，输入未变且看不到未来)")
print(f"  位置117的logits最大差异: {diff_pos117:.8f} (应≈0，输入未变且看不到位置118+)")
print(f"  位置127的logits最大差异: {diff_pos127:.8f} (自身输入被修改，差异正常)")

# 因果性判断：只检查输入未变的位置
causal_pass = diff_pos0 < 0.01 and diff_pos50 < 0.01 and diff_pos117 < 0.01
causal_status = "✅ 通过" if causal_pass else "❌ 失败"
print(f"\n  因果性测试: {causal_status}")
if causal_pass:
    print("  → 所有未修改位置的输出完全不变，确认无未来信息泄露")

# ===== 测试1b：更严格的因果性验证 =====
print("\n===== 测试1b：严格因果性验证 - 修改位置k只影响k及之后 =====")
x5 = torch.randint(0, vocab_size, (1, seq_len), device=device)
x6 = x5.clone()
# 只修改位置60
x6[0, 60] = (x5[0, 60] + 1) % vocab_size

with torch.no_grad():
    logits5, _ = model(x5)
    logits6, _ = model(x6)

# 位置0-59不应受影响（输入完全相同，且因果掩码下看不到位置60）
diff_strict_before = (logits5[0, :60, :] - logits6[0, :60, :]).abs().max().item()
# 位置60+应受影响（位置60的输入变了，因果掩码下后续位置能看到位置60）
diff_strict_at = (logits5[0, 60, :] - logits6[0, 60, :]).abs().max().item()
diff_strict_after = (logits5[0, 61:, :] - logits6[0, 61:, :]).abs().max().item()

print(f"  只修改位置60后:")
print(f"  位置0-59的logits最大差异:  {diff_strict_before:.8f} (应≈0)")
print(f"  位置60的logits最大差异:    {diff_strict_at:.8f} (应>0，自身输入变了)")
print(f"  位置61+的logits最大差异:   {diff_strict_after:.8f} (应>0，能看到位置60)")

strict_pass = diff_strict_before < 0.01 and diff_strict_at > 0.001 and diff_strict_after > 0.001
strict_status = "✅ 通过" if strict_pass else "❌ 失败"
print(f"\n  严格因果性测试: {strict_status}")

# ===== 测试2：对比修复前后的输出差异 =====
print("\n===== 测试2：因果掩码 vs 双向注意力 - 输出差异分析 =====")
y = torch.cat([x1[:, 1:], torch.zeros(2, 1, dtype=torch.long, device=device)], dim=1)

with torch.no_grad():
    # 修复后（因果掩码）
    logits_fixed, _ = model(x1)
    loss_fixed = F.cross_entropy(logits_fixed.reshape(-1, vocab_size), y.reshape(-1))

    # 模拟修复前（无因果掩码）
    h = model.tok_embedding(x1) + model.pos_embedding(torch.arange(seq_len, device=device))
    for i, (mh_layer, st_layer, st_proj, fuse_gate) in enumerate(
        zip(model.mh_layers, model.st_layers, model.st_projs, model.fuse_gates)
    ):
        h_st_bi = st_layer(h)  # 无因果掩码
        h_st_bi = st_proj(h_st_bi)

        states_i = model._init_states(2, device, torch.float32)
        mh_result = mh_layer(h, state=states_i[i], return_aux=True)
        if len(mh_result) == 3:
            h_mh_bi, _, _ = mh_result
        else:
            h_mh_bi, _ = mh_result

        gate_input = torch.cat([h_st_bi, h_mh_bi], dim=-1)
        gate_logits = fuse_gate(gate_input)
        gate_weights = F.softmax(gate_logits, dim=-1)
        h = gate_weights[..., 0:1] * h_st_bi + gate_weights[..., 1:2] * h_mh_bi

    h = model.norm(h)
    logits_bi = F.linear(h, model.tok_embedding.weight)
    loss_bi = F.cross_entropy(logits_bi.reshape(-1, vocab_size), y.reshape(-1))

print(f"  修复前（双向注意力）Loss: {loss_bi.item():.3f}, PPL: {math.exp(loss_bi.item()):.1f}")
print(f"  修复后（因果注意力）Loss: {loss_fixed.item():.3f}, PPL: {math.exp(loss_fixed.item()):.1f}")
print(f"  Loss差异: {loss_fixed.item() - loss_bi.item():.3f}")

# 关键验证：双向注意力下，修改未来token应该影响当前位置
x_bi1 = x1.clone()
x_bi2 = x1.clone()
x_bi2[:, -10:] = (x_bi1[:, -10:] + 1) % vocab_size

with torch.no_grad():
    # 因果掩码下
    logits_c1, _ = model(x_bi1)
    logits_c2, _ = model(x_bi2)
    causal_diff = (logits_c1[:, 0, :] - logits_c2[:, 0, :]).abs().max().item()

    # 双向注意力下
    def forward_bidirectional(model, x_input):
        h = model.tok_embedding(x_input) + model.pos_embedding(
            torch.arange(x_input.shape[1], device=x_input.device))
        for i, (mh_layer, st_layer, st_proj, fuse_gate) in enumerate(
            zip(model.mh_layers, model.st_layers, model.st_projs, model.fuse_gates)
        ):
            h_st_bi = st_layer(h)
            h_st_bi = st_proj(h_st_bi)
            states_i = model._init_states(x_input.shape[0], x_input.device, torch.float32)
            mh_result = mh_layer(h, state=states_i[i], return_aux=True)
            if len(mh_result) == 3:
                h_mh_bi, _, _ = mh_result
            else:
                h_mh_bi, _ = mh_result
            gate_input = torch.cat([h_st_bi, h_mh_bi], dim=-1)
            gate_logits = fuse_gate(gate_input)
            gate_weights = F.softmax(gate_logits, dim=-1)
            h = gate_weights[..., 0:1] * h_st_bi + gate_weights[..., 1:2] * h_mh_bi
        h = model.norm(h)
        return F.linear(h, model.tok_embedding.weight)

    logits_b1 = forward_bidirectional(model, x_bi1)
    logits_b2 = forward_bidirectional(model, x_bi2)
    bidirectional_diff = (logits_b1[:, 0, :] - logits_b2[:, 0, :]).abs().max().item()

print(f"\n  修改最后10个token后，位置0的输出差异:")
print(f"    因果掩码模式:     {causal_diff:.8f} (应≈0)")
print(f"    双向注意力模式:   {bidirectional_diff:.8f} (应>0，存在信息泄露)")

# 验证：因果掩码下位置0不变，双向注意力下位置0会变
leakage_pass = causal_diff < 0.01 and bidirectional_diff > 0.001
leakage_status = "✅ 通过" if leakage_pass else "❌ 失败"
print(f"\n  信息泄露对比测试: {leakage_status}")
if leakage_pass:
    print("  → 因果掩码成功阻止了未来信息泄露，双向注意力则存在泄露")

# ===== 测试3：修改单个token只影响其后续位置 =====
print("\n===== 测试3：局部影响验证 - 修改位置k只应影响位置k及之后 =====")
x3 = torch.randint(0, vocab_size, (1, seq_len), device=device)
x4 = x3.clone()
x4[0, 50] = (x3[0, 50] + 1) % vocab_size

with torch.no_grad():
    logits3, _ = model(x3)
    logits4, _ = model(x4)

diff_before = (logits3[0, :50, :] - logits4[0, :50, :]).abs().max().item()
diff_at = (logits3[0, 50, :] - logits4[0, 50, :]).abs().max().item()
diff_after = (logits3[0, 51:, :] - logits4[0, 51:, :]).abs().max().item()

print(f"  修改位置50后:")
print(f"  位置0-49的logits最大差异: {diff_before:.8f} (应≈0)")
print(f"  位置50的logits最大差异:   {diff_at:.8f} (应>0，自身输入变了)")
print(f"  位置51+的logits最大差异:  {diff_after:.8f} (应>0，能看到位置50)")

local_pass = diff_before < 0.01 and diff_at > 0.001 and diff_after > 0.001
local_status = "✅ 通过" if local_pass else "❌ 失败"
print(f"\n  局部影响测试: {local_status}")

# ===== 总结 =====
print("\n" + "="*60)
all_pass = causal_pass and strict_pass and leakage_pass and local_pass
if all_pass:
    print("✅ 所有验证通过！修复后的ST分支不再泄露未来信息")
    print("\n验证要点总结:")
    print("  1. 修改未来token不影响任何未修改位置的输出（因果掩码有效）")
    print("  2. 修改位置k只影响位置k及其后续位置（严格因果性）")
    print("  3. 因果掩码模式下无信息泄露，双向注意力模式下存在泄露（对比验证）")
    print("  4. 局部修改的传播范围符合因果性约束")
else:
    print("❌ 存在验证失败项，需要进一步检查")
    if not causal_pass:
        print("  - 因果性测试失败：修改未来token仍影响当前位置")
    if not strict_pass:
        print("  - 严格因果性测试失败：修改位置k影响了位置k之前")
    if not leakage_pass:
        print("  - 信息泄露对比测试失败")
    if not local_pass:
        print("  - 局部影响测试失败：修改位置k影响了位置k之前")
