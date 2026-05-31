"""
验证假设3：read_topk=8 导致 probs 分布过于均匀（entropy≈0.95），
slot_out = weighted_avg(slot_v) 的 std 被压缩到只有 hard read 的 40%

对比四种配置：read_topk=8, read_topk=2, read_topk=1, hard_read=True
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ckpt_path = PROJECT_ROOT / "models" / "hybrid_lm" / "best_model.pt"
ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
config = ckpt['config']
vocab_size = ckpt['vocab_size']

print(f"设备: {DEVICE}")
print(f"检查点配置: {config}")
print(f"词汇表大小: {vocab_size}")
print()

from scripts.pretrain_hybrid_lm import HybridLanguageModel

model = HybridLanguageModel(
    vocab_size=vocab_size, dim=config['dim'], n_layers=config['n_layers'],
    n_heads=config['n_heads'], slots=config['slots'],
    chunk_size=config.get('seq_len', 512)
).to(DEVICE)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()

print(f"模型训练时的 MHDSRA2 配置:")
print(f"  read_topk={model.mh_layers[0].cfg.read_topk}")
print(f"  hard_read={model.mh_layers[0].cfg.hard_read}")
print(f"  slots={model.mh_layers[0].cfg.slots}")
print(f"  dim={model.mh_layers[0].cfg.dim}")
print(f"  heads={model.mh_layers[0].cfg.heads}")
print()

batch_size = 4
seq_len = config.get('seq_len', 512)
torch.manual_seed(42)
x_tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=DEVICE)


def compute_normalized_entropy(probs, eps=1e-8):
    """计算归一化熵：H / H_max，其中 H_max = ln(K)"""
    K = probs.shape[-1]
    if K <= 1:
        return 0.0
    entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(float(K), device=probs.device))
    norm_entropy = entropy / max_entropy
    return norm_entropy.mean().item()


def run_slot_read_with_config(mh_layer, q_heads, state, read_topk, hard_read):
    """临时修改配置并执行 _slot_read，返回 (slot_out, read_aux)"""
    orig_topk = mh_layer.cfg.read_topk
    orig_hard = mh_layer.cfg.hard_read

    mh_layer.cfg.read_topk = read_topk
    mh_layer.cfg.hard_read = hard_read

    slot_out, read_aux = mh_layer._slot_read(q_heads, state)

    mh_layer.cfg.read_topk = orig_topk
    mh_layer.cfg.hard_read = hard_read if orig_hard else orig_hard
    mh_layer.cfg.hard_read = orig_hard

    return slot_out, read_aux


configs = [
    ("read_topk=8, hard_read=False", 8, False),
    ("read_topk=2, hard_read=False", 2, False),
    ("read_topk=1, hard_read=False", 1, False),
    ("hard_read=True (topk=1)",      1, True),
]

print("=" * 100)
print("假设3验证：read_topk 对 probs 分布和 slot_out 幅度的影响")
print("=" * 100)
print()

with torch.no_grad():
    positions = torch.arange(seq_len, device=DEVICE)
    h = model.tok_embedding(x_tokens) + model.pos_embedding(positions)

    for i in range(model.n_layers):
        mh = model.mh_layers[i]
        states_i = mh.init_state(batch_size, device=DEVICE, dtype=torch.float32)

        q, k, v = mh.qkv(h).chunk(3, dim=-1)
        q_heads = mh._to_heads(q)
        k_heads = mh._to_heads(k)
        v_heads = mh._to_heads(v)

        print(f"{'='*100}")
        print(f"Layer {i}  |  q_heads std={q_heads.std().item():.6f}  |  slot_v std={states_i.slot_v.std().item():.6f}")
        print(f"{'='*100}")
        print(f"{'配置':<30} {'Norm Entropy':>14} {'slot_out std':>14} {'slot_out_proj std':>18} {'vs hard_read std比':>20}")
        print(f"{'-'*30} {'-'*14} {'-'*14} {'-'*18} {'-'*20}")

        hard_read_std = None
        results = []

        for cfg_name, topk, hard in configs:
            slot_out, read_aux = run_slot_read_with_config(mh, q_heads, states_i, topk, hard)

            probs = read_aux["read_probs"]
            norm_entropy = compute_normalized_entropy(probs)

            slot_out_std = slot_out.std().item()
            slot_out_full = mh._from_heads(slot_out)
            slot_out_proj = mh.out_proj(slot_out_full)
            slot_out_proj_std = slot_out_proj.std().item()

            results.append((cfg_name, norm_entropy, slot_out_std, slot_out_proj_std))

            if hard:
                hard_read_std = slot_out_std
                hard_read_proj_std = slot_out_proj_std

        for cfg_name, norm_entropy, slot_out_std, slot_out_proj_std in results:
            if hard_read_std is not None and slot_out_std > 0:
                ratio = slot_out_std / hard_read_std
                proj_ratio = slot_out_proj_std / hard_read_proj_std
                ratio_str = f"{ratio:.4f} (proj: {proj_ratio:.4f})"
            else:
                ratio_str = "1.0000 (baseline)"
            print(f"{cfg_name:<30} {norm_entropy:>14.6f} {slot_out_std:>14.6f} {slot_out_proj_std:>18.6f} {ratio_str:>20}")

        # ST 分支对比
        causal_mask = model._get_causal_mask(seq_len, DEVICE)
        h_st = model.st_layers[i](h, src_mask=causal_mask, is_causal=True)
        h_st = model.st_projs[i](h_st)
        st_std = h_st.std().item()

        print(f"")
        print(f"  ST 分支 std: {st_std:.6f}")
        for cfg_name, norm_entropy, slot_out_std, slot_out_proj_std in results:
            mh_st_ratio = slot_out_proj_std / st_std
            print(f"  {cfg_name}: slot_out_proj / ST = {mh_st_ratio:.4f}")
        print()

        # 详细分析 read_topk=8 的 probs 分布
        print(f"  --- read_topk=8 详细 probs 分布 (Layer {i}, batch=0, head=0, token=0) ---")
        mh_layer_cfg_bak_topk = mh.cfg.read_topk
        mh_layer_cfg_bak_hard = mh.cfg.hard_read
        mh.cfg.read_topk = 8
        mh.cfg.hard_read = False
        _, aux8 = mh._slot_read(q_heads, states_i)
        mh.cfg.read_topk = mh_layer_cfg_bak_topk
        mh.cfg.hard_read = mh_layer_cfg_bak_hard

        probs8 = aux8["read_probs"][0, 0, 0]
        print(f"  probs (top-8): {probs8.cpu().numpy()}")
        print(f"  max prob: {probs8.max().item():.6f}, min prob: {probs8.min().item():.6f}")
        print(f"  top-1 mass: {probs8[0].item():.6f}, top-2 mass: {(probs8[0]+probs8[1]).item():.6f}")

        # hard_read 详细 probs
        mh.cfg.read_topk = 1
        mh.cfg.hard_read = True
        _, aux_hard = mh._slot_read(q_heads, states_i)
        mh.cfg.read_topk = mh_layer_cfg_bak_topk
        mh.cfg.hard_read = mh_layer_cfg_bak_hard

        probs_hard = aux_hard["read_probs"][0, 0, 0]
        print(f"  hard_read probs: {probs_hard.cpu().numpy()}")
        print()

        # slot_v 的多样性：如果 slot_v 本身就很相似，加权平均不会大幅压缩 std
        slot_v = states_i.slot_v
        slot_v_per_head = slot_v[0, 0]
        pairwise_cos = F.cosine_similarity(
            slot_v_per_head.unsqueeze(1), slot_v_per_head.unsqueeze(0), dim=-1
        )
        off_diag = pairwise_cos[~torch.eye(pairwise_cos.size(0), dtype=bool, device=DEVICE)]
        print(f"  slot_v pairwise cosine sim: mean={off_diag.mean().item():.6f}, std={off_diag.std().item():.6f}")
        print(f"  slot_v norm: mean={slot_v.norm(dim=-1).mean().item():.6f}, std={slot_v.norm(dim=-1).std().item():.6f}")
        print()

        # 逐步更新 h 用于下一层
        mh_result = mh(h, state=states_i, return_aux=True)
        if len(mh_result) == 3:
            h_mh, _, _ = mh_result
        else:
            h_mh, _ = mh_result

        h_st_raw = model.st_layers[i](h, src_mask=causal_mask, is_causal=True)
        h_st_proj = model.st_projs[i](h_st_raw)

        fuse_input = torch.cat([h_mh, h_st_proj], dim=-1)
        gate_logits = model.fuse_gates[i](fuse_input)
        gate_weights = torch.softmax(gate_logits, dim=-1)
        mh_w = gate_weights[:, :, 0:1]
        st_w = gate_weights[:, :, 1:2]

        h = mh_w * model.mh_out_norms[i](h_mh) * model.mh_scales[i] + st_w * h_st_proj

print()
print("=" * 100)
print("总结与结论")
print("=" * 100)
print()
print("假设3：read_topk=8 导致 probs 分布过于均匀（entropy≈0.95），")
print("       slot_out = weighted_avg(slot_v) 的 std 被压缩到只有 hard read 的 40%")
print()
print("验证要点：")
print("1. read_topk=8 时 normalized entropy 是否接近 0.95？")
print("2. read_topk=8 时 slot_out std 是否约为 hard_read 时的 40%？")
print("3. read_topk 减小时 entropy 和 std 如何变化？")
print("4. slot_out_proj 经过 out_proj 后，幅度差异是否仍然存在？")
