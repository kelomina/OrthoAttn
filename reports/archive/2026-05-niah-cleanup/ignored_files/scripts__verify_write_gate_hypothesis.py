"""验证假设1和假设2：write_gate太小导致slot_v几乎不更新。

Verify Hypothesis 1: write_gate is too small (~0.03), causing slot_v to barely update.
Verify Hypothesis 2: slot_v stays near initialization (std~0.177), amplitude only 21% of v_heads.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pretrain_hybrid_lm import HybridLanguageModel

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CKPT_PATH = PROJECT_ROOT / "models" / "hybrid_lm" / "best_model.pt"


def main():
    print("=" * 70)
    print("验证假设1: write_gate 太小（约0.03），导致 slot_v 几乎不更新")
    print("验证假设2: slot_v 停留在初始化值附近（std≈0.177），幅度只有 v_heads 的 21%")
    print("=" * 70)
    print(f"设备: {DEVICE}")

    # 1. 加载检查点
    print(f"\n加载检查点: {CKPT_PATH}")
    ckpt = torch.load(str(CKPT_PATH), map_location=DEVICE, weights_only=False)
    config = ckpt.get("config", {})
    vocab_size = ckpt.get("vocab_size", 32000)
    print(f"  配置: {config}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  训练步数: {ckpt.get('step', 'N/A')}")
    print(f"  best_ppl: {ckpt.get('best_ppl', 'N/A')}")

    dim = config.get("dim", 256)
    n_layers = config.get("n_layers", 8)
    n_heads = config.get("n_heads", 8)
    slots = config.get("slots", 256)
    seq_len = config.get("seq_len", 512)

    # 2. 创建模型并加载权重
    print("\n创建模型...")
    model = HybridLanguageModel(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        slots=slots,
        chunk_size=seq_len,
    )
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  缺少键: {len(missing)} 个")
    if unexpected:
        print(f"  多余键: {len(unexpected)} 个")
    model.eval()
    print("  模型加载完成")

    # 3. 逐层验证
    batch_size = 1
    results = []

    print(f"\n{'='*70}")
    print(f"逐层验证 (batch_size={batch_size}, seq_len={seq_len})")
    print(f"{'='*70}")

    # 生成随机输入
    torch.manual_seed(42)
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=DEVICE)

    # 先做一次完整的 forward pass 获取每层的中间表示
    # 但我们需要逐层控制，所以手动逐层执行

    with torch.no_grad():
        # 初始化所有层的状态
        all_states_before = []
        for i, mh_layer in enumerate(model.mh_layers):
            state = mh_layer.init_state(batch_size, device=DEVICE, dtype=torch.float32)
            all_states_before.append(state)

        # 记录 init 后的 slot_v 统计
        init_stats = []
        for i, state in enumerate(all_states_before):
            sv = state.slot_v.float()
            init_stats.append({
                "std": sv.std().item(),
                "mean": sv.mean().item(),
                "abs_mean": sv.abs().mean().item(),
                "min": sv.min().item(),
                "max": sv.max().item(),
            })

        # 逐层执行 forward，记录每层的 slot_v 变化
        # 模拟 HybridLanguageModel 的 forward 逻辑
        positions = torch.arange(seq_len, device=DEVICE)
        h = model.tok_embedding(x) + model.pos_embedding(positions)

        layer_results = []

        for i, (mh_layer, st_layer, st_proj, fuse_gate, mh_out_norm, mh_scale) in enumerate(
            zip(model.mh_layers, model.st_layers, model.st_projs, model.fuse_gates, model.mh_out_norms, model.mh_scales)
        ):
            # 记录 forward 前的 slot_v
            slot_v_before = all_states_before[i].slot_v.float().clone()

            # ST分支
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=DEVICE)
            h_st = st_layer(h, src_mask=causal_mask, is_causal=True)
            h_st = st_proj(h_st)

            # MHDSRA2分支 - 使用 return_aux=True 获取 write_stats
            mh_result = mh_layer(h, state=all_states_before[i], return_aux=True)
            if len(mh_result) == 3:
                h_mh, new_state, aux = mh_result
            else:
                h_mh, new_state = mh_result
                aux = {}

            # 记录 forward 后的 slot_v
            slot_v_after = new_state.slot_v.float().clone()

            # 计算 slot_v 变化量
            delta = (slot_v_after - slot_v_before).abs()
            delta_mean = delta.mean().item()
            slot_v_before_norm = slot_v_before.abs().mean().item()
            change_pct = (delta_mean / max(slot_v_before_norm, 1e-8)) * 100

            # 从 aux 中提取 write_gate_mean
            write_stats = aux.get("write_stats", {}) if aux else {}
            write_gate_mean = write_stats.get("write_gate_mean", None)
            token_gate_mean = write_stats.get("token_gate_mean", None)
            forget_gate_mean = write_stats.get("forget_gate_mean", None)
            novelty_mean = write_stats.get("novelty_mean", None)
            conflict_mean = write_stats.get("conflict_mean", None)
            write_mass_mean = write_stats.get("write_mass_mean", None)

            # 计算 v_heads 的统计
            # v_heads 是当前 chunk 的 v 投影（需要重新计算）
            qkv = mh_layer.qkv(h)
            _, k_heads, v_heads = qkv.chunk(3, dim=-1)
            v_heads = v_heads.reshape(batch_size, seq_len, n_heads, -1).transpose(1, 2)
            v_heads = v_heads.float()

            v_heads_std = v_heads.std().item()
            v_heads_abs_mean = v_heads.abs().mean().item()

            slot_v_after_std = slot_v_after.std().item()
            slot_v_after_abs_mean = slot_v_after.abs().mean().item()

            # slot_v / v_heads 的比值
            ratio_std = slot_v_after_std / max(v_heads_std, 1e-8)
            ratio_abs = slot_v_after_abs_mean / max(v_heads_abs_mean, 1e-8)

            result = {
                "layer": i,
                "write_gate_mean": write_gate_mean,
                "token_gate_mean": token_gate_mean,
                "forget_gate_mean": forget_gate_mean,
                "novelty_mean": novelty_mean,
                "conflict_mean": conflict_mean,
                "write_mass_mean": write_mass_mean,
                "slot_v_std_init": init_stats[i]["std"],
                "slot_v_std_after": slot_v_after_std,
                "slot_v_abs_mean_init": init_stats[i]["abs_mean"],
                "slot_v_abs_mean_after": slot_v_after_abs_mean,
                "delta_mean": delta_mean,
                "slot_v_before_norm": slot_v_before_norm,
                "change_pct": change_pct,
                "v_heads_std": v_heads_std,
                "v_heads_abs_mean": v_heads_abs_mean,
                "ratio_std": ratio_std,
                "ratio_abs": ratio_abs,
            }
            layer_results.append(result)

            # 应用 LayerNorm 和缩放
            h_mh = mh_out_norm(h_mh) * mh_scale

            # 动态融合门控
            gate_input = torch.cat([h_st, h_mh], dim=-1)
            gate_logits = fuse_gate(gate_input)
            gate_weights = torch.softmax(gate_logits, dim=-1)
            h = gate_weights[..., 0:1] * h_st + gate_weights[..., 1:2] * h_mh

            # 更新状态
            all_states_before[i] = new_state

    # 4. 输出结果
    print(f"\n{'='*70}")
    print("验证结果")
    print(f"{'='*70}")

    # 表1: write_gate 相关
    print(f"\n--- 假设1: write_gate 是否太小（约0.03）---")
    print(f"{'Layer':<8} {'write_gate_mean':<20} {'token_gate_mean':<20} {'forget_gate_mean':<20} {'novelty_mean':<18} {'conflict_mean':<18} {'write_mass_mean':<20}")
    for r in layer_results:
        wg = f"{r['write_gate_mean']:.6f}" if r['write_gate_mean'] is not None else "N/A"
        tg = f"{r['token_gate_mean']:.6f}" if r['token_gate_mean'] is not None else "N/A"
        fg = f"{r['forget_gate_mean']:.6f}" if r['forget_gate_mean'] is not None else "N/A"
        nv = f"{r['novelty_mean']:.6f}" if r['novelty_mean'] is not None else "N/A"
        cf = f"{r['conflict_mean']:.6f}" if r['conflict_mean'] is not None else "N/A"
        wm = f"{r['write_mass_mean']:.6f}" if r['write_mass_mean'] is not None else "N/A"
        print(f"{r['layer']:<8} {wg:<20} {tg:<20} {fg:<20} {nv:<18} {cf:<18} {wm:<20}")

    # 表2: slot_v 变化
    print(f"\n--- slot_v 变化量 ---")
    print(f"{'Layer':<8} {'std_init':<14} {'std_after':<14} {'delta_mean':<14} {'before_norm':<14} {'change_pct':<14}")
    for r in layer_results:
        print(f"{r['layer']:<8} {r['slot_v_std_init']:<14.6f} {r['slot_v_std_after']:<14.6f} {r['delta_mean']:<14.6f} {r['slot_v_before_norm']:<14.6f} {r['change_pct']:<14.2f}%")

    # 表3: slot_v vs v_heads 比值
    print(f"\n--- 假设2: slot_v / v_heads 比值 ---")
    print(f"{'Layer':<8} {'slot_v_std':<14} {'v_heads_std':<14} {'ratio_std':<14} {'slot_v_abs':<14} {'v_heads_abs':<14} {'ratio_abs':<14}")
    for r in layer_results:
        print(f"{r['layer']:<8} {r['slot_v_std_after']:<14.6f} {r['v_heads_std']:<14.6f} {r['ratio_std']:<14.4f} {r['slot_v_abs_mean_after']:<14.6f} {r['v_heads_abs_mean']:<14.6f} {r['ratio_abs']:<14.4f}")

    # 5. 汇总统计
    print(f"\n{'='*70}")
    print("汇总统计")
    print(f"{'='*70}")

    write_gates = [r['write_gate_mean'] for r in layer_results if r['write_gate_mean'] is not None]
    token_gates = [r['token_gate_mean'] for r in layer_results if r['token_gate_mean'] is not None]
    change_pcts = [r['change_pct'] for r in layer_results]
    ratio_stds = [r['ratio_std'] for r in layer_results]
    ratio_abss = [r['ratio_abs'] for r in layer_results]

    if write_gates:
        avg_wg = sum(write_gates) / len(write_gates)
        print(f"  write_gate_mean 平均值: {avg_wg:.6f}")
        print(f"  write_gate_mean 范围: [{min(write_gates):.6f}, {max(write_gates):.6f}]")
    if token_gates:
        avg_tg = sum(token_gates) / len(token_gates)
        print(f"  token_gate_mean 平均值: {avg_tg:.6f}")
    avg_change = sum(change_pcts) / len(change_pcts)
    print(f"  slot_v 变化百分比 平均值: {avg_change:.2f}%")
    print(f"  slot_v 变化百分比 范围: [{min(change_pcts):.2f}%, {max(change_pcts):.2f}%]")
    avg_ratio_std = sum(ratio_stds) / len(ratio_stds)
    avg_ratio_abs = sum(ratio_abss) / len(ratio_abss)
    print(f"  slot_v/v_heads std 比值 平均值: {avg_ratio_std:.4f}")
    print(f"  slot_v/v_heads abs 比值 平均值: {avg_ratio_abs:.4f}")

    # 6. 结论
    print(f"\n{'='*70}")
    print("结论")
    print(f"{'='*70}")

    # 假设1: write_gate ~0.03
    if write_gates:
        avg_wg = sum(write_gates) / len(write_gates)
        if avg_wg < 0.1:
            print(f"  ✅ 假设1 成立: write_gate_mean 平均值 = {avg_wg:.6f}，确实很小")
            if avg_wg < 0.05:
                print(f"     且非常接近假设值 0.03（实际 {avg_wg:.4f}）")
        else:
            print(f"  ❌ 假设1 不成立: write_gate_mean 平均值 = {avg_wg:.6f}，并非很小")

    # 假设1 补充: slot_v 几乎不更新
    if avg_change < 5:
        print(f"  ✅ 假设1 补充成立: slot_v 变化百分比平均 {avg_change:.2f}%，几乎不更新")
    elif avg_change < 20:
        print(f"  ⚠️ 假设1 补充部分成立: slot_v 变化百分比平均 {avg_change:.2f}%，更新量较小")
    else:
        print(f"  ❌ 假设1 补充不成立: slot_v 变化百分比平均 {avg_change:.2f}%，更新量不小")

    # 假设2: slot_v std ~0.177
    init_stds = [r['slot_v_std_init'] for r in layer_results]
    after_stds = [r['slot_v_std_after'] for r in layer_results]
    avg_init_std = sum(init_stds) / len(init_stds)
    avg_after_std = sum(after_stds) / len(after_stds)
    print(f"\n  slot_v std (init): 平均 {avg_init_std:.6f}")
    print(f"  slot_v std (after 1 forward): 平均 {avg_after_std:.6f}")

    if abs(avg_init_std - 0.177) < 0.05:
        print(f"  ✅ 假设2 部分成立: slot_v init std ≈ {avg_init_std:.4f}，接近假设值 0.177")
    else:
        print(f"  ❌ 假设2 部分不成立: slot_v init std = {avg_init_std:.4f}，与假设值 0.177 差距较大")

    # slot_v / v_heads 比值 ~0.21
    if 0.1 < avg_ratio_std < 0.35:
        print(f"  ✅ 假设2 部分成立: slot_v/v_heads std 比值 ≈ {avg_ratio_std:.4f}，接近假设值 0.21")
    else:
        print(f"  ❌ 假设2 部分不成立: slot_v/v_heads std 比值 = {avg_ratio_std:.4f}，与假设值 0.21 差距较大")

    if 0.1 < avg_ratio_abs < 0.35:
        print(f"  ✅ 假设2 部分成立: slot_v/v_heads abs 比值 ≈ {avg_ratio_abs:.4f}，接近假设值 0.21")
    else:
        print(f"  ❌ 假设2 部分不成立: slot_v/v_heads abs 比值 = {avg_ratio_abs:.4f}，与假设值 0.21 差距较大")

    # 清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
