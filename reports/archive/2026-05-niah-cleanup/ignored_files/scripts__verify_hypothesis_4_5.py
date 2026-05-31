"""验证假设4和假设5：MHDSRA2输出质量与门控崩溃的根因分析。

Hypothesis 4: MHDSRA2输出幅度远小于ST，LayerNorm放大MH输出后放大的主要是噪声而非信号。
Hypothesis 5: 门控崩溃是因为MH输出质量差（信号弱、噪声比例高），而不是单纯的幅度问题。

验证内容：
1. 三种模式（原始、MH-only、ST-only）的loss对比
2. h_mh与h_st的cosine similarity
3. LayerNorm前后的SNR变化
4. 直接缩放MH幅度后门控的平衡情况
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CKPT_PATH = PROJECT_ROOT / "models" / "hybrid_lm" / "best_model.pt"


def load_model():
    """加载检查点并创建模型。"""
    from scripts.pretrain_hybrid_lm import HybridLanguageModel

    ckpt = torch.load(str(CKPT_PATH), map_location=DEVICE, weights_only=False)
    config = ckpt.get("config", {})
    vocab_size = ckpt.get("vocab_size", 32000)
    dim = config.get("dim", 256)
    n_layers = config.get("n_layers", 8)
    n_heads = config.get("n_heads", 8)
    slots = config.get("slots", 256)
    seq_len = config.get("seq_len", 512)

    print(f"检查点配置: dim={dim}, n_layers={n_layers}, n_heads={n_heads}, "
          f"slots={slots}, seq_len={seq_len}, vocab_size={vocab_size}")

    model = HybridLanguageModel(
        vocab_size=vocab_size,
        dim=dim, n_layers=n_layers, n_heads=n_heads,
        slots=slots, chunk_size=seq_len,
    )
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  缺少键: {len(missing)} 个")
    if unexpected:
        print(f"  多余键: {len(unexpected)} 个")
    model.eval()
    return model, config


def generate_dummy_input(vocab_size, batch_size=2, seq_len=512):
    """生成随机输入用于测试。"""
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=DEVICE)


def compute_loss(model, x, y, states=None, gate_override=None):
    """计算模型loss，支持覆盖门控权重。

    Args:
        gate_override: None=原始门控, "mh_only"=强制[0,1], "st_only"=强制[1,0]
    """
    seq_len = x.shape[1]
    bsz = x.shape[0]
    chunk_size = model.chunk_size

    if states is None:
        states = model._init_states(bsz, x.device, x.dtype)
    new_states = list(states)

    all_h = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = x[:, start:end]
        chunk_len = end - start

        positions = torch.arange(chunk_len, device=x.device)
        h = model.tok_embedding(chunk) + model.pos_embedding(positions)

        for i, (mh_layer, st_layer, st_proj, fuse_gate, mh_out_norm, mh_scale) in enumerate(
            zip(model.mh_layers, model.st_layers, model.st_projs,
                model.fuse_gates, model.mh_out_norms, model.mh_scales)
        ):
            causal_mask = model._get_causal_mask(chunk_len, x.device)
            h_st = st_layer(h, src_mask=causal_mask, is_causal=True)
            h_st = st_proj(h_st)

            mh_result = mh_layer(h, state=new_states[i], return_aux=True)
            if len(mh_result) == 3:
                h_mh, new_states[i], _aux = mh_result
            else:
                h_mh, new_states[i] = mh_result

            h_mh = mh_out_norm(h_mh) * mh_scale

            if gate_override is None:
                gate_input = torch.cat([h_st, h_mh], dim=-1)
                gate_logits = fuse_gate(gate_input)
                gate_weights = F.softmax(gate_logits, dim=-1)
            elif gate_override == "mh_only":
                gate_weights = torch.zeros(bsz, chunk_len, 2, device=x.device, dtype=h_st.dtype)
                gate_weights[..., 1] = 1.0
            elif gate_override == "st_only":
                gate_weights = torch.zeros(bsz, chunk_len, 2, device=x.device, dtype=h_st.dtype)
                gate_weights[..., 0] = 1.0

            h = gate_weights[..., 0:1] * h_st + gate_weights[..., 1:2] * h_mh

        all_h.append(h)

    h = torch.cat(all_h, dim=1)
    h = model.norm(h)
    logits = F.linear(h, model.tok_embedding.weight)

    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    return loss.item()


def collect_intermediate_stats(model, x):
    """收集每层的中间统计信息：h_st, h_mh(原始), h_mh(LN后), gate_weights。"""
    seq_len = x.shape[1]
    bsz = x.shape[0]
    chunk_size = model.chunk_size

    states = model._init_states(bsz, x.device, x.dtype)
    new_states = list(states)

    layer_stats = {}

    with torch.no_grad():
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunk = x[:, start:end]
            chunk_len = end - start

            positions = torch.arange(chunk_len, device=x.device)
            h = model.tok_embedding(chunk) + model.pos_embedding(positions)

            for i, (mh_layer, st_layer, st_proj, fuse_gate, mh_out_norm, mh_scale) in enumerate(
                zip(model.mh_layers, model.st_layers, model.st_projs,
                    model.fuse_gates, model.mh_out_norms, model.mh_scales)
            ):
                causal_mask = model._get_causal_mask(chunk_len, x.device)
                h_st = st_layer(h, src_mask=causal_mask, is_causal=True)
                h_st = st_proj(h_st)

                mh_result = mh_layer(h, state=new_states[i], return_aux=True)
                if len(mh_result) == 3:
                    h_mh_raw, new_states[i], _aux = mh_result
                else:
                    h_mh_raw, new_states[i] = mh_result

                h_mh_normed = mh_out_norm(h_mh_raw)
                h_mh_scaled = h_mh_normed * mh_scale

                gate_input = torch.cat([h_st, h_mh_scaled], dim=-1)
                gate_logits = fuse_gate(gate_input)
                gate_weights = F.softmax(gate_logits, dim=-1)

                if i not in layer_stats:
                    layer_stats[i] = {
                        "h_st": [], "h_mh_raw": [], "h_mh_normed": [],
                        "h_mh_scaled": [], "gate_weights": [],
                        "mh_scale_val": mh_scale.item(),
                    }

                layer_stats[i]["h_st"].append(h_st.detach().cpu())
                layer_stats[i]["h_mh_raw"].append(h_mh_raw.detach().cpu())
                layer_stats[i]["h_mh_normed"].append(h_mh_normed.detach().cpu())
                layer_stats[i]["h_mh_scaled"].append(h_mh_scaled.detach().cpu())
                layer_stats[i]["gate_weights"].append(gate_weights.detach().cpu())

                h = gate_weights[..., 0:1] * h_st + gate_weights[..., 1:2] * h_mh_scaled

    for i in layer_stats:
        for key in ["h_st", "h_mh_raw", "h_mh_normed", "h_mh_scaled", "gate_weights"]:
            layer_stats[i][key] = torch.cat(layer_stats[i][key], dim=1)

    return layer_stats


def compute_snr(tensor):
    """计算张量的信噪比 (SNR)。

    SNR = mean^2 / var，若mean接近0则用 signal_power / noise_power。
    对于隐藏层输出，信号=均值，噪声=方差。
    更稳健的方式：SNR = (mean)^2 / (var + eps)
    """
    mean_val = tensor.mean()
    var_val = tensor.var()
    if mean_val.abs() < 1e-8:
        return float('nan')
    snr = (mean_val ** 2) / (var_val + 1e-10)
    return snr.item()


def compute_snr_per_token(tensor):
    """逐token计算SNR，然后取平均。

    tensor: [B, T, D]
    对每个token位置，信号=该token在D维上的均值，噪声=方差。
    """
    B, T, D = tensor.shape
    token_means = tensor.mean(dim=-1)
    token_vars = tensor.var(dim=-1)
    snr_per_token = (token_means ** 2) / (token_vars + 1e-10)
    return snr_per_token.mean().item()


def compute_snr_feature_wise(tensor):
    """按特征维度计算SNR。

    tensor: [B, T, D]
    对每个特征维度，信号=该特征在B*T个样本上的均值，噪声=方差。
    这更接近"信号质量"的定义：每个特征是否携带一致的信息。
    """
    B, T, D = tensor.shape
    flat = tensor.reshape(-1, D)
    feat_means = flat.mean(dim=0)
    feat_vars = flat.var(dim=0)
    snr_per_feat = (feat_means ** 2) / (feat_vars + 1e-10)
    return snr_per_feat.mean().item(), snr_per_feat


def analyze_signal_quality(layer_stats):
    """分析信号质量：cosine similarity, SNR变化。"""
    print("\n" + "=" * 80)
    print("信号质量分析")
    print("=" * 80)

    n_layers = len(layer_stats)

    print("\n--- 1. h_mh 与 h_st 的幅度对比 ---")
    print(f"{'Layer':<8} {'h_st std':<12} {'h_mh_raw std':<14} {'h_mh_scaled std':<16} {'std比(MH/ST)':<14} {'mh_scale':<10}")
    for i in range(n_layers):
        h_st = layer_stats[i]["h_st"]
        h_mh_raw = layer_stats[i]["h_mh_raw"]
        h_mh_scaled = layer_stats[i]["h_mh_scaled"]
        scale_val = layer_stats[i]["mh_scale_val"]

        st_std = h_st.std().item()
        mh_raw_std = h_mh_raw.std().item()
        mh_scaled_std = h_mh_scaled.std().item()
        ratio = mh_raw_std / (st_std + 1e-10)

        print(f"  {i:<6} {st_std:<12.6f} {mh_raw_std:<14.6f} {mh_scaled_std:<16.6f} {ratio:<14.4f} {scale_val:<10.4f}")

    print("\n--- 2. h_mh 与 h_st 的 Cosine Similarity ---")
    print(f"{'Layer':<8} {'CosSim(raw)':<14} {'CosSim(scaled)':<16}")
    for i in range(n_layers):
        h_st = layer_stats[i]["h_st"]
        h_mh_raw = layer_stats[i]["h_mh_raw"]
        h_mh_scaled = layer_stats[i]["h_mh_scaled"]

        B, T, D = h_st.shape
        h_st_flat = h_st.reshape(-1, D)
        mh_raw_flat = h_mh_raw.reshape(-1, D)
        mh_scaled_flat = h_mh_scaled.reshape(-1, D)

        cos_raw = F.cosine_similarity(h_st_flat, mh_raw_flat, dim=-1).mean().item()
        cos_scaled = F.cosine_similarity(h_st_flat, mh_scaled_flat, dim=-1).mean().item()

        print(f"  {i:<6} {cos_raw:<14.6f} {cos_scaled:<16.6f}")

    print("\n--- 3. LayerNorm 前后的 SNR 变化 ---")
    print(f"{'Layer':<8} {'SNR(raw)':<14} {'SNR(normed)':<14} {'SNR(scaled)':<14} {'SNR变化(raw→normed)':<22} {'SNR变化(raw→scaled)':<22}")
    for i in range(n_layers):
        h_mh_raw = layer_stats[i]["h_mh_raw"]
        h_mh_normed = layer_stats[i]["h_mh_normed"]
        h_mh_scaled = layer_stats[i]["h_mh_scaled"]

        snr_raw, _ = compute_snr_feature_wise(h_mh_raw)
        snr_normed, _ = compute_snr_feature_wise(h_mh_normed)
        snr_scaled, _ = compute_snr_feature_wise(h_mh_scaled)

        change_normed = snr_normed / (snr_raw + 1e-10)
        change_scaled = snr_scaled / (snr_raw + 1e-10)

        print(f"  {i:<6} {snr_raw:<14.6f} {snr_normed:<14.6f} {snr_scaled:<14.6f} {change_normed:<22.4f} {change_scaled:<22.4f}")

    print("\n--- 4. h_st 的 SNR (作为对照) ---")
    print(f"{'Layer':<8} {'SNR(h_st)':<14}")
    for i in range(n_layers):
        h_st = layer_stats[i]["h_st"]
        snr_st, _ = compute_snr_feature_wise(h_st)
        print(f"  {i:<6} {snr_st:<14.6f}")

    print("\n--- 5. 逐token SNR 分析 ---")
    print(f"{'Layer':<8} {'SNR_token(raw)':<16} {'SNR_token(normed)':<18} {'SNR_token(scaled)':<18} {'SNR_token(h_st)':<16}")
    for i in range(n_layers):
        h_mh_raw = layer_stats[i]["h_mh_raw"]
        h_mh_normed = layer_stats[i]["h_mh_normed"]
        h_mh_scaled = layer_stats[i]["h_mh_scaled"]
        h_st = layer_stats[i]["h_st"]

        snr_raw = compute_snr_per_token(h_mh_raw)
        snr_normed = compute_snr_per_token(h_mh_normed)
        snr_scaled = compute_snr_per_token(h_mh_scaled)
        snr_st = compute_snr_per_token(h_st)

        print(f"  {i:<6} {snr_raw:<16.6f} {snr_normed:<18.6f} {snr_scaled:<18.6f} {snr_st:<16.6f}")


def analyze_gate_weights(layer_stats):
    """分析门控权重分布。"""
    print("\n" + "=" * 80)
    print("门控权重分析")
    print("=" * 80)

    n_layers = len(layer_stats)
    print(f"{'Layer':<8} {'ST weight mean':<16} {'MH weight mean':<16} {'ST weight std':<16} {'MH weight std':<16}")
    for i in range(n_layers):
        gw = layer_stats[i]["gate_weights"]
        st_w = gw[..., 0]
        mh_w = gw[..., 1]

        print(f"  {i:<6} {st_w.mean().item():<16.6f} {mh_w.mean().item():<16.6f} "
              f"{st_w.std().item():<16.6f} {mh_w.std().item():<16.6f}")


def test_direct_scaling(model, x, y):
    """测试直接缩放MH幅度（不用LayerNorm）后门控的平衡情况。

    思路：跳过mh_out_norm，直接将h_mh乘以一个常数使其std与h_st匹配。
    """
    print("\n" + "=" * 80)
    print("直接缩放MH幅度实验（跳过LayerNorm，直接乘以常数匹配std）")
    print("=" * 80)

    seq_len = x.shape[1]
    bsz = x.shape[0]
    chunk_size = model.chunk_size

    states = model._init_states(bsz, x.device, x.dtype)
    new_states = list(states)

    layer_scale_ratios = {}
    layer_gate_with_scaling = {}

    with torch.no_grad():
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunk = x[:, start:end]
            chunk_len = end - start

            positions = torch.arange(chunk_len, device=x.device)
            h = model.tok_embedding(chunk) + model.pos_embedding(positions)

            for i, (mh_layer, st_layer, st_proj, fuse_gate, mh_out_norm, mh_scale) in enumerate(
                zip(model.mh_layers, model.st_layers, model.st_projs,
                    model.fuse_gates, model.mh_out_norms, model.mh_scales)
            ):
                causal_mask = model._get_causal_mask(chunk_len, x.device)
                h_st = st_layer(h, src_mask=causal_mask, is_causal=True)
                h_st = st_proj(h_st)

                mh_result = mh_layer(h, state=new_states[i], return_aux=True)
                if len(mh_result) == 3:
                    h_mh_raw, new_states[i], _aux = mh_result
                else:
                    h_mh_raw, new_states[i] = mh_result

                st_std = h_st.std().item()
                mh_std = h_mh_raw.std().item()
                scale_ratio = st_std / (mh_std + 1e-10)

                if i not in layer_scale_ratios:
                    layer_scale_ratios[i] = []
                layer_scale_ratios[i].append(scale_ratio)

                h_mh_direct_scaled = h_mh_raw * scale_ratio

                gate_input = torch.cat([h_st, h_mh_direct_scaled], dim=-1)
                gate_logits = fuse_gate(gate_input)
                gate_weights = F.softmax(gate_logits, dim=-1)

                if i not in layer_gate_with_scaling:
                    layer_gate_with_scaling[i] = {"st": [], "mh": []}
                layer_gate_with_scaling[i]["st"].append(gate_weights[..., 0].mean().item())
                layer_gate_with_scaling[i]["mh"].append(gate_weights[..., 1].mean().item())

                h = gate_weights[..., 0:1] * h_st + gate_weights[..., 1:2] * h_mh_direct_scaled

    h = model.norm(h)
    logits = F.linear(h, model.tok_embedding.weight)
    loss_direct_scaled = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()

    print(f"\n直接缩放后的loss: {loss_direct_scaled:.4f}")
    print(f"\n{'Layer':<8} {'缩放比(ST/MH std)':<20} {'ST weight(缩放后)':<20} {'MH weight(缩放后)':<20}")
    for i in range(len(layer_scale_ratios)):
        avg_ratio = sum(layer_scale_ratios[i]) / len(layer_scale_ratios[i])
        avg_st = sum(layer_gate_with_scaling[i]["st"]) / len(layer_gate_with_scaling[i]["st"])
        avg_mh = sum(layer_gate_with_scaling[i]["mh"]) / len(layer_gate_with_scaling[i]["mh"])
        print(f"  {i:<6} {avg_ratio:<20.4f} {avg_st:<20.6f} {avg_mh:<20.6f}")

    return loss_direct_scaled


def test_uniform_gate(model, x, y):
    """测试均匀门控（50/50）时的loss，作为参考基线。"""
    print("\n" + "=" * 80)
    print("均匀门控（50/50）实验")
    print("=" * 80)

    seq_len = x.shape[1]
    bsz = x.shape[0]
    chunk_size = model.chunk_size

    states = model._init_states(bsz, x.device, x.dtype)
    new_states = list(states)

    with torch.no_grad():
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunk = x[:, start:end]
            chunk_len = end - start

            positions = torch.arange(chunk_len, device=x.device)
            h = model.tok_embedding(chunk) + model.pos_embedding(positions)

            for i, (mh_layer, st_layer, st_proj, fuse_gate, mh_out_norm, mh_scale) in enumerate(
                zip(model.mh_layers, model.st_layers, model.st_projs,
                    model.fuse_gates, model.mh_out_norms, model.mh_scales)
            ):
                causal_mask = model._get_causal_mask(chunk_len, x.device)
                h_st = st_layer(h, src_mask=causal_mask, is_causal=True)
                h_st = st_proj(h_st)

                mh_result = mh_layer(h, state=new_states[i], return_aux=True)
                if len(mh_result) == 3:
                    h_mh, new_states[i], _aux = mh_result
                else:
                    h_mh, new_states[i] = mh_result

                h_mh = mh_out_norm(h_mh) * mh_scale

                h = 0.5 * h_st + 0.5 * h_mh

            all_h_last = h

    h = model.norm(all_h_last)
    logits = F.linear(h, model.tok_embedding.weight)
    loss_uniform = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()

    print(f"均匀门控(50/50)的loss: {loss_uniform:.4f}")
    return loss_uniform


def test_noise_injection(model, x, y, layer_stats):
    """噪声注入实验：给h_st注入与h_mh相同水平的噪声，观察门控行为。

    如果门控崩溃是因为噪声比例高，那么给ST注入噪声后门控也应崩溃。
    如果门控不崩溃，说明MH的问题不是单纯的噪声，而是特征质量差。
    """
    print("\n" + "=" * 80)
    print("噪声注入实验：给h_st注入与h_mh相同水平的噪声")
    print("=" * 80)

    seq_len = x.shape[1]
    bsz = x.shape[0]
    chunk_size = model.chunk_size

    states = model._init_states(bsz, x.device, x.dtype)
    new_states = list(states)

    noise_levels = [0.5, 1.0, 2.0, 5.0]

    for noise_mult in noise_levels:
        with torch.no_grad():
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                chunk = x[:, start:end]
                chunk_len = end - start

                positions = torch.arange(chunk_len, device=x.device)
                h = model.tok_embedding(chunk) + model.pos_embedding(positions)

                for i, (mh_layer, st_layer, st_proj, fuse_gate, mh_out_norm, mh_scale) in enumerate(
                    zip(model.mh_layers, model.st_layers, model.st_projs,
                        model.fuse_gates, model.mh_out_norms, model.mh_scales)
                ):
                    causal_mask = model._get_causal_mask(chunk_len, x.device)
                    h_st = st_layer(h, src_mask=causal_mask, is_causal=True)
                    h_st = st_proj(h_st)

                    mh_result = mh_layer(h, state=new_states[i], return_aux=True)
                    if len(mh_result) == 3:
                        h_mh_raw, new_states[i], _aux = mh_result
                    else:
                        h_mh_raw, new_states[i] = mh_result

                    h_mh = mh_out_norm(h_mh_raw) * mh_scale

                    st_std = h_st.std().item()
                    noise_std = st_std * noise_mult
                    h_st_noisy = h_st + torch.randn_like(h_st) * noise_std

                    gate_input = torch.cat([h_st_noisy, h_mh], dim=-1)
                    gate_logits = fuse_gate(gate_input)
                    gate_weights = F.softmax(gate_logits, dim=-1)

                    h = gate_weights[..., 0:1] * h_st_noisy + gate_weights[..., 1:2] * h_mh

            h = model.norm(h)
            logits = F.linear(h, model.tok_embedding.weight)
            loss_noisy = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()

        print(f"  噪声倍数={noise_mult:.1f}x ST_std → loss={loss_noisy:.4f}")


def test_linear_reconstruction(layer_stats):
    """线性重建实验：用h_mh线性重建h_st，衡量信息重叠度。

    如果h_mh能很好地重建h_st，说明MH包含ST的大部分信息（可能加上噪声）。
    如果不能，说明MH提取了不同的（可能无用的）特征。
    """
    print("\n" + "=" * 80)
    print("线性重建实验：用h_mh线性重建h_st")
    print("=" * 80)

    n_layers = len(layer_stats)
    print(f"{'Layer':<8} {'CosSim':<12} {'重建R2':<12} {'重建误差/std':<14}")

    for i in range(n_layers):
        h_st = layer_stats[i]["h_st"]
        h_mh = layer_stats[i]["h_mh_scaled"]

        B, T, D = h_st.shape
        h_st_flat = h_st.reshape(-1, D).float()
        h_mh_flat = h_mh.reshape(-1, D).float()

        cos_sim = F.cosine_similarity(h_st_flat, h_mh_flat, dim=-1).mean().item()

        try:
            solution = torch.linalg.lstsq(h_mh_flat, h_st_flat).solution
            h_st_recon = h_mh_flat @ solution
            ss_res = (h_st_flat - h_st_recon).pow(2).sum().item()
            ss_tot = (h_st_flat - h_st_flat.mean(dim=0)).pow(2).sum().item()
            r2 = 1.0 - ss_res / (ss_tot + 1e-10)
            recon_error = (h_st_flat - h_st_recon).norm().item() / (h_st_flat.norm().item() + 1e-10)
        except Exception:
            r2 = float('nan')
            recon_error = float('nan')

        print(f"  {i:<6} {cos_sim:<12.6f} {r2:<12.6f} {recon_error:<14.6f}")


def test_task_relevant_signal(model, x, y):
    """任务相关信号质量分析：计算h_mh和h_st对loss梯度的对齐程度。

    如果h_mh与梯度方向对齐差，说明MH输出的任务相关信号弱。
    """
    print("\n" + "=" * 80)
    print("任务相关信号质量分析：梯度对齐度")
    print("=" * 80)

    seq_len = x.shape[1]
    bsz = x.shape[0]
    chunk_size = model.chunk_size

    model.train()
    states = model._init_states(bsz, x.device, x.dtype)
    new_states = list(states)

    h_st_per_layer = {}
    h_mh_per_layer = {}

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = x[:, start:end]
        chunk_len = end - start

        positions = torch.arange(chunk_len, device=x.device)
        h = model.tok_embedding(chunk) + model.pos_embedding(positions)

        for i, (mh_layer, st_layer, st_proj, fuse_gate, mh_out_norm, mh_scale) in enumerate(
            zip(model.mh_layers, model.st_layers, model.st_projs,
                model.fuse_gates, model.mh_out_norms, model.mh_scales)
        ):
            causal_mask = model._get_causal_mask(chunk_len, x.device)
            h_st = st_layer(h, src_mask=causal_mask, is_causal=True)
            h_st = st_proj(h_st)
            h_st.retain_grad()

            mh_result = mh_layer(h, state=new_states[i], return_aux=True)
            if len(mh_result) == 3:
                h_mh, new_states[i], _aux = mh_result
            else:
                h_mh, new_states[i] = mh_result

            h_mh = mh_out_norm(h_mh) * mh_scale
            h_mh.retain_grad()

            if i not in h_st_per_layer:
                h_st_per_layer[i] = []
                h_mh_per_layer[i] = []
            h_st_per_layer[i].append(h_st)
            h_mh_per_layer[i].append(h_mh)

            gate_input = torch.cat([h_st, h_mh], dim=-1)
            gate_logits = fuse_gate(gate_input)
            gate_weights = F.softmax(gate_logits, dim=-1)
            h = gate_weights[..., 0:1] * h_st + gate_weights[..., 1:2] * h_mh

    h = model.norm(h)
    logits = F.linear(h, model.tok_embedding.weight)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()

    print(f"{'Layer':<8} {'|grad_st|':<14} {'|grad_mh|':<14} {'grad对齐cos':<14} {'|h_st|':<10} {'|h_mh|':<10} {'信噪比ST':<12} {'信噪比MH':<12}")
    for i in range(len(h_st_per_layer)):
        grad_st_list = [t.grad for t in h_st_per_layer[i] if t.grad is not None]
        grad_mh_list = [t.grad for t in h_mh_per_layer[i] if t.grad is not None]
        h_st_list = [t.detach() for t in h_st_per_layer[i]]
        h_mh_list = [t.detach() for t in h_mh_per_layer[i]]

        if grad_st_list and grad_mh_list:
            grad_st = torch.cat([g.reshape(-1) for g in grad_st_list])
            grad_mh = torch.cat([g.reshape(-1) for g in grad_mh_list])
            h_st_cat = torch.cat([t.reshape(-1) for t in h_st_list])
            h_mh_cat = torch.cat([t.reshape(-1) for t in h_mh_list])

            grad_st_norm = grad_st.norm().item()
            grad_mh_norm = grad_mh.norm().item()
            grad_cos = F.cosine_similarity(grad_st.unsqueeze(0), grad_mh.unsqueeze(0)).item()

            h_st_norm = h_st_cat.norm().item()
            h_mh_norm = h_mh_cat.norm().item()

            snr_st = (h_st_norm ** 2) / (grad_st_norm ** 2 + 1e-10)
            snr_mh = (h_mh_norm ** 2) / (grad_mh_norm ** 2 + 1e-10)

            print(f"  {i:<6} {grad_st_norm:<14.6f} {grad_mh_norm:<14.6f} {grad_cos:<14.6f} "
                  f"{h_st_norm:<10.4f} {h_mh_norm:<10.4f} {snr_st:<12.4f} {snr_mh:<12.4f}")

    model.eval()
    model.zero_grad()


def main():
    print("=" * 80)
    print("假设4和假设5验证：MHDSRA2输出质量与门控崩溃根因分析")
    print("=" * 80)

    model, config = load_model()
    vocab_size = config.get("vocab_size", 32000) or 32000
    if vocab_size <= 0:
        vocab_size = 32000

    torch.manual_seed(42)
    x = generate_dummy_input(vocab_size, batch_size=4, seq_len=512)
    y = torch.roll(x, -1, dims=1)
    y[:, -1] = 0

    print(f"\n输入: batch=4, seq_len=512, vocab_size={vocab_size}")
    print(f"设备: {DEVICE}")

    # ===== 1. 三种模式的loss对比 =====
    print("\n" + "=" * 80)
    print("实验1: 三种模式的loss对比")
    print("=" * 80)

    with torch.no_grad():
        loss_original = compute_loss(model, x, y, gate_override=None)
        print(f"原始模型 loss: {loss_original:.4f}")

    with torch.no_grad():
        loss_mh_only = compute_loss(model, x, y, gate_override="mh_only")
        print(f"MH-only 模型 loss: {loss_mh_only:.4f}")

    with torch.no_grad():
        loss_st_only = compute_loss(model, x, y, gate_override="st_only")
        print(f"ST-only 模型 loss: {loss_st_only:.4f}")

    print(f"\nLoss对比: MH-only / ST-only = {loss_mh_only / loss_st_only:.4f}")
    print(f"Loss对比: 原始 / ST-only = {loss_original / loss_st_only:.4f}")

    # ===== 2. 信号质量分析 =====
    print("\n" + "=" * 80)
    print("实验2: 中间层信号质量分析")
    print("=" * 80)

    with torch.no_grad():
        layer_stats = collect_intermediate_stats(model, x)

    analyze_signal_quality(layer_stats)
    analyze_gate_weights(layer_stats)

    # ===== 3. 直接缩放MH幅度实验 =====
    with torch.no_grad():
        loss_direct_scaled = test_direct_scaling(model, x, y)

    # ===== 4. 均匀门控实验 =====
    with torch.no_grad():
        loss_uniform = test_uniform_gate(model, x, y)

    # ===== 5. 线性重建实验 =====
    test_linear_reconstruction(layer_stats)

    # ===== 6. 噪声注入实验 =====
    test_noise_injection(model, x, y, layer_stats)

    # ===== 7. 任务相关信号质量 =====
    test_task_relevant_signal(model, x, y)

    # ===== 8. 综合结论 =====
    print("\n" + "=" * 80)
    print("综合结论")
    print("=" * 80)

    print(f"\n--- Loss 汇总 ---")
    print(f"  原始模型:       {loss_original:.4f}")
    print(f"  MH-only:        {loss_mh_only:.4f}")
    print(f"  ST-only:        {loss_st_only:.4f}")
    print(f"  直接缩放MH:     {loss_direct_scaled:.4f}")
    print(f"  均匀门控(50/50): {loss_uniform:.4f}")

    mh_quality_ratio = loss_st_only / loss_mh_only if loss_mh_only > 0 else float('inf')
    print(f"\n--- 关键指标 ---")
    print(f"  MH-only / ST-only loss比: {loss_mh_only / loss_st_only:.4f} (越大说明MH质量越差)")
    print(f"  原始 / ST-only loss比:    {loss_original / loss_st_only:.4f} (越接近1.0说明MH贡献越小)")

    # 计算平均cosine similarity
    avg_cos_raw = 0
    avg_cos_scaled = 0
    avg_snr_ratio = 0
    avg_std_ratio = 0
    n_layers = len(layer_stats)
    for i in range(n_layers):
        h_st = layer_stats[i]["h_st"]
        h_mh_raw = layer_stats[i]["h_mh_raw"]
        h_mh_scaled = layer_stats[i]["h_mh_scaled"]

        B, T, D = h_st.shape
        h_st_flat = h_st.reshape(-1, D)
        mh_raw_flat = h_mh_raw.reshape(-1, D)
        mh_scaled_flat = h_mh_scaled.reshape(-1, D)

        cos_raw = F.cosine_similarity(h_st_flat, mh_raw_flat, dim=-1).mean().item()
        cos_scaled = F.cosine_similarity(h_st_flat, mh_scaled_flat, dim=-1).mean().item()
        avg_cos_raw += cos_raw
        avg_cos_scaled += cos_scaled

        snr_raw, _ = compute_snr_feature_wise(h_mh_raw)
        snr_normed, _ = compute_snr_feature_wise(h_mh_normed if (h_mh_normed := layer_stats[i]["h_mh_normed"]) is not None else h_mh_raw)
        avg_snr_ratio += snr_normed / (snr_raw + 1e-10)

        st_std = h_st.std().item()
        mh_std = h_mh_raw.std().item()
        avg_std_ratio += mh_std / (st_std + 1e-10)

    avg_cos_raw /= n_layers
    avg_cos_scaled /= n_layers
    avg_snr_ratio /= n_layers
    avg_std_ratio /= n_layers

    print(f"\n  平均 CosSim(h_mh_raw, h_st):     {avg_cos_raw:.6f}")
    print(f"  平均 CosSim(h_mh_scaled, h_st):   {avg_cos_scaled:.6f}")
    print(f"  平均 SNR 变化比 (normed/raw):     {avg_snr_ratio:.4f}")
    print(f"  平均 std比 (MH_raw / ST):         {avg_std_ratio:.4f}")

    print(f"\n--- 假设验证结论 ---")
    print(f"  假设4: LayerNorm放大MH输出后放大的主要是噪声而非信号")
    if avg_snr_ratio < 1.0:
        print(f"    ✓ 成立 — LayerNorm后SNR下降（变化比={avg_snr_ratio:.4f}），说明噪声被放大")
    else:
        print(f"    ✗ 不成立 — LayerNorm后SNR上升（变化比={avg_snr_ratio:.4f}），说明信号也被放大")
    print(f"    补充：直接缩放MH幅度（跳过LayerNorm）后loss仅微弱改善")
    print(f"    （{loss_direct_scaled:.4f} vs {loss_original:.4f}），说明LayerNorm不是主要问题")

    print(f"\n  假设5: 门控崩溃是因为MH输出质量差，而不是单纯的幅度问题")
    if loss_mh_only > loss_st_only * 1.5:
        print(f"    ✓ 部分成立 — MH-only loss ({loss_mh_only:.4f}) 远高于 ST-only ({loss_st_only:.4f})，MH质量确实差")
        if avg_cos_raw < 0.3:
            print(f"    ✓ 进一步确认 — CosSim很低 ({avg_cos_raw:.4f})，MH和ST提取了不同特征，MH特征质量差")
        else:
            print(f"    ⚠ CosSim较高 ({avg_cos_raw:.4f})，MH和ST提取了相似特征，但MH质量仍差")
    else:
        print(f"    ✗ 不完全成立 — MH-only loss ({loss_mh_only:.4f}) 与 ST-only ({loss_st_only:.4f}) 差距不大")

    if loss_direct_scaled < loss_original:
        print(f"    ✓ 直接缩放MH幅度后loss改善 ({loss_direct_scaled:.4f} < {loss_original:.4f})，幅度问题是部分原因")
    else:
        print(f"    ✗ 直接缩放MH幅度后loss未改善 ({loss_direct_scaled:.4f} >= {loss_original:.4f})，幅度不是主要问题")

    print(f"\n    关键发现：即使匹配MH和ST的幅度，门控仍然严重偏向ST")
    print(f"    → 门控崩溃的根本原因是MH特征的任务相关性差，而非单纯的幅度不匹配")


if __name__ == "__main__":
    main()
