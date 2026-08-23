"""MQAR (Multi-Query Associative Recall) 对抗性压力测试与严苛边界验证套件.

Adversarial Stress Test Suite for Multi-Query Associative Recall (MQAR).

中文说明:
- 调用方 / Called by: `pytest tests/test_mqar_adversarial_stress.py`
- 目的: 作为独立的对抗性挑战者 (Challenger 2)，对 MQAR 数据生成、因果性、未来泄漏抵御、损失掩码、词表互斥及 Oracle 全知探针进行极端对抗性压力测试。
- 覆盖维度:
  1. 因果完整性与未来信息泄漏对抗测试 (Causal Integrity & Anti-Leakage)
  2. 损失掩码严格性与非查询位置扰动不变性 (Loss Masking Invariance under Noise)
  3. 词表互斥性与 Distractor 碰撞防御 (Disjoint Vocabulary & Distractor Collision Defense)
  4. Oracle 全知探针抗欺骗能力 (Adversarial Traps, Key Shadowing, Distractor Flooding, Permutations)
  5. 极端超参尺度压测 (Extreme Scale: Smallest V=4, Largest V=65536, Long Sequence L=4096, Dense KV)
  6. 评测流水线端到端验证 (evaluate_mqar directly with Oracle)
  7. 紧凑极限长度 (Minimal Seq Len L = 2K + Q)
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dsra.domain.mqar import MQARConfig, MQAROracleModel, generate_mqar_batch
from scripts.benchmark_mqar import evaluate_mqar


@pytest.fixture
def device() -> torch.device:
    """获取测试设备 (优先 cuda:0，回退 cpu)."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_adversarial_causal_integrity_and_anti_leakage(device):
    """对抗测试 1: 验证前缀 KV 放置与后缀 Query 的因果时间次序、输入序列 X 中绝无未来目标泄漏.

    Adversarial verification of causal order and strict anti-leakage in input sequence X.
    """
    test_scales = [
        {"vocab_size": 256, "seq_len": 512, "num_kv_pairs": 8, "num_queries": 8, "insert_mode": "uniform"},
        {"vocab_size": 256, "seq_len": 512, "num_kv_pairs": 8, "num_queries": 8, "insert_mode": "random"},
        {"vocab_size": 512, "seq_len": 1024, "num_kv_pairs": 16, "num_queries": 16, "insert_mode": "random"},
        {"vocab_size": 128, "seq_len": 256, "num_kv_pairs": 4, "num_queries": 2, "insert_mode": "uniform"},
    ]

    for cfg_kwargs in test_scales:
        cfg = MQARConfig(**cfg_kwargs)
        batch_size = 8
        # 生成多个批次进行统计验证
        for b_seed in range(5):
            X, Y, qpos, targets = generate_mqar_batch(
                batch_size, cfg, device=device, seed=10000 + b_seed
            )

            k_start = 1
            k_end = 1 + cfg.key_pool_size
            v_start = k_end
            v_end = k_end + cfg.val_pool_size

            for b in range(batch_size):
                seq = X[b].tolist()
                q_positions = qpos[b].tolist()
                first_qpos = min(q_positions)

                # 1. 验证所有前缀 KV 均位于首个 Query 之前
                prefix_kvs = {}
                for t in range(first_qpos):
                    if t + 1 < first_qpos:
                        k, v = seq[t], seq[t + 1]
                        if k_start <= k < k_end and v_start <= v < v_end:
                            prefix_kvs[k] = v

                assert len(prefix_kvs) == cfg.num_kv_pairs, (
                    f"Expected {cfg.num_kv_pairs} valid prefix KV pairs, found {len(prefix_kvs)}"
                )

                # 2. 验证 Query 位置的 Token 必须是有效的 Key
                for q_idx, q_p in enumerate(q_positions):
                    q_key = seq[q_p]
                    expected_val = int(targets[b, q_idx].item())
                    assert k_start <= q_key < k_end, f"Token at qpos={q_p} is not a valid Key: {q_key}"
                    assert q_key in prefix_kvs, f"Query key {q_key} was not present in prefix KV pairs!"
                    assert prefix_kvs[q_key] == expected_val, "Prefix KV mapping mismatch!"

                    # 3. 验证输入序列 X 在 qpos 处及之后绝无直接泄露的对应 Value 作为 prompt 提示
                    if q_p + 1 < cfg.seq_len:
                        next_tok = seq[q_p + 1]
                        # 下一个 token 只能是 Distractor (>= filler_start) 或 下一个 Query Key (k_start <= tok < k_end)
                        # 绝不能是 Value (v_start <= tok < v_end)
                        assert not (v_start <= next_tok < v_end), (
                            f"Leakage detected! X[b, {q_p + 1}]={next_tok} is in Value range [{v_start}, {v_end})"
                        )

                # 4. 验证在整个查询区间 [first_qpos, L) 的输入序列 X 中，绝无出现任何 Value tokens
                for t in range(first_qpos, cfg.seq_len):
                    tok = seq[t]
                    assert not (v_start <= tok < v_end), (
                        f"Value token leakage in query region! X[b, {t}]={tok} in Value range [{v_start}, {v_end})"
                    )


def test_adversarial_loss_masking_and_perturbation_invariance(device):
    """对抗测试 2: 验证损失掩码在非 Query 位置严格为 0，且向非 Query 位置注入巨幅噪声时损失和梯度完全不变.

    Adversarial verification that non-query positions produce zero loss and have zero gradient.
    """
    cfg = MQARConfig(vocab_size=256, seq_len=512, num_kv_pairs=8, num_queries=8)
    batch_size = 4
    X, Y, qpos, targets = generate_mqar_batch(batch_size, cfg, device=device, seed=2026)

    # 1. 验证非 query 位置的目标全为 0 (ignore_index=0)
    for b in range(batch_size):
        query_set = set(qpos[b].tolist())
        for t in range(cfg.seq_len):
            if t in query_set:
                assert Y[b, t].item() > 0
            else:
                assert Y[b, t].item() == 0

    # 2. 构造可导 Logits 张量并注入非 Query 位置的极大对抗噪声
    logits_clean = torch.randn((batch_size, cfg.seq_len, cfg.vocab_size), device=device, requires_grad=True)
    crit = nn.CrossEntropyLoss(ignore_index=0)

    loss_clean = crit(logits_clean.view(-1, cfg.vocab_size), Y.view(-1))
    loss_clean.backward()
    grad_clean = logits_clean.grad.clone()

    # 验证非 Query 位置的梯度严格为 0
    for b in range(batch_size):
        query_set = set(qpos[b].tolist())
        for t in range(cfg.seq_len):
            if t not in query_set:
                assert torch.all(grad_clean[b, t] == 0.0), f"Non-zero gradient found at non-query step {t}!"

    # 3. 对抗性扰动: 在所有非 Query 位置注入 [-1000.0, 1000.0] 的巨幅噪声
    logits_perturbed = logits_clean.detach().clone()
    logits_perturbed.requires_grad_(True)
    for b in range(batch_size):
        query_set = set(qpos[b].tolist())
        for t in range(cfg.seq_len):
            if t not in query_set:
                logits_perturbed.data[b, t] = (torch.rand(cfg.vocab_size, device=device) - 0.5) * 2000.0

    loss_perturbed = crit(logits_perturbed.view(-1, cfg.vocab_size), Y.view(-1))
    loss_perturbed.backward()

    # 验证损失计算对非 Query 位置的扰动具有绝对不变性
    diff = torch.abs(loss_clean - loss_perturbed).item()
    assert diff < 1e-6, f"Loss changed under non-query perturbation! diff={diff}"

    # 验证 Query 位置处的梯度与原始状态完全一致
    for b in range(batch_size):
        for q_p in qpos[b].tolist():
            assert torch.allclose(logits_perturbed.grad[b, q_p], grad_clean[b, q_p], atol=1e-6)


def test_adversarial_vocabulary_disjointness_and_distractor_collision_defense(device):
    """对抗测试 3: 词表四路严格互斥划分与 Distractor 碰撞防御.

    Adversarial verification of disjoint vocabulary sets across boundary cases and random seeds.
    """
    boundary_vocab_configs = [
        # 极小词表: V=4, K=1
        {"vocab_size": 4, "seq_len": 4, "num_kv_pairs": 1, "num_queries": 1},
        # 奇数与质数词表
        {"vocab_size": 5, "seq_len": 6, "num_kv_pairs": 1, "num_queries": 1},
        {"vocab_size": 7, "seq_len": 8, "num_kv_pairs": 1, "num_queries": 1},
        {"vocab_size": 13, "seq_len": 16, "num_kv_pairs": 2, "num_queries": 2},
        {"vocab_size": 31, "seq_len": 32, "num_kv_pairs": 4, "num_queries": 4},
        # 紧凑临界容量 V = 2K + 2
        {"vocab_size": 18, "seq_len": 32, "num_kv_pairs": 8, "num_queries": 8},
        # 大词表
        {"vocab_size": 4096, "seq_len": 1024, "num_kv_pairs": 64, "num_queries": 64},
        {"vocab_size": 65536, "seq_len": 2048, "num_kv_pairs": 128, "num_queries": 128},
    ]

    for cfg_kwargs in boundary_vocab_configs:
        cfg = MQARConfig(**cfg_kwargs)
        k_start = 1
        k_end = 1 + cfg.key_pool_size
        v_start = k_end
        v_end = k_end + cfg.val_pool_size
        f_start = v_end
        f_end = cfg.vocab_size

        pad_set = {0}
        key_set = set(range(k_start, k_end))
        val_set = set(range(v_start, v_end))
        filler_set = set(range(f_start, f_end))

        # 集合论互斥性断言
        assert len(pad_set & key_set) == 0
        assert len(pad_set & val_set) == 0
        assert len(pad_set & filler_set) == 0
        assert len(key_set & val_set) == 0
        assert len(key_set & filler_set) == 0
        assert len(val_set & filler_set) == 0
        assert len(pad_set) + len(key_set) + len(val_set) + len(filler_set) == cfg.vocab_size

        # 生成一批真实数据并验证实际采样中无任何碰撞与越界
        X, Y, qpos, targets = generate_mqar_batch(4, cfg, device=device, seed=42)
        x_flat = X.view(-1).tolist()
        for tok in x_flat:
            assert tok != 0, "Pad token (0) found in input X!"
            assert 0 <= tok < cfg.vocab_size, f"Token {tok} out of vocab bounds [0, {cfg.vocab_size})!"

        # 验证 targets 仅包含 val_set
        for tgt in targets.view(-1).tolist():
            assert tgt in val_set, f"Target {tgt} is not in val_set!"


def test_adversarial_oracle_probe_traps_and_robustness(device):
    """对抗测试 4: 构造对抗陷阱序列对 Oracle 探针进行抗欺骗、抗重名覆盖与边界压测.

    Adversarial traps: Distractor flooding, adjacent false patterns, key shadowing, and unseen keys.
    """
    vocab_size = 256
    cfg = MQARConfig(vocab_size=vocab_size, seq_len=512, num_kv_pairs=8)
    oracle = MQAROracleModel.from_config(cfg).to(device)
    oracle.eval()

    k_start = 1
    k_end = 1 + cfg.key_pool_size
    v_start = k_end
    v_end = k_end + cfg.val_pool_size
    f_start = v_end

    # 1. 对抗陷阱 1: Distractor 伪键值对陷阱 (如 [key, filler], [filler, val], [val, key], [filler, filler])
    # 验证 Oracle 绝不会因局部出现伪模式而误记录记忆
    seq_trap = torch.full((1, 100), f_start, dtype=torch.long, device=device)
    seq_trap[0, 10] = k_start        # Key
    seq_trap[0, 11] = f_start + 1    # Filler (不是 Value! 伪对)
    seq_trap[0, 20] = f_start + 2    # Filler
    seq_trap[0, 21] = v_start        # Value (前面是 Filler! 伪对)
    seq_trap[0, 30] = v_start + 1    # Value
    seq_trap[0, 31] = k_start + 1    # Key (顺序颠倒! 伪对)

    # 在步骤 50 输入查询 k_start 与 k_start + 1，预期 Logits 全为 0 (未命中任何合法 KV)
    seq_trap[0, 50] = k_start
    seq_trap[0, 51] = k_start + 1

    with torch.no_grad():
        trap_logits = oracle(seq_trap)
        assert trap_logits[0, 50].max().item() == 0.0, "Oracle mistakenly memorized a false (key, filler) pair!"
        assert trap_logits[0, 51].max().item() == 0.0, "Oracle mistakenly memorized a false (val, key) pair!"

    # 2. 对抗陷阱 2: Key Shadowing / 动态更新覆盖测试
    # 序列中先后出现 (K1, V1) 和 (K1, V2)，验证因果更新:
    # 在第二对出现之前查询 K1 必须返回 V1; 在第二对出现之后查询 K1 必须返回 V2.
    seq_shadow = torch.full((1, 60), f_start, dtype=torch.long, device=device)
    k1 = k_start
    v1 = v_start
    v2 = v_start + 1

    seq_shadow[0, 5] = k1
    seq_shadow[0, 6] = v1    # 写入 (k1, v1)
    seq_shadow[0, 15] = k1   # 第一次查询 k1 (应为 v1)
    seq_shadow[0, 25] = k1
    seq_shadow[0, 26] = v2   # 覆盖写入 (k1, v2)
    seq_shadow[0, 40] = k1   # 第二次查询 k1 (应更新为 v2)

    with torch.no_grad():
        shadow_logits = oracle(seq_shadow)
        pred_1 = shadow_logits[0, 15].argmax(dim=-1).item()
        pred_2 = shadow_logits[0, 40].argmax(dim=-1).item()
        assert pred_1 == v1, f"Pre-update query failed! Expected {v1}, got {pred_1}"
        assert pred_2 == v2, f"Post-update query failed! Expected {v2}, got {pred_2}"

    # 3. 对抗陷阱 3: 未出现 Key 的零幻觉测试 (Unseen Query Zero Hallucination)
    seq_unseen = torch.full((1, 30), f_start, dtype=torch.long, device=device)
    seq_unseen[0, 2] = k_start
    seq_unseen[0, 3] = v_start
    unseen_key = k_start + 2
    seq_unseen[0, 10] = unseen_key

    with torch.no_grad():
        unseen_logits = oracle(seq_unseen)
        assert unseen_logits[0, 10].max().item() == 0.0, "Oracle hallucinated a prediction for an unseen key!"

    # 4. 对抗陷阱 4: 极端长序列 (L=4096, K=128) 下的 100.0% 完美召回
    cfg_long = MQARConfig(vocab_size=1024, seq_len=4096, num_kv_pairs=128, num_queries=128)
    oracle_long = MQAROracleModel.from_config(cfg_long).to(device)
    X_long, Y_long, qpos_long, targets_long = generate_mqar_batch(2, cfg_long, device=device, seed=999)

    with torch.no_grad():
        logits_long = oracle_long(X_long)
        loss_long = F.cross_entropy(logits_long.view(-1, cfg_long.vocab_size), Y_long.view(-1), ignore_index=0)
        assert loss_long.item() < 1e-4

        # 计算 Top-1 准确率
        for b in range(2):
            for q_idx in range(qpos_long.shape[1]):
                pos = int(qpos_long[b, q_idx].item())
                tgt = int(targets_long[b, q_idx].item())
                pred = int(logits_long[b, pos].argmax(dim=-1).item())
                assert pred == tgt, f"Oracle failed at long seq query {q_idx}: pred {pred} != target {tgt}"


def test_adversarial_benchmark_evaluation_pipeline_with_oracle(device):
    """对抗测试 5: 端到端调用 benchmark_mqar 的 evaluate_mqar 验证其计算 Oracle 获得 100% 准确率与 0.0 损失."""
    configs = [
        MQARConfig(vocab_size=256, seq_len=512, num_kv_pairs=8, num_queries=8),
        MQARConfig(vocab_size=128, seq_len=256, num_kv_pairs=4, num_queries=2),
        MQARConfig(vocab_size=64, seq_len=128, num_kv_pairs=4, num_queries=4, insert_mode="random"),
    ]

    for cfg in configs:
        oracle = MQAROracleModel.from_config(cfg).to(device)
        metrics = evaluate_mqar(
            model=oracle,
            config=cfg,
            device=device,
            eval_batches=5,
            batch_size=4,
        )
        assert metrics["accuracy"] == 1.0, f"evaluate_mqar reported accuracy {metrics['accuracy']} != 1.0"
        assert metrics["loss"] < 1e-4, f"evaluate_mqar reported loss {metrics['loss']} >= 1e-4"
        assert metrics["correct_queries"] == metrics["total_queries"]


def test_adversarial_minimal_boundary_length(device):
    """对抗测试 6: 紧凑极限长度测试 (L = 2K + Q，无任何多余 Filler 空间)."""
    cfg_tight = MQARConfig(vocab_size=64, seq_len=12, num_kv_pairs=4, num_queries=4, insert_mode="uniform")
    X, Y, qpos, targets = generate_mqar_batch(2, cfg_tight, device=device, seed=777)
    assert X.shape == (2, 12)
    assert Y.shape == (2, 12)
    assert qpos.shape == (2, 4)

    oracle = MQAROracleModel.from_config(cfg_tight).to(device)
    with torch.no_grad():
        logits = oracle(X)
        loss = F.cross_entropy(logits.view(-1, cfg_tight.vocab_size), Y.view(-1), ignore_index=0)
        assert loss.item() < 1e-4
