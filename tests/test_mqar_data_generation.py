"""Multi-Query Associative Recall (MQAR) 领域数据生成、边界条件与 Oracle 探针单元测试.

Multi-Query Associative Recall (MQAR) Data Generation, Edge Cases, and Ground Truth Oracle Probe Tests.

中文说明:
- 调用方 / Called by: `pytest tests/test_mqar_data_generation.py`
- 作用:
  1. 验证 MQARConfig 的全面边界校验与自适应词表容量计算;
  2. 验证生成张量形状、数据类型、设备放置与词表四路互斥性;
  3. 验证因果键值对放置、无未来信息泄漏、自回归 Next-Token 目标对齐与损失掩码 (ignore_index=0);
  4. 验证随机种子隔离性与插入模式 ("uniform" / "random");
  5. 验证 Ground Truth Oracle 全知探针模型在各配置下达到 100.0% 准确率与 0.0 损失.
"""

import pytest
import torch
import torch.nn.functional as F

from src.dsra.domain.mqar import MQARConfig, MQAROracleModel, generate_mqar_batch


def test_mqar_config_validation_valid():
    """测试 MQARConfig 正常配置与默认参数解析."""
    cfg = MQARConfig(vocab_size=256, seq_len=1024, num_kv_pairs=8)
    assert cfg.vocab_size == 256
    assert cfg.seq_len == 1024
    assert cfg.num_kv_pairs == 8
    assert cfg.num_queries is None
    assert cfg.key_pool_size is not None
    assert cfg.val_pool_size is not None
    assert cfg.key_pool_size >= 8
    assert cfg.val_pool_size >= 8
    assert cfg.key_pool_size + cfg.val_pool_size + 2 <= 256


def test_mqar_config_validation_errors():
    """测试 MQARConfig 各类非法参数抛出预期的 ValueError."""
    # 1. vocab_size < 4
    with pytest.raises(ValueError, match="vocab_size must be >= 4"):
        MQARConfig(vocab_size=3)

    # 2. num_kv_pairs < 1
    with pytest.raises(ValueError, match="num_kv_pairs must be >= 1"):
        MQARConfig(vocab_size=256, num_kv_pairs=0)

    # 3. num_queries < 1
    with pytest.raises(ValueError, match="num_queries must be >= 1"):
        MQARConfig(vocab_size=256, num_kv_pairs=8, num_queries=0)

    # 4. num_queries > num_kv_pairs
    with pytest.raises(ValueError, match="cannot exceed num_kv_pairs"):
        MQARConfig(vocab_size=256, num_kv_pairs=4, num_queries=8)

    # 5. seq_len 太短无法容纳 2K + Q
    with pytest.raises(ValueError, match="is too short"):
        MQARConfig(vocab_size=256, seq_len=10, num_kv_pairs=4, num_queries=4)

    # 6. 非法 insert_mode
    with pytest.raises(ValueError, match="insert_mode must be 'uniform' or 'random'"):
        MQARConfig(vocab_size=256, insert_mode="invalid_mode")

    # 7. 手动指定的 key_pool_size / val_pool_size 超出词表容量
    with pytest.raises(ValueError, match="exceeds vocab_size"):
        MQARConfig(vocab_size=64, key_pool_size=32, val_pool_size=32)

    # 8. 手动指定的 key_pool_size 小于 num_kv_pairs
    with pytest.raises(ValueError, match="exceeds key_pool_size"):
        MQARConfig(vocab_size=256, num_kv_pairs=16, key_pool_size=8)


def test_mqar_dynamic_vocab_scaling():
    """测试小词表 (V=32, 64)、大词表 (V=8192) 及极小词表 (V=4) 的动态词表缩放."""
    # 1. 小词表 V=32
    cfg32 = MQARConfig(vocab_size=32, seq_len=64, num_kv_pairs=4, num_queries=4)
    assert cfg32.key_pool_size >= 4
    assert cfg32.val_pool_size >= 4
    assert cfg32.key_pool_size + cfg32.val_pool_size + 2 <= 32
    X32, Y32, qpos32, targets32 = generate_mqar_batch(2, cfg32, device="cpu", seed=101)
    assert X32.shape == (2, 64)
    assert (X32 < 32).all()
    assert (X32 >= 1).all()

    # 2. 中词表 V=64
    cfg64 = MQARConfig(vocab_size=64, seq_len=128, num_kv_pairs=8, num_queries=8)
    assert cfg64.key_pool_size >= 8
    assert cfg64.val_pool_size >= 8
    X64, Y64, qpos64, targets64 = generate_mqar_batch(2, cfg64, device="cpu", seed=102)
    assert X64.shape == (2, 128)

    # 3. 大词表 V=8192, 支持大容量 KV (如 K=128)
    cfg8192 = MQARConfig(vocab_size=8192, seq_len=2048, num_kv_pairs=128, num_queries=64)
    assert cfg8192.key_pool_size >= 128
    assert cfg8192.val_pool_size >= 128
    X8192, Y8192, qpos8192, targets8192 = generate_mqar_batch(2, cfg8192, device="cpu", seed=103)
    assert X8192.shape == (2, 2048)
    assert qpos8192.shape == (2, 64)
    assert targets8192.shape == (2, 64)

    # 4. 极小词表 V=4, K=1, Q=1, L=3
    cfg4 = MQARConfig(vocab_size=4, seq_len=3, num_kv_pairs=1, num_queries=1)
    X4, Y4, qpos4, targets4 = generate_mqar_batch(1, cfg4, device="cpu", seed=104)
    assert X4.shape == (1, 3)
    # Token 0 为 Pad, 1 为 Key, 2 为 Value, 3 为 Filler
    assert cfg4.key_pool_size == 1
    assert cfg4.val_pool_size == 1


def test_generate_mqar_batch_shapes_and_values():
    """测试 MQAR 批次生成的张量形状、设备、数值范围与目标映射一致性."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = MQARConfig(vocab_size=256, seq_len=512, num_kv_pairs=8, num_queries=8)
    batch_size = 4

    X, Y, qpos, targets = generate_mqar_batch(batch_size, cfg, device=device, seed=42)

    # 1. 检查形状与设备
    assert X.shape == (batch_size, 512)
    assert Y.shape == (batch_size, 512)
    assert qpos.shape == (batch_size, 8)
    assert targets.shape == (batch_size, 8)
    assert X.device.type == device.type
    assert Y.device.type == device.type
    assert qpos.device.type == device.type
    assert targets.device.type == device.type

    # 2. 检查 Query 处与 Targets 的完全一致性
    for b in range(batch_size):
        for q_idx in range(8):
            pos = int(qpos[b, q_idx].item())
            expected_val = int(targets[b, q_idx].item())
            assert int(Y[b, pos].item()) == expected_val
            assert expected_val > 0

    # 3. 检查非 Query 处 Target 全为 0 (ignore_index=0)
    for b in range(batch_size):
        mask = torch.ones(512, dtype=torch.bool, device=device)
        mask[qpos[b]] = False
        assert (Y[b, mask] == 0).all().item()


def test_vocabulary_partitioning_disjointness():
    """测试词表四路互斥划分: [0]=Pad, Keys, Values, Fillers 绝无交集."""
    cfg = MQARConfig(vocab_size=256, seq_len=512, num_kv_pairs=12, num_queries=12)
    k_pool = cfg.key_pool_size
    v_pool = cfg.val_pool_size
    key_start = 1
    key_end = key_start + k_pool
    val_start = key_end
    val_end = val_start + v_pool
    filler_start = val_end
    filler_end = cfg.vocab_size

    # 集合互斥检查
    keys_set = set(range(key_start, key_end))
    vals_set = set(range(val_start, val_end))
    fillers_set = set(range(filler_start, filler_end))

    assert 0 not in keys_set
    assert 0 not in vals_set
    assert 0 not in fillers_set
    assert keys_set.isdisjoint(vals_set)
    assert vals_set.isdisjoint(fillers_set)
    assert keys_set.isdisjoint(fillers_set)
    assert len(keys_set) + len(vals_set) + len(fillers_set) + 1 == cfg.vocab_size

    X, Y, qpos, targets = generate_mqar_batch(4, cfg, device="cpu", seed=2026)

    # 检查输入序列 X 中绝不含有 Pad Token (0)
    assert (X != 0).all().item()

    # 检查 targets 严格落在 Values 集合内
    for val in targets.view(-1).tolist():
        assert val in vals_set


def test_causal_key_value_placement_and_zero_future_leakage():
    """测试因果时间顺序: 前缀 KV 放置在查询之前，查询后无答案 prompt 泄漏."""
    cfg = MQARConfig(vocab_size=256, seq_len=512, num_kv_pairs=8, num_queries=8)
    X, Y, qpos, targets = generate_mqar_batch(4, cfg, device="cpu", seed=888)

    k_start = 1
    k_end = 1 + cfg.key_pool_size
    v_start = k_end
    v_end = k_end + cfg.val_pool_size

    for b in range(4):
        seq = X[b].tolist()
        # 提取前缀中出现的 (k, v) 对
        prefix_kv = {}
        first_qpos = int(qpos[b, 0].item())
        for t in range(first_qpos):
            if t + 1 < first_qpos:
                k, v = seq[t], seq[t + 1]
                if k_start <= k < k_end and v_start <= v < v_end:
                    prefix_kv[k] = v

        # 检查所有查询 key 都在前缀中预先出现，且查询后的下一个 token 不是对应 value
        for q_idx in range(8):
            pos = int(qpos[b, q_idx].item())
            q_key = seq[pos]
            expected_val = int(targets[b, q_idx].item())
            assert q_key in prefix_kv
            assert prefix_kv[q_key] == expected_val

            # 防泄漏检查: X[pos + 1] 绝不能是期望的 value (不能泄露答案作为后续 prompt)
            if pos + 1 < cfg.seq_len:
                assert seq[pos + 1] != expected_val or seq[pos + 1] not in range(v_start, v_end)


def test_insert_mode_uniform_and_random():
    """测试 uniform 与 random 两种插入模式的正确性与方差多样性."""
    cfg_uni = MQARConfig(vocab_size=256, seq_len=512, num_kv_pairs=8, insert_mode="uniform")
    cfg_rnd = MQARConfig(vocab_size=256, seq_len=512, num_kv_pairs=8, insert_mode="random")

    X_uni, Y_uni, qpos_uni, _ = generate_mqar_batch(4, cfg_uni, device="cpu", seed=1)
    X_rnd, Y_rnd, qpos_rnd, _ = generate_mqar_batch(4, cfg_rnd, device="cpu", seed=1)

    # uniform 模式下所有样本的 query 位置应相同
    for b in range(1, 4):
        assert (qpos_uni[b] == qpos_uni[0]).all()

    # random 模式下不同样本的 query 位置应呈现抖动差异
    has_variance = False
    for b in range(1, 4):
        if not (qpos_rnd[b] == qpos_rnd[0]).all():
            has_variance = True
            break
    assert has_variance


def test_device_flexibility_and_string_argument():
    """测试 device 参数对 str 与 torch.device 的兼容支持."""
    cfg = MQARConfig(vocab_size=128, seq_len=256, num_kv_pairs=4)

    # 1. 字符串 'cpu'
    X_cpu, Y_cpu, _, _ = generate_mqar_batch(2, cfg, device="cpu")
    assert X_cpu.device.type == "cpu"

    # 2. torch.device('cpu')
    X_dev, Y_dev, _, _ = generate_mqar_batch(2, cfg, device=torch.device("cpu"))
    assert X_dev.device.type == "cpu"

    # 3. 若有 CUDA 则测试 'cuda:0'
    if torch.cuda.is_available():
        X_cuda, Y_cuda, _, _ = generate_mqar_batch(2, cfg, device="cuda:0")
        assert X_cuda.device.type == "cuda"
        assert X_cuda.device.index == 0


def test_generator_seed_reproducibility():
    """测试使用 seed 参数时的数据生成完全可复现且隔离."""
    cfg = MQARConfig(vocab_size=256, seq_len=512, num_kv_pairs=8)
    X1, Y1, qpos1, tgt1 = generate_mqar_batch(2, cfg, device="cpu", seed=12345)
    X2, Y2, qpos2, tgt2 = generate_mqar_batch(2, cfg, device="cpu", seed=12345)
    X3, Y3, qpos3, tgt3 = generate_mqar_batch(2, cfg, device="cpu", seed=54321)

    assert torch.equal(X1, X2)
    assert torch.equal(Y1, Y2)
    assert torch.equal(qpos1, qpos2)
    assert torch.equal(tgt1, tgt2)
    assert not torch.equal(X1, X3)


def test_mqar_oracle_model_100_percent_accuracy_and_zero_loss():
    """测试 Ground Truth Oracle 全知探针在多种尺度与配置下均达到 100.0% 准确率与 0.0 损失."""
    test_configs = [
        MQARConfig(vocab_size=64, seq_len=128, num_kv_pairs=4, num_queries=4),
        MQARConfig(vocab_size=256, seq_len=512, num_kv_pairs=8, num_queries=8),
        MQARConfig(vocab_size=1024, seq_len=1024, num_kv_pairs=16, num_queries=16),
        MQARConfig(vocab_size=128, seq_len=256, num_kv_pairs=8, num_queries=4),  # Q < K
        MQARConfig(vocab_size=32, seq_len=64, num_kv_pairs=2, num_queries=2),    # 小词表
    ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for cfg in test_configs:
        oracle = MQAROracleModel.from_config(cfg).to(device)
        oracle.eval()

        batch_size = 4
        X, Y, qpos, targets = generate_mqar_batch(batch_size, cfg, device=device, seed=4242)

        with torch.no_grad():
            logits = oracle(X)  # [B, L, V]

            # 1. 计算交叉熵损失 (ignore_index=0)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), Y.view(-1), ignore_index=0)
            assert loss.item() < 1e-4, f"Oracle loss={loss.item()} is not ~0.0 for config {cfg}"

            # 2. 计算 Top-1 准确率
            total_queries = 0
            correct_queries = 0
            for b in range(batch_size):
                for q_idx in range(qpos.shape[1]):
                    pos = int(qpos[b, q_idx].item())
                    expected = int(targets[b, q_idx].item())
                    pred = int(logits[b, pos].argmax(dim=-1).item())
                    if pred == expected:
                        correct_queries += 1
                    total_queries += 1

            acc = correct_queries / total_queries
            assert acc == 1.0, f"Oracle accuracy={acc} is not 100.0% for config {cfg}"
