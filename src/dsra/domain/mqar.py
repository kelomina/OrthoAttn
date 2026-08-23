"""Multi-Query Associative Recall (MQAR) 数据集规范、数据生成器与 Ground Truth Oracle 探针.

Multi-Query Associative Recall (MQAR) Domain Specification, Data Generator, and Ground Truth Oracle Probe.

中文说明:
- 调用方 / Called by:
  * `scripts.benchmark_mqar` (MQAR 基准评测脚本)
  * `tests.test_mqar_data_generation` (MQAR 单元与边界测试套件)
- 被调用方 / Callee:
  * `torch` 核心张量与随机数生成算子
  * `torch.nn.Module` 神经网络基类
- 参考来源:
  * Stanford HazyResearch Zoology (ICLR 2024 / `zoology.data.associative_recall`)
  * Google DeepMind Titans (2025)
  * RecurrentGemma
- 核心规范与数学机理:
  1. 词表互斥划分 (Disjoint Vocabulary Partitioning):
     - `[0]`: Padding / Loss Mask (`ignore_index = 0`), 绝不出现在输入序列 X 中;
     - `[1 .. key_end)`: Keys 候选池, 容量为 `key_pool_size`;
     - `[val_start .. val_end)`: Values 候选池, 容量为 `val_pool_size`;
     - `[filler_start .. vocab_size)`: Fillers / Distractors 候选池;
     - 四者满足: {0} ∩ Keys = ∅, Keys ∩ Values = ∅, Values ∩ Fillers = ∅, Keys ∩ Fillers = ∅.
  2. 前缀键值对放置 (Prefix Key-Value Placement):
     - 在序列前半部插入 K 个严格无放回采样的键值对 (k_i, v_i);
     - 局部结构为 X[pos] = k_i, X[pos + 1] = v_i, 其余位置由 Filler 填充.
  3. 后缀自回归 Query 生成与因果掩码 (Autoregressive Query & Causal Alignment):
     - 在序列后半部放置 Q 个打乱重排的 Query Keys (X[qpos] = q_k);
     - 自回归因果对齐: 输入为 X[qpos] 时, 期望输出 Next Token 为对应 Value (Y[qpos] = q_v);
     - 无未来信息泄漏: X[qpos + 1] 为 Filler 或下一 Query, 绝不出现真实 Value;
     - 其余所有非 Query 预测位置 Y 均为 0, 配合 `ignore_index = 0` 进行损失计算与无偏评估.
  4. Ground Truth Oracle 全知探针 (`MQAROracleModel`):
     - 纯因果前缀 KV 查表系统, 严格遵循时间因果律扫描输入序列并在 Query 处输出高置信度 logits,
       达到 100.0% 准确率与 0.0 交叉熵损失, 作为评测流水线真实性与理论上限的金标准.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn


@dataclass(frozen=True)
class MQARConfig:
    """MQAR 基准任务配置规范.

    MQAR Benchmark Task Configuration Specification.

    Attributes:
        vocab_size (int): 词表总大小 V，需 >= 4. Default: 256.
        seq_len (int): 序列总长度 L，需 >= 2 * num_kv_pairs + num_queries. Default: 1024.
        num_kv_pairs (int): 序列前部插入的 (Key, Value) 对数 K，需 >= 1. Default: 8.
        num_queries (Optional[int]): 序列后部查询数 Q，需满足 1 <= Q <= K. 若为 None 则默认 Q = K. Default: None.
        key_pool_size (Optional[int]): Key 候选池容量. 若为 None 则自适应计算. Default: None.
        val_pool_size (Optional[int]): Value 候选池容量. 若为 None 则自适应计算. Default: None.
        insert_mode (str): 插入模式，支持 "uniform" (均匀间隔) 或 "random" (随机非重叠). Default: "uniform".
        device: (Union[str, torch.device]): 运行设备. Default: "cpu".
        seed (Optional[int]): 随机数种子. Default: None.
    """
    vocab_size: int = 256
    seq_len: int = 1024
    num_kv_pairs: int = 8
    num_queries: Optional[int] = None
    key_pool_size: Optional[int] = None
    val_pool_size: Optional[int] = None
    insert_mode: str = "uniform"
    device: Union[str, torch.device] = "cpu"
    seed: Optional[int] = None

    def __post_init__(self):
        """执行全面的参数合法性校验与自适应词表容量计算.

        Validates parameters and computes dynamic pool sizes.
        """
        # 1. 基础尺寸校验
        if self.vocab_size < 4:
            raise ValueError(f"vocab_size must be >= 4 (to accommodate pad, key, val, filler), got {self.vocab_size}")
        if self.num_kv_pairs < 1:
            raise ValueError(f"num_kv_pairs must be >= 1, got {self.num_kv_pairs}")

        # 2. 查询数量校验 (1 <= Q <= K)
        num_q = self.num_queries if self.num_queries is not None else self.num_kv_pairs
        if num_q < 1:
            raise ValueError(f"num_queries must be >= 1, got {num_q}")
        if num_q > self.num_kv_pairs:
            raise ValueError(f"num_queries ({num_q}) cannot exceed num_kv_pairs ({self.num_kv_pairs})")

        # 3. 序列长度校验 (容纳 2K 个 KV token 与 Q 个 Query token)
        min_seq_len = 2 * self.num_kv_pairs + num_q
        if self.seq_len < min_seq_len:
            raise ValueError(
                f"seq_len ({self.seq_len}) is too short for {self.num_kv_pairs} KV pairs and "
                f"{num_q} queries (minimum required is {min_seq_len})"
            )

        # 4. 插入模式校验
        if self.insert_mode not in ("uniform", "random"):
            raise ValueError(f"insert_mode must be 'uniform' or 'random', got '{self.insert_mode}'")

        # 5. 自适应词表池大小分配 (保证 Pad=0, Key池, Val池, Filler池严格互斥且均有效)
        # 默认分配策略: 将非 0 空间的大致 1/4 给 key，1/4 给 val，其余给 filler
        if self.key_pool_size is None:
            default_k_pool = max(1, (self.vocab_size - 2) // 4)
            # 若用户指定的 num_kv_pairs 大于默认池，但在词表容量允许范围内，则自适应扩展
            if self.num_kv_pairs > default_k_pool and (2 * self.num_kv_pairs + 2 <= self.vocab_size):
                k_pool = self.num_kv_pairs
            else:
                k_pool = default_k_pool
        else:
            k_pool = self.key_pool_size

        if self.val_pool_size is None:
            default_v_pool = max(1, (self.vocab_size - 2) // 4)
            if self.num_kv_pairs > default_v_pool and (k_pool + self.num_kv_pairs + 2 <= self.vocab_size):
                v_pool = self.num_kv_pairs
            else:
                v_pool = default_v_pool
        else:
            v_pool = self.val_pool_size

        object.__setattr__(self, "key_pool_size", k_pool)
        object.__setattr__(self, "val_pool_size", v_pool)

        # 6. 池大小与词表容量约束校验
        if self.key_pool_size < 1:
            raise ValueError(f"key_pool_size must be >= 1, got {self.key_pool_size}")
        if self.val_pool_size < 1:
            raise ValueError(f"val_pool_size must be >= 1, got {self.val_pool_size}")

        if self.key_pool_size + self.val_pool_size + 2 > self.vocab_size:
            raise ValueError(
                f"key_pool_size ({self.key_pool_size}) + val_pool_size ({self.val_pool_size}) + 2 "
                f"exceeds vocab_size ({self.vocab_size})"
            )

        if self.num_kv_pairs > self.key_pool_size:
            raise ValueError(
                f"num_kv_pairs ({self.num_kv_pairs}) exceeds key_pool_size ({self.key_pool_size})"
            )
        if self.num_kv_pairs > self.val_pool_size:
            raise ValueError(
                f"num_kv_pairs ({self.num_kv_pairs}) exceeds val_pool_size ({self.val_pool_size})"
            )


def generate_mqar_batch(
    batch_size_or_config: Union[int, MQARConfig] = 16,
    config: Optional[MQARConfig] = None,
    device: Optional[Union[torch.device, str]] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """生成一批严格符合 Stanford Zoology 标准的 MQAR 评测序列.

    Generates a batch of standard Multi-Query Associative Recall (MQAR) sequences.

    中文说明:
    - 调用方 / Called by: `scripts.benchmark_mqar`, `tests.test_mqar_data_generation`
    - 被调用方 / Callee: `torch.randint`, `torch.randperm`, `torch.zeros`
    - 参数 / Args:
        batch_size_or_config: 批大小 (int) 或 MQARConfig 配置对象
        config: MQAR 配置 (当首个参数为 int 时传入)
        device: 生成目标设备 (如 "cuda:0", "cpu", 或 torch.device)
        seed: 独立随机数种子 (使用 torch.Generator 隔离，避免污染全局 RNG 状态)
    - 返回值 / Returns:
        X: [B, L] 输入 token 序列 (LongTensor)
        Y: [B, L] 目标 token 序列 (LongTensor, 仅 query 目标预测位置有值，其余为 0)
        query_positions: [B, Q] Query token 所在时间步索引 (LongTensor)
        target_values: [B, Q] 期望预测的目标 token ID (LongTensor)
    - 错误处理 / Errors:
        若参数不合法或容量不足，由 MQARConfig 校验抛出 ValueError.
    - 副作用 / Side Effects:
        当指定 seed 时，采用独立的 torch.Generator，不改变全局 PyTorch 随机状态.
    """
    # 统一参数解析
    if isinstance(batch_size_or_config, MQARConfig):
        cfg = batch_size_or_config
        batch_size = int(kwargs.get("batch_size", config if isinstance(config, int) else 16))
    else:
        batch_size = int(batch_size_or_config)
        if config is None:
            raise ValueError("MQARConfig must be provided when first argument is batch_size")
        cfg = config

    # 解析目标设备
    target_device = device if device is not None else cfg.device
    if isinstance(target_device, str):
        dev = torch.device(target_device)
    else:
        dev = target_device

    # 随机数生成器隔离
    actual_seed = seed if seed is not None else cfg.seed
    gen = None
    if actual_seed is not None:
        if dev.type == "cuda":
            gen = torch.Generator(device=dev)
            gen.manual_seed(actual_seed)
        else:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(actual_seed)

    L = cfg.seq_len
    K = cfg.num_kv_pairs
    Q = cfg.num_queries if cfg.num_queries is not None else K
    V = cfg.vocab_size
    k_pool = cfg.key_pool_size
    v_pool = cfg.val_pool_size

    # 词表区间计算
    key_start = 1
    key_end = key_start + k_pool
    val_start = key_end
    val_end = val_start + v_pool
    filler_start = val_end
    filler_end = V

    # 1. 默认填充 Filler tokens
    if gen is not None:
        X = torch.randint(filler_start, filler_end, (batch_size, L), generator=gen, device=dev, dtype=torch.long)
    else:
        X = torch.randint(filler_start, filler_end, (batch_size, L), device=dev, dtype=torch.long)

    Y = torch.zeros((batch_size, L), device=dev, dtype=torch.long)

    # 2. 确定前半部分 (KV 空间) 与后半部分 (Query 空间) 的划分点
    if L // 2 >= 2 * K:
        kv_half_end = L // 2
        query_half_start = L // 2
    else:
        kv_half_end = 2 * K
        query_half_start = 2 * K

    q_positions = torch.zeros((batch_size, Q), device=dev, dtype=torch.long)
    target_values = torch.zeros((batch_size, Q), device=dev, dtype=torch.long)

    insert_mode = kwargs.get("insert_mode", cfg.insert_mode)

    # 3. 逐样本生成键值对与查询
    for b in range(batch_size):
        # 3.1 采样不重复的 K 个 keys 和 K 个 values
        if gen is not None:
            perm_keys = torch.randperm(k_pool, generator=gen, device=dev)[:K] + key_start
            perm_vals = torch.randperm(v_pool, generator=gen, device=dev)[:K] + val_start
        else:
            perm_keys = torch.randperm(k_pool, device=dev)[:K] + key_start
            perm_vals = torch.randperm(v_pool, device=dev)[:K] + val_start

        # 3.2 确定 KV 插入位置
        if insert_mode == "uniform":
            avail_kv_slack = kv_half_end - 2 * K
            spacing_kv = avail_kv_slack // (K + 1) if K > 0 else 0
            kv_pos_list = [spacing_kv + i * (2 + spacing_kv) for i in range(K)]
        else:
            # random mode: 在 [0, kv_half_end - 2*K] 内随机采样 K 个增量间隔
            avail_kv_slack = kv_half_end - 2 * K
            if avail_kv_slack > 0:
                if gen is not None:
                    rand_cuts = torch.sort(torch.randint(0, avail_kv_slack + 1, (K,), generator=gen, device=dev))[0]
                else:
                    rand_cuts = torch.sort(torch.randint(0, avail_kv_slack + 1, (K,), device=dev))[0]
                kv_pos_list = [int(rand_cuts[i].item()) + 2 * i for i in range(K)]
            else:
                kv_pos_list = [2 * i for i in range(K)]

        # 插入 KV 对: pos 处放置 key, pos+1 处放置 value
        for i in range(K):
            pos = kv_pos_list[i]
            X[b, pos] = perm_keys[i]
            X[b, pos + 1] = perm_vals[i]

        # 3.3 采样 Queries (从 K 个已有 keys 中无放回选取 Q 个)
        if gen is not None:
            query_perm = torch.randperm(K, generator=gen, device=dev)[:Q]
        else:
            query_perm = torch.randperm(K, device=dev)[:Q]

        q_keys = perm_keys[query_perm]
        q_vals = perm_vals[query_perm]

        # 3.4 确定 Query 插入位置
        query_space = L - query_half_start
        if insert_mode == "uniform":
            avail_q_slack = query_space - Q
            spacing_q = avail_q_slack // (Q + 1) if Q > 0 else 0
            q_pos_list = [query_half_start + spacing_q + j * (1 + spacing_q) for j in range(Q)]
        else:
            avail_q_slack = query_space - Q
            if avail_q_slack > 0:
                if gen is not None:
                    rand_q_cuts = torch.sort(torch.randint(0, avail_q_slack + 1, (Q,), generator=gen, device=dev))[0]
                else:
                    rand_q_cuts = torch.sort(torch.randint(0, avail_q_slack + 1, (Q,), device=dev))[0]
                q_pos_list = [query_half_start + int(rand_q_cuts[j].item()) + j for j in range(Q)]
            else:
                q_pos_list = [query_half_start + j for j in range(Q)]

        for j in range(Q):
            qpos = q_pos_list[j]
            q_positions[b, j] = qpos
            X[b, qpos] = q_keys[j]
            # 严格因果自回归对齐: 在输入为 q_key 时，模型在 qpos 处的 output 预测 Next Token 为 q_val
            Y[b, qpos] = q_vals[j]
            target_values[b, j] = q_vals[j]

    return X, Y, q_positions, target_values


class MQAROracleModel(nn.Module):
    """Ground Truth Oracle 全知探针模型: 纯因果前缀 KV 查表模型.

    Ground Truth Oracle Probe: A Pure Causal Prefix KV Lookup Model.

    中文说明:
    - 调用方 / Called by:
      * `scripts.benchmark_mqar` (MQAR 评测流水线理论上限验证)
      * `tests.test_mqar_data_generation` (Oracle 100% 准确率断言测试)
    - 被调用方 / Callee:
      * `torch.nn.Module`, `torch.zeros`
    - 作用:
      * 作为 MQAR 评测流水线的理论上界与真值探针 (Ground Truth Probe);
      * 验证数据生成器、损失计算掩码 (ignore_index=0) 与准确率统计函数的绝对自洽性;
      * 排除因评测逻辑自身缺陷造成的虚假准确率上限.
    - 因果性保证 (Strict Causality):
      * 在时间步 t，模型仅利用已在前缀 X[b, :t+1] 中观察到的相邻 (k, v) 键值对构建动态记忆表;
      * 当在时间步 t 输入查询 key 时，在对应 value 的 logit 维度赋予超大激活值 (logit_scale=100.0);
      * 绝不访问任何未来时间步 (t' > t) 的 token.
    - 参数 / Args:
        vocab_size: 词表总大小 V
        key_pool_size: Key 候选池大小 (可选)
        val_pool_size: Value 候选池大小 (可选)
        logit_scale: 命中目标 Value 时的 Logit 输出幅值 (默认 100.0)
    - 返回值 / Returns:
        logits: [B, L, vocab_size] 形状的预测张量 (Float32)
    """

    def __init__(
        self,
        vocab_size: int = 256,
        key_pool_size: Optional[int] = None,
        val_pool_size: Optional[int] = None,
        logit_scale: float = 100.0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.key_pool_size = key_pool_size
        self.val_pool_size = val_pool_size
        self.logit_scale = float(logit_scale)
        # 注册空 buffer 以便追踪 .to(device) 等设备迁移操作
        self.register_buffer("_device_tracker", torch.empty(0))

    @classmethod
    def from_config(cls, config: MQARConfig, logit_scale: float = 100.0) -> "MQAROracleModel":
        """基于 MQARConfig 配置快速构建 Oracle 模型实例.

        Creates an MQAROracleModel instance from an MQARConfig.
        """
        return cls(
            vocab_size=config.vocab_size,
            key_pool_size=config.key_pool_size,
            val_pool_size=config.val_pool_size,
            logit_scale=logit_scale,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向纯因果查表推理.

        Forward causal lookup inference.

        参数 / Args:
            input_ids: [B, L] 输入 token 序列 (LongTensor)

        返回 / Returns:
            logits: [B, L, vocab_size] 预测 logit 分布 (FloatTensor)
        """
        B, L = input_ids.shape
        device = input_ids.device
        logits = torch.zeros((B, L, self.vocab_size), device=device, dtype=torch.float32)

        # 计算 Key 与 Value 的有效词表区间 (若未显式指定则按自适应公式计算)
        k_pool = self.key_pool_size if self.key_pool_size is not None else max(1, (self.vocab_size - 2) // 4)
        v_pool = self.val_pool_size if self.val_pool_size is not None else max(1, (self.vocab_size - 2) // 4)
        k_start, k_end = 1, 1 + k_pool
        v_start, v_end = k_end, k_end + v_pool

        # 逐样本严格因果前缀扫描
        for b in range(B):
            seq = input_ids[b].tolist()
            kv_memory: dict[int, int] = {}
            for t in range(L):
                # 1. 因果更新: 若前一步与当前步构成合法的 (k, v) 键值对，则在时间步 t 写入记忆表
                if t >= 1:
                    prev_tok = seq[t - 1]
                    curr_tok = seq[t]
                    if k_start <= prev_tok < k_end and v_start <= curr_tok < v_end:
                        kv_memory[prev_tok] = curr_tok

                # 2. 因果预测: 若当前时间步的输入 token 命中已记忆的 Key，则在输出 logits 的对应 Value 位置赋予高置信度
                current_tok = seq[t]
                if current_tok in kv_memory:
                    val = kv_memory[current_tok]
                    if val < self.vocab_size:
                        logits[b, t, val] = self.logit_scale

        return logits
