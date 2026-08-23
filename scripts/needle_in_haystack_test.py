import argparse
import gc
import json
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import random
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dsra.dsra_model import MultiLayerMHDSRA2Model  # noqa: E402
from src.dsra.domain import normalize_model_type  # noqa: E402
from src.dsra.report_utils import build_capacity_markdown, ensure_reports_dir, write_json, write_markdown  # noqa: E402
from src.dsra.swanlab_utils import init_swanlab, SwanLabRunProxy  # noqa: E402

DEFAULT_SEQ_LENGTHS = [
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
    2097152,
]

CAPACITY_TEST_LENGTHS = [
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
    2097152,
]

PAD_TOKEN_ID = 0
QUERY_TOKEN_ID = 1
NEEDLE_KEY_TOKEN_ID = 2
FILLER_TOKEN_START = 4
NIAH_DEPTHS = (0.1, 0.5, 0.9)
NIAH_EVAL_INTERVAL = 20
DEFAULT_NIAH_EVAL_BATCHES_PER_DEPTH = 32
DEFAULT_NIAH_LIGHT_EVAL_BATCHES_PER_DEPTH = 1
DEFAULT_NIAH_ROBUST_EVAL_BATCHES_PER_DEPTH = 32
DEFAULT_NIAH_TEST_BATCHES_PER_DEPTH = 0
DEFAULT_NIAH_CAPACITY_BATCHES_PER_DEPTH = 3
NIAH_MIN_EVAL_SAMPLES_FOR_EARLY_STOP = 24
NIAH_TEST_SEED_OFFSET = 1_000_003
INVALID_REPORT_NAME_CHARS = set('/\\:<>"|?*')
NIAH_CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}
DEFAULT_NIAH_READOUT_MODE = "model"
ALLOWED_NIAH_READOUT_MODES = (
    "model",
    "needle_copy",
    "needle_pair_copy",
    "span_predictor",
)
DEFAULT_NIAH_SPAN_CANDIDATE_FILTER = "all"
ALLOWED_NIAH_SPAN_CANDIDATE_FILTERS = (
    "all",
    "key_value_pair",
    "prefer_key_value_pair",
)
DEFAULT_NIAH_SPAN_LOSS_MODE = "single_positive"
ALLOWED_NIAH_SPAN_LOSS_MODES = ("single_positive", "multi_positive")
NIAH_SPAN_CANDIDATE_DIAGNOSTIC_FIELDS = (
    "mean_span_valid_candidate_count",
    "mean_span_raw_candidate_count",
    "mean_span_pair_candidate_count",
    "span_pair_available_rate",
    "span_filter_fallback_rate",
    "span_selected_pair_rate",
    "span_target_pair_candidate_rate",
    "span_target_value_page_candidate_rate",
    "span_target_pair_page_candidate_rate",
    "span_target_value_top_token_rate",
    "span_target_pair_top_token_rate",
    "span_target_value_seed_token_rate",
    "span_target_pair_seed_token_rate",
)
DEFAULT_RETRIEVAL_PROJECTION_CONTRASTIVE_TEMPERATURE = 0.10


def resolve_device(device_name):
    """Resolve NIAH CLI device names into a torch device.

    中文说明:
    - 调用方 / Called by: CLI entrypoint and report verification helpers
    - 调用对象 / Calls: `torch.cuda.is_available`, `torch.device`
    - 作用 / Purpose: 统一解析 `auto/cpu/cuda/cuda:0`，避免脚本入口散落设备选择逻辑
    - 参数 / Parameters: `device_name` 是命令行传入的设备名称，默认推荐 `auto`
    - 返回 / Returns: `torch.device`
    - 接入 / Integration: NIAH CLI 与手动验证入口都应复用本函数
    - 错误处理 / Error handling: `cuda` 不可用但显式请求 CUDA 时抛出 `RuntimeError`
    - 副作用 / Side effects: 无
    - 事务边界 / Transaction boundary: 不涉及事务
    - 并发与幂等 / Concurrency and idempotency: 纯解析函数，可重复调用
    - 关键词 / Keywords:
      device|auto|cuda|cpu|niah|cli|torch|gpu|resolve|设备

    English documentation:
    Function name:
        resolve_device
    Purpose:
        Resolve NIAH CLI device names into a concrete torch device.
    Called by:
        CLI and report verification helpers.
    Calls:
        `torch.cuda.is_available`, `torch.device`.
    Parameters:
        - device_name: requested device string.
    Returns:
        `torch.device`.
    Error handling:
        Raises `RuntimeError` when CUDA is explicitly requested but unavailable.
    Side effects:
        None.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Deterministic for the same runtime environment.
    English keywords:
        device, auto, cuda, cpu, niah, cli, torch, gpu, resolve, runtime
    """
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but torch.cuda.is_available() is false: {device_name}")
    return torch.device(device_name)


def seed_all(seed, *, cudnn_benchmark=False):
    """Seed Python and PyTorch RNGs for reproducible NIAH reports.

    中文说明:
    - 调用方 / Called by: `run_niah_verification_case`, `run_niah_test`,
      `run_niah_capacity_test`, CLI entrypoint
    - 调用对象 / Calls: `random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`
    - 作用 / Purpose: 固化 NIAH 报告中的随机 haystack、query position、needle value
      和模型初始化；需要生产性能口径时允许显式打开 cuDNN benchmark
    - 参数 / Parameters: `seed` 是整数随机种子；`cudnn_benchmark` 控制 cuDNN 自动调优
    - 返回 / Returns: None
    - 接入 / Integration: 生成 reports/ 交付物前调用，保证结果可复现
    - 错误处理 / Error handling: CUDA 不可用时跳过 CUDA seed，不抛出
    - 副作用 / Side effects: 修改 RNG 全局状态
    - 事务边界 / Transaction boundary: 不涉及事务
    - 并发与幂等 / Concurrency and idempotency: 重复设置同一 seed 会重置随机序列
    - 关键词 / Keywords:
      seed|reproducible|niah|torch|cuda|random|report|benchmark|deterministic|随机

    English documentation:
    Function name:
        seed_all
    Purpose:
        Seed Python and PyTorch RNGs for reproducible NIAH reports, while allowing
        callers to opt into cuDNN benchmark mode for production-performance runs.
    Called by:
        Verification case runner, legacy NIAH sweeps, capacity sweeps, and CLI entrypoint.
    Calls:
        Python random and torch seed APIs.
    Parameters:
        - seed: integer seed.
        - cudnn_benchmark: whether to enable cuDNN autotuning.
    Returns:
        None.
    Error handling:
        Skips CUDA seeding when CUDA is unavailable.
    Side effects:
        Mutates global RNG state.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Re-seeding repeats the random sequence.
    English keywords:
        seed, reproducible, niah, torch, cuda, random, report, benchmark, deterministic, rng
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = not cudnn_benchmark
        torch.backends.cudnn.benchmark = cudnn_benchmark
    if not cudnn_benchmark:
        torch.use_deterministic_algorithms(True, warn_only=True)


def capture_niah_rng_state(device):
    """Capture RNG state before temporary held-out NIAH evaluation.

    中文说明:
    - 调用方 / Called by: `run_niah_verification_case`
    - 作用 / Purpose: 在独立 test eval 临时重置随机种子前保存 Python、CPU torch
      和 CUDA RNG 状态，避免 held-out 评分污染后续诊断随机流。
    - 返回 / Returns: 可传给 `restore_niah_rng_state()` 的状态字典。

    English documentation:
    Capture Python, CPU torch, and optional CUDA RNG state before held-out test eval.
    """
    return {
        "python": random.getstate(),
        "torch": torch.random.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if device.type == "cuda" and torch.cuda.is_available()
            else None
        ),
    }


def restore_niah_rng_state(state):
    """Restore RNG state after temporary held-out NIAH evaluation.

    中文说明:
    - 调用方 / Called by: `run_niah_verification_case`
    - 作用 / Purpose: 恢复 `capture_niah_rng_state()` 保存的随机状态，让 test eval
      只影响 test 指标，不影响后续 final diagnostics 或其它随机流程。
    - 返回 / Returns: None。

    English documentation:
    Restore Python, CPU torch, and optional CUDA RNG state after held-out test eval.
    """
    random.setstate(state["python"])
    torch.random.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def is_oom_error(exc):
    """Return whether an exception message is likely a CUDA/accelerator OOM.

    中文说明:
    - 调用方 / Called by: `run_niah_capacity_test`, `run_niah_test`,
      `scripts.next_round_benchmark_runner.run_niah_section`
    - 调用对象 / Calls: `str.lower`, `isinstance`
    - 作用 / Purpose: 将 CUDA/accelerator 内存失败归一为可跳过的 OOM 结果，
      同时避免把普通 CUDA memory 诊断或 memory-leak 文本误判为 OOM
    - 参数 / Parameters: `exc` 是捕获到的异常对象
    - 返回 / Returns: bool；疑似内存不足时为 true
    - 内部关键变量 / Internal variables: `message` 是小写后的异常文本
    - 接入 / Integration: 仅用于 benchmark 容错分支，不应包裹普通训练错误
    - 错误处理 / Error handling: 本函数不抛出自定义错误，无法识别时返回 false
    - 副作用 / Side effects: 无
    - 事务边界 / Transaction boundary: 不涉及 Unit of Work、数据库或文件事务
    - 并发与幂等 / Concurrency and idempotency: 纯字符串判断，可重复调用
    - 关键词 / Keywords:
      oom|cuda|memory|accelerator|runtimeerror|benchmark|skip|guard|错误|内存

    English documentation:
    Function name:
        is_oom_error
    Purpose:
        Detect likely CUDA or accelerator out-of-memory errors without masking
        unrelated runtime failures.
    Called by:
        NIAH capacity/sweep runners and next-round benchmark runner.
    Calls:
        `str.lower` and `isinstance`.
    Parameters:
        - exc: caught exception object.
    Returns:
        Boolean OOM classification.
    Error handling:
        Returns false for unrecognized messages.
    Side effects:
        None.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Pure and deterministic for the same message.
    English keywords:
        oom, cuda, memory, accelerator, runtimeerror, benchmark, skip, guard, error, memory
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    oom_patterns = (
        "out of memory",
        "not enough memory",
        "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed",
        "cuda error: memory allocation",
        "hip error: memory allocation",
        "mps backend out of memory",
    )
    return any(pattern in message for pattern in oom_patterns)


def resolve_niah_run_seed(seed):
    """Resolve an optional NIAH sweep seed into a traceable integer seed.

    中文说明:
    - 调用方 / Called by: `run_niah_test`, `run_niah_capacity_test`
    - 调用对象 / Calls: `time.time`, `int`
    - 作用 / Purpose: 让 legacy sweep 在未显式传 seed 时仍记录本轮运行使用的随机种子，
      避免 haystack/query/model 初始化不可追溯
    - 参数 / Parameters: `seed` 是可选整数；None 时使用当前 Unix 时间秒
    - 返回 / Returns: int，供 `seed_all` 使用并写入/打印到结果
    - 内部关键变量 / Internal variables: `resolved_seed` 是本轮 sweep 的唯一种子口径
    - 接入 / Integration: 仅用于脚本层 sweep，不应在单 batch 生成函数内隐式重置 RNG
    - 错误处理 / Error handling: 非整数兼容 `int(seed)` 转换，失败时抛出原始异常
    - 副作用 / Side effects: None；不直接修改 RNG
    - 事务边界 / Transaction boundary: 不涉及 Unit of Work、数据库或文件事务
    - 并发与幂等 / Concurrency and idempotency: 显式 seed 可复现；None seed 只保证可追溯
    - 关键词 / Keywords:
      seed|time|traceable|niah|sweep|capacity|reproducible|rng|legacy|种子

    English documentation:
    Function name:
        resolve_niah_run_seed
    Purpose:
        Convert an optional NIAH sweep seed into a traceable integer seed.
    Called by:
        `run_niah_test` and `run_niah_capacity_test`.
    Calls:
        `time.time` and `int`.
    Parameters:
        - seed: optional integer seed; when None, current Unix seconds are used.
    Returns:
        Integer seed for `seed_all` and result metadata.
    Error handling:
        Propagates conversion errors for invalid seed inputs.
    Side effects:
        None.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Explicit seeds are reproducible; time-derived seeds are traceable only.
    English keywords:
        seed, time, traceable, niah, sweep, capacity, reproducible, rng, legacy, random
    """
    return int(time.time()) if seed is None else int(seed)


def normalize_niah_readout_mode(value=DEFAULT_NIAH_READOUT_MODE):
    """Normalize the optional NIAH evaluation readout mode.

    中文说明:
    - 调用方 / Called by: `evaluate_niah_depths`, `run_niah_verification_case`
    - 调用对象 / Calls: 无。
    - 作用 / Purpose: 默认保留原始 query logits 分类评估；显式选择
      `needle_copy` 时，评估会从模型预测召回候选中复制 needle value；显式选择
      `needle_pair_copy` 时，评估会优先寻找候选里的 needle key/right-neighbor value 成对结构；
      `span_predictor` 则只在实验模型创建候选 span 小头时可用，用模型打分选择候选。
    - 参数 / Parameters: `value` 是 readout 模式字符串或 None。
    - 返回 / Returns: 规范化后的模式名。
    - 错误处理 / Error handling: 非法模式抛 `ValueError`，避免静默跑错实验。
    - 副作用 / Side effects: 无。
    - 关键词 / Keywords:
      niah|readout|needle_copy|default|validation|mode|copy|评估

    English documentation:
    Function name:
        normalize_niah_readout_mode
    Purpose:
        Keep the default query-logit evaluation unchanged while making the
        experimental needle-copy and pair-copy readouts explicit and validated.
    """
    mode = str(value or DEFAULT_NIAH_READOUT_MODE).strip()
    if mode not in ALLOWED_NIAH_READOUT_MODES:
        allowed = ", ".join(ALLOWED_NIAH_READOUT_MODES)
        raise ValueError(f"Unsupported niah_readout_mode: {mode!r}; allowed: {allowed}")
    return mode


def normalize_niah_span_candidate_filter(value=DEFAULT_NIAH_SPAN_CANDIDATE_FILTER):
    """Normalize the opt-in span predictor candidate filter.

    中文说明:
    - 调用方 / Called by: `run_niah_verification_case`, `evaluate_niah_depths`,
      span predictor loss/readout helpers.
    - 调用对象 / Calls: 无。
    - 作用 / Purpose: 默认 `all` 保持旧 span predictor 候选行为；显式
      `key_value_pair` 时，只允许模型已经召回到 needle key 与右侧 value 成对结构的候选
      进入 span predictor 训练和读出；显式 `prefer_key_value_pair` 时优先使用成对候选，
      没有成对候选则回退到旧候选池，减少宽候选池噪声但避免 readout 可用率直接归零。
    - 参数 / Parameters: `value` 是候选过滤模式字符串或 None。
    - 返回 / Returns: 规范化后的过滤模式名。
    - 错误处理 / Error handling: 非法模式抛 `ValueError`，避免静默跑错实验。
    - 副作用 / Side effects: 无。
    - 关键词 / Keywords:
      niah|span_predictor|candidate_filter|key_value_pair|retrieval|候选过滤
    """
    mode = str(value or DEFAULT_NIAH_SPAN_CANDIDATE_FILTER).strip()
    if mode not in ALLOWED_NIAH_SPAN_CANDIDATE_FILTERS:
        allowed = ", ".join(ALLOWED_NIAH_SPAN_CANDIDATE_FILTERS)
        raise ValueError(
            f"Unsupported niah_span_candidate_filter: {mode!r}; allowed: {allowed}"
        )
    return mode


def normalize_niah_span_loss_mode(value=DEFAULT_NIAH_SPAN_LOSS_MODE):
    """Normalize the opt-in span predictor loss mode.

    中文说明:
    - 调用方 / Called by: `compute_retrieval_span_predictor_loss`,
      `run_niah_verification_case`。
    - 调用对象 / Calls: 无。
    - 作用 / Purpose: 默认 `single_positive` 保持旧行为，只奖励第一个正例；
      显式 `multi_positive` 时把所有能复制正确 token 的候选都视为正例，避免把其它
      正确候选当成 softmax 负例。
    - 参数 / Parameters: `value` 是 loss 模式字符串或 None。
    - 返回 / Returns: 规范化后的 loss 模式名。
    - 错误处理 / Error handling: 非法模式抛 `ValueError`，避免静默跑错实验。
    - 副作用 / Side effects: 无。
    - 关键词 / Keywords:
      niah|span_predictor|loss_mode|multi_positive|retrieval|多正例
    """
    mode = str(value or DEFAULT_NIAH_SPAN_LOSS_MODE).strip()
    if mode not in ALLOWED_NIAH_SPAN_LOSS_MODES:
        allowed = ", ".join(ALLOWED_NIAH_SPAN_LOSS_MODES)
        raise ValueError(f"Unsupported niah_span_loss_mode: {mode!r}; allowed: {allowed}")
    return mode


def cleanup_after_oom():
    """Release Python and CUDA cached memory after an OOM path.

    中文说明:
    - 调用方 / Called by: `run_niah_test`, `run_niah_capacity_test`
    - 调用对象 / Calls: `gc.collect`, `torch.cuda.empty_cache`
    - 作用 / Purpose: 将昂贵的 cache 清理限制到 OOM 恢复路径，避免训练循环每步清理导致
      benchmark 性能失真
    - 参数 / Parameters: None
    - 返回 / Returns: None
    - 内部关键变量 / Internal variables: 无
    - 接入 / Integration: 仅在捕获 OOM 后调用；常规训练循环不应调用本函数
    - 错误处理 / Error handling: CUDA 不可用时跳过 CUDA cache 清理
    - 副作用 / Side effects: 触发 Python GC，并释放 CUDA caching allocator 的空闲块
    - 事务边界 / Transaction boundary: 不涉及 Unit of Work、数据库或文件事务
    - 并发与幂等 / Concurrency and idempotency: 重复调用安全，但会影响性能计时
    - 关键词 / Keywords:
      oom|gc|empty_cache|cuda|cleanup|benchmark|performance|memory|恢复|显存

    English documentation:
    Function name:
        cleanup_after_oom
    Purpose:
        Release Python and CUDA cached memory only after OOM recovery paths.
    Called by:
        `run_niah_test` and `run_niah_capacity_test`.
    Calls:
        `gc.collect` and `torch.cuda.empty_cache`.
    Parameters:
        None.
    Returns:
        None.
    Error handling:
        Skips CUDA cache cleanup when CUDA is unavailable.
    Side effects:
        Runs Python GC and clears free CUDA cache blocks.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Repeatable but expensive; do not use inside timed training loops.
    English keywords:
        oom, gc, empty_cache, cuda, cleanup, benchmark, performance, memory, recovery, gpu
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_niah_depth_for_optimizer_step(optimizer_step, depths=NIAH_DEPTHS):
    """Select a deterministic round-robin NIAH depth for one optimizer step.

    中文说明:
    - 调用方 / Called by: `run_single_niah_test`, `run_niah_verification_case`
    - 调用对象 / Calls: tuple indexing
    - 作用 / Purpose: 用确定性轮询替代随机 depth 选择，确保每个 depth 均匀进入训练流程；
      本项目的 NIAH CLI 每轮循环都会执行一次 `optimizer.step()`，因此真实语义是 step 而非 epoch
    - 参数 / Parameters: `optimizer_step` 是从 0 开始的优化器步数；`depths` 是非空 depth 序列
    - 返回 / Returns: float depth ratio
    - 内部关键变量 / Internal variables: `depth_values` 是不可变 depth 序列
    - 接入 / Integration: 新增 NIAH 训练入口时应复用该函数，避免重新引入随机 depth 调度
    - 错误处理 / Error handling: `depths` 为空时抛出 `ValueError`
    - 副作用 / Side effects: 无；不消耗随机数
    - 事务边界 / Transaction boundary: 不涉及 Unit of Work、数据库或文件事务
    - 并发与幂等 / Concurrency and idempotency: 纯函数，同一 optimizer_step/depths 结果稳定
    - 关键词 / Keywords:
      depth|round_robin|deterministic|niah|optimizer_step|schedule|reproducible|train|轮询|深度

    English documentation:
    Function name:
        get_niah_depth_for_optimizer_step
    Purpose:
        Select NIAH depth by deterministic round-robin scheduling for one optimizer step.
    Called by:
        `run_single_niah_test` and `run_niah_verification_case`.
    Calls:
        Tuple indexing.
    Parameters:
        - optimizer_step: zero-based optimizer step index.
        - depths: non-empty sequence of depth ratios.
    Returns:
        Depth ratio as a float.
    Error handling:
        Raises `ValueError` when no depths are configured.
    Side effects:
        None; does not consume RNG state.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Pure and deterministic.
    English keywords:
        depth, round_robin, deterministic, niah, optimizer_step, schedule, reproducible, train, polling, depth
    """
    depth_values = tuple(depths)
    if not depth_values:
        raise ValueError("depths must not be empty")
    return depth_values[optimizer_step % len(depth_values)]


def get_niah_depth_for_epoch(epoch, depths=NIAH_DEPTHS):
    """Compatibility wrapper for historical epoch-named NIAH depth scheduling.

    中文说明:
    - 调用方 / Called by: legacy tests and historical callers that still pass an `epoch` name
    - 调用对象 / Calls: `get_niah_depth_for_optimizer_step`
    - 作用 / Purpose: 保持旧函数名可用，同时把真实语义委托给 optimizer step 调度函数
    - 参数 / Parameters: `epoch` 是历史命名，实际表示从 0 开始的 optimizer step
    - 返回 / Returns: float depth ratio
    - 接入 / Integration: 新代码应调用 `get_niah_depth_for_optimizer_step`
    - 错误处理 / Error handling: 透传空 depths 的 `ValueError`
    - 副作用 / Side effects: 无
    - 事务边界 / Transaction boundary: 不涉及事务
    - 并发与幂等 / Concurrency and idempotency: 纯函数
    - 关键词 / Keywords:
      compatibility|epoch|optimizer_step|depth|schedule|niah|legacy|alias|兼容|命名

    English documentation:
    Function name:
        get_niah_depth_for_epoch
    Purpose:
        Preserve the historical epoch-named API while delegating to optimizer-step scheduling.
    Called by:
        Legacy tests and historical callers.
    Calls:
        `get_niah_depth_for_optimizer_step`.
    Parameters:
        - epoch: historical name for a zero-based optimizer step.
        - depths: non-empty sequence of depth ratios.
    Returns:
        Depth ratio as a float.
    Error handling:
        Propagates `ValueError` from the step scheduler.
    Side effects:
        None.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Pure and deterministic.
    English keywords:
        compatibility, epoch, optimizer_step, depth, schedule, niah, legacy, alias, naming, migration
    """
    return get_niah_depth_for_optimizer_step(epoch, depths)

def generate_haystack_with_needle(batch_size, seq_len, vocab_size, needle_depth_ratio=0.5):
    """Generate CPU-side NIAH token ids with a single supervised query position.

    Returns (X, Y, needle_positions): needle_positions are the token-2 anchor
    positions for each sample, usable to compute an auxiliary loss at the needle
    value position (needle_positions + 1).

    中文说明:
    - 调用方 / Called by: `run_single_niah_test`, `run_single_niah_capacity_test`, tests
    - 调用对象 / Calls: `torch.randint`, `warnings.warn`, tensor assignment
    - 作用 / Purpose: 生成 Needle-In-A-Haystack 输入；固定 query/key token，但为每个样本
      随机生成 needle value 和 needle 位置（在目标 depth_ratio ±5% 范围内随机），
      并在 needle 之后随机放置 query/key pair，避免模型只学习固定答案、固定位置或
      固定末尾查询位置而伪造检索成功；使用向量化随机填充避免 2M token 场景下
      逐 token Python 循环成为瓶颈
    - 参数 / Parameters:
      `batch_size` 是样本数；`seq_len` 是上下文长度；`vocab_size` 必须大于 4；
      `needle_depth_ratio` 控制 needle 在上下文中的相对深度
    - 返回 / Returns: `(X, Y)` CPU long tensors；`Y` 只在查询位置包含目标值，其余为 PAD
    - 内部关键变量 / Internal variables:
      `needle_values` 是每个样本独立随机目标值；`needle_positions` 是每个样本的插入位置；
      `query_key_positions/query_positions` 是随机查询入口位置
    - 接入 / Integration: 该函数只负责数据构造，不负责设备搬运或模型调用
    - 错误处理 / Error handling: `seq_len` 太短或 `vocab_size` 太小会抛出 `ValueError`
    - 副作用 / Side effects: 消耗 PyTorch RNG；不写文件、不访问网络
    - 事务边界 / Transaction boundary: 不涉及 Unit of Work 或持久化事务
    - 并发与幂等 / Concurrency and idempotency: 依赖当前 RNG 状态；固定 seed 后可复现
    - 关键词 / Keywords:
      niah|needle|haystack|variable_value|2m|query_token|target|cpu|memory|生成

    English documentation:
    Function name:
        generate_haystack_with_needle
    Purpose:
        Build CPU-side Needle-In-A-Haystack tensors with per-sample random values
        and randomized post-needle query positions, so the benchmark cannot be
        solved by always predicting one fixed token or attending to the final token.
    Called by:
        NIAH train and capacity-test entry points.
    Calls:
        `torch.randint`, `warnings.warn`, and tensor assignment.
    Parameters:
        - batch_size: number of samples.
        - seq_len: context length.
        - vocab_size: vocabulary size, must be greater than 4.
        - needle_depth_ratio: relative needle depth before the final query.
    Returns:
        `(X, Y)` CPU long tensors with one supervised query target per sample.
    Error handling:
        Raises `ValueError` for invalid sequence or vocabulary sizes.
    Side effects:
        Advances the PyTorch RNG state only.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Reproducible when the caller fixes the random seed.
    English keywords:
        niah, needle, haystack, variable_value, 2m, query_token, target, cpu, memory, generation
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if vocab_size <= FILLER_TOKEN_START:
        raise ValueError("vocab_size must be greater than 4 so value/filler tokens are available")
    if vocab_size < 10:
        warnings.warn(
            "NIAH vocab_size is below 10; the small answer space can make retrieval accuracy easier to saturate.",
            RuntimeWarning,
            stacklevel=2,
        )
    if seq_len < 6:
        raise ValueError("seq_len must be at least 6 to place a needle and final query")
    if not 0.0 <= needle_depth_ratio <= 1.0:
        raise ValueError("needle_depth_ratio must be within [0.0, 1.0]")

    X = torch.randint(FILLER_TOKEN_START, vocab_size, (batch_size, seq_len), dtype=torch.long)
    Y = torch.full((batch_size, seq_len), PAD_TOKEN_ID, dtype=torch.long)

    max_needle_pos = seq_len - 5
    center = int(max_needle_pos * needle_depth_ratio)
    half_window = max(1, max_needle_pos // 20)
    lo = max(0, center - half_window)
    hi = min(max_needle_pos, center + half_window) + 1
    needle_positions = torch.randint(lo, hi, (batch_size,), dtype=torch.long)
    batch_indices = torch.arange(batch_size, dtype=torch.long)
    needle_values = torch.randint(
        FILLER_TOKEN_START,
        vocab_size,
        (batch_size,),
        dtype=torch.long,
    )

    X[batch_indices, needle_positions] = NEEDLE_KEY_TOKEN_ID
    X[batch_indices, needle_positions + 1] = needle_values
    min_query_key_pos_per_sample = needle_positions + 3
    range_sizes = ((seq_len - 1) - min_query_key_pos_per_sample).clamp(min=1)
    query_key_positions = min_query_key_pos_per_sample + (torch.rand(batch_size) * range_sizes.float()).long().clamp(max=range_sizes - 1)
    query_positions = query_key_positions + 1
    X[batch_indices, query_key_positions] = NEEDLE_KEY_TOKEN_ID
    X[batch_indices, query_positions] = QUERY_TOKEN_ID
    Y[batch_indices, query_positions] = needle_values

    return X, Y, needle_positions

def get_niah_runtime_config(seq_len):
    """Return conservative NIAH runtime defaults by context length.

    中文说明:
    - 调用方 / Called by: `run_single_niah_test`
    - 调用对象 / Calls: 无外部函数
    - 作用 / Purpose: 根据上下文长度选择 batch、epoch 和 chunk；2M 场景使用 1024 chunk
      降低 Python 循环开销，同时依赖 selected logits 路径保持显存有界
    - 参数 / Parameters: `seq_len` 是 NIAH 上下文长度，必须为正整数
    - 返回 / Returns: dict，包含 `batch_size/epochs/chunk_size`
    - 接入 / Integration: 只影响 NIAH benchmark 训练入口，不改变模型公共接口
    - 错误处理 / Error handling: 当前保持历史宽松行为，不额外校验负数
    - 副作用 / Side effects: 无
    - 事务边界 / Transaction boundary: 不涉及事务
    - 并发与幂等 / Concurrency and idempotency: 纯函数，可重复调用
    - 关键词 / Keywords:
      runtime|niah|2m|chunk_size|epochs|batch_size|memory|benchmark|mhdsra2|配置

    English documentation:
    Function name:
        get_niah_runtime_config
    Purpose:
        Select conservative NIAH runtime defaults for each context length.
    Called by:
        `run_single_niah_test`.
    Calls:
        None.
    Parameters:
        - seq_len: context length.
    Returns:
        Dictionary with batch size, epoch count, and chunk size.
    Error handling:
        Preserves existing permissive behavior for invalid lengths.
    Side effects:
        None.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Pure and deterministic.
    English keywords:
        runtime, niah, 2m, chunk_size, epochs, batch_size, memory, benchmark, mhdsra2, config
    """
    if seq_len <= 32768:
        cfg = {"batch_size": 4, "epochs": 400, "chunk_size": 256}
    elif seq_len <= 131072:
        cfg = {"batch_size": 2, "epochs": 250, "chunk_size": 256}
    elif seq_len <= 524288:
        cfg = {"batch_size": 1, "epochs": 120, "chunk_size": 256}
    else:
        cfg = {"batch_size": 1, "epochs": 60, "chunk_size": 1024}
    chunk = cfg["chunk_size"]
    if seq_len % chunk != 0:
        print(
            f"WARNING: seq_len={seq_len} is not divisible by chunk_size={chunk}; "
            f"the model will internally pad to the next chunk boundary, which may "
            f"introduce slight performance overhead."
        )
    return cfg


def _find_query_positions_or_final(X):
    """Find one query marker per batch row, otherwise fall back to the final token.

    中文说明:
    - 调用方 / Called by: `extract_query_targets`, `extract_query_positions_and_targets`
    - 调用对象 / Calls: tensor comparison, `nonzero`, `torch.bincount`, tensor assignment
    - 作用 / Purpose: 统一解析 NIAH 查询位置；当输入中某些样本缺失 query 或存在重复 query 时，
      回退到协议规定的最终查询位置，避免未初始化位置导致准确率和 loss 失真
    - 参数 / Parameters:
      `X` 是 `[batch, seq_len]` token tensor，必须包含至少 2 个维度
    - 返回 / Returns: CPU long tensor `[batch]`，每个样本一个查询位置
    - 内部关键变量 / Internal variables:
      `query_counts` 统计每个 batch row 的 query token 数量，用于发现重复或缺失查询；
      `query_rows_cpu` 来自当前 tensor 的 `nonzero`，越界时直接抛出防御性错误
    - 接入 / Integration: 仅供 NIAH 数据提取函数调用，不应绕过该函数重复实现查询定位
    - 错误处理 / Error handling: 非二维输入抛出 `ValueError`
    - 副作用 / Side effects: 无；不修改输入 tensor
    - 事务边界 / Transaction boundary: 不涉及 Unit of Work、数据库或文件事务
    - 并发与幂等 / Concurrency and idempotency: 纯张量解析，对同一输入可重复调用
    - 关键词 / Keywords:
      query|position|fallback|duplicate|missing|niah|needle|target|accuracy|查询

    English documentation:
    Function name:
        _find_query_positions_or_final
    Purpose:
        Find exactly one query marker per batch row, or use the final token as
        the protocol fallback when query markers are missing or duplicated.
    Called by:
        `extract_query_targets` and `extract_query_positions_and_targets`.
    Calls:
        Tensor comparison, `nonzero`, `torch.bincount`, and tensor assignment.
    Parameters:
        - X: token tensor with shape `[batch, seq_len]`.
    Returns:
        CPU long tensor with one query position per batch item.
    Error handling:
        Raises `ValueError` for non-2D token tensors.
    Side effects:
        None.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Pure and deterministic for the same tensor values.
    English keywords:
        query, position, fallback, duplicate, missing, niah, needle, target, accuracy, extraction
    """
    if X.dim() != 2:
        raise ValueError(f"expected token ids with shape [B, SeqLen], got {tuple(X.shape)}")

    batch_size, seq_len = X.shape
    final_positions = torch.full((batch_size,), seq_len - 1, dtype=torch.long)
    query_rows, query_cols = (X == QUERY_TOKEN_ID).nonzero(as_tuple=True)
    query_rows_cpu = query_rows.detach().cpu()
    query_cols_cpu = query_cols.detach().cpu()
    if query_rows_cpu.numel() > 0 and int(query_rows_cpu.max().item()) >= batch_size:
        raise ValueError("query row index exceeds batch size")
    query_counts = torch.bincount(query_rows_cpu, minlength=batch_size)
    if query_counts.numel() != batch_size:
        return final_positions
    if bool((query_counts == 1).all().item()):
        query_positions = torch.empty(batch_size, dtype=torch.long)
        query_positions[query_rows_cpu] = query_cols_cpu
        return query_positions
    return final_positions


def extract_query_targets(X, Y, logits):
    """Extract full-forward logits and labels at NIAH query positions.

    中文说明:
    - 调用方 / Called by: 当前未发现项目内明确调用方；保留给 legacy 全序列 logits 路径
    - 调用对象 / Calls: `_find_query_positions_or_final`, tensor indexing
    - 作用 / Purpose: 从全序列 logits 中抽取 query token 的监督位置，保持旧全量 forward 路径
      与 selected logits 路径使用同一查询定位规则
    - 参数 / Parameters:
      `X/Y` 是 token 和标签 tensor；`logits` 是 `[batch, seq_len, vocab]` 模型输出
    - 返回 / Returns: `(logits_target, targets)`；targets 会对齐到 logits 所在设备
    - 内部关键变量 / Internal variables:
      `query_positions` 是每个样本用于计算 loss/accuracy 的查询位置
    - 接入 / Integration: 仅用于需要全序列 logits 的 legacy NIAH 路径；长上下文应优先使用
      `extract_query_positions_and_targets` 与 `forward_selected_logits`
    - 错误处理 / Error handling: 非二维输入由 `_find_query_positions_or_final` 抛出；
      `logits` 不是全序列三维输出时抛出 `ValueError`
    - 副作用 / Side effects: 无；不修改输入、不写文件
    - 事务边界 / Transaction boundary: 不涉及 Unit of Work、数据库或文件事务
    - 并发与幂等 / Concurrency and idempotency: 纯索引操作，对同一输入可重复调用
    - 关键词 / Keywords:
      logits|target|query|legacy|full_forward|niah|needle|accuracy|loss|提取

    English documentation:
    Function name:
        extract_query_targets
    Purpose:
        Extract supervised query logits and labels from full-sequence logits.
    Called by:
        No confirmed in-repository caller has been found; retained for legacy usage.
    Calls:
        `_find_query_positions_or_final` and tensor indexing.
    Parameters:
        - X/Y: token and label tensors.
        - logits: full-sequence logits with shape `[batch, seq_len, vocab]`.
    Returns:
        Query-position logits and device-aligned targets.
    Error handling:
        Raises `ValueError` when `logits` is not a full-sequence 3D tensor,
        and propagates invalid-shape errors from the query-position helper.
    Side effects:
        None.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Pure indexing for the same tensors.
    English keywords:
        logits, target, query, legacy, full_forward, niah, needle, accuracy, loss, extraction
    """
    if logits.dim() != 3:
        raise ValueError(
            "extract_query_targets expects full-sequence logits with shape [batch, seq_len, vocab]; "
            "use extract_query_positions_and_targets with forward_selected_logits outputs"
        )
    query_positions = _find_query_positions_or_final(X)
    batch_size = X.shape[0]
    batch_indices = torch.arange(batch_size, device=logits.device)
    logits_positions = query_positions.to(device=logits.device)
    logits_target = logits[batch_indices, logits_positions, :]
    target_positions = query_positions.to(device=Y.device)
    target_batch_indices = torch.arange(batch_size, device=Y.device)
    targets = Y[target_batch_indices, target_positions].to(device=logits.device)
    return logits_target, targets


def extract_query_positions_and_targets(X, Y, device):
    """Return query token positions and targets without moving the full sequence to GPU.

    中文说明:
    - 调用方 / Called by: `run_single_niah_test`, `run_single_niah_capacity_test`
    - 调用对象 / Calls: `_find_query_positions_or_final`, indexing
    - 作用 / Purpose: 从 CPU 侧 NIAH 样本中提取查询位置，并仅将小型 target tensor 搬到训练设备
    - 参数 / Parameters:
      `X/Y` 是 CPU token/target tensors；`device` 是目标训练设备
    - 返回 / Returns: `(query_positions, targets)`；positions 为 CPU `[B]`，targets 为设备侧 `[B]`
    - 接入 / Integration: 与 `MultiLayerMHDSRA2Model.forward_selected_logits` 配套使用
    - 错误处理 / Error handling: 若未能为每个 batch 找到唯一 query，则回退到最后一个 token
      并避免未初始化位置污染测试结果
    - 副作用 / Side effects: 无；不修改输入
    - 事务边界 / Transaction boundary: 不涉及事务
    - 并发与幂等 / Concurrency and idempotency: 对同一输入重复调用结果一致
    - 关键词 / Keywords:
      query_position|target|niah|cpu|gpu|selected_logits|memory|extract|needle|查询

    English documentation:
    Function name:
        extract_query_positions_and_targets
    Purpose:
        Extract supervised query positions and small target tensors for memory-bounded NIAH training.
    Called by:
        NIAH train and capacity-test entry points.
    Calls:
        `_find_query_positions_or_final` and indexing.
    Parameters:
        - X/Y: CPU token and target tensors.
        - device: target device for labels.
    Returns:
        CPU query positions and device-side target labels.
    Error handling:
        Falls back to the final token if a unique query token is not found per sample.
    Side effects:
        None.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Deterministic for the same tensors.
    English keywords:
        query_position, target, niah, cpu, gpu, selected_logits, memory, extract, needle, query
    """
    batch_size, _ = X.shape
    query_positions = _find_query_positions_or_final(X)
    target_positions = query_positions.to(device=Y.device)
    target_batch_indices = torch.arange(batch_size, device=Y.device)
    targets = Y[target_batch_indices, target_positions].to(device)
    return query_positions, targets


def compute_retrieval_evidence_gate_loss(
    aux,
    evidence_positions,
    *,
    device,
    rank_margin: float = 0.0,
    score_margin: float = 0.0,
):
    """Train retrieval gate confidence against known synthetic evidence positions.

    中文说明:
    - 调用方 / Called by: `run_niah_verification_case`
    - 调用对象 / Calls: tensor comparison, `F.binary_cross_entropy`
    - 作用 / Purpose: NIAH 训练样本知道 needle value 的真实历史位置；本函数检查 external
      retrieval 的候选位置是否包含该证据，并用这个命中标签监督 retrieval gate。
    - 参数 / Parameters:
      `aux` 来自 `forward_selected_logits(..., return_aux=True)`;
      `evidence_positions` 是每个样本的 needle value 全局位置；
      `rank_margin` 默认 0.0，显式大于 0 时增加“正确证据权重要超过最强错误候选”
      的 margin 排名约束；
      `score_margin` 默认 0.0，显式大于 0 时对 softmax 前 retrieval raw score 做同类
      margin 约束，避免只在后验权重上间接推动候选排序。
    - 返回 / Returns: `(loss, metrics)`；aux 不可用时返回 0 loss 和 unavailable 指标。
    - 错误处理 / Error handling: 缺少 metadata/gate 时不抛错，返回 unavailable，避免默认训练路径受影响。
    - 副作用 / Side effects: 无；不修改模型状态。
    - 关键词 / Keywords:
      niah|retrieval|evidence|gate|auxiliary_loss|metadata|needle_position|检索监督

    English documentation:
    Function name:
        compute_retrieval_evidence_gate_loss
    Purpose:
        Supervise retrieval gate probability with whether the known synthetic
        evidence token was returned by external paged retrieval.
    """
    zero = torch.tensor(0.0, device=device)
    metrics = {
        "available": False,
        "unavailable_reason": None,
        "hit_rate": 0.0,
        "gate_mean": 0.0,
        "evidence_weight_mean": 0.0,
        "best_negative_weight_mean": 0.0,
        "evidence_margin_mean": 0.0,
        "target_rank_mean": None,
        "top1_rate": None,
        "ranking_loss": 0.0,
        "margin_loss": 0.0,
        "score_margin_loss": 0.0,
        "evidence_score_mean": 0.0,
        "best_negative_score_mean": 0.0,
        "evidence_score_margin_mean": 0.0,
        "score_target_rank_mean": None,
        "score_top1_rate": None,
        "gate_loss": 0.0,
        "positive_count": 0,
    }
    if not isinstance(aux, dict):
        metrics["unavailable_reason"] = "missing_aux"
        return zero, metrics
    last_layer = aux.get("last_layer")
    if not isinstance(last_layer, dict):
        metrics["unavailable_reason"] = "missing_last_layer_aux"
        return zero, metrics
    metadata = last_layer.get("selected_retrieval_metadata")
    gate_by_sample = last_layer.get(
        "selected_gate_retrieval_by_sample_for_loss",
        last_layer.get(
            "selected_gate_retrieval_by_sample",
            None,
        ),
    )
    if not isinstance(metadata, dict) or not isinstance(gate_by_sample, torch.Tensor):
        metrics["unavailable_reason"] = "missing_selected_metadata_or_gate"
        return zero, metrics
    positions = metadata.get("positions")
    mask = metadata.get("mask")
    if not isinstance(positions, torch.Tensor) or not isinstance(mask, torch.Tensor):
        metrics["unavailable_reason"] = "missing_selected_positions_or_mask"
        return zero, metrics

    evidence = torch.as_tensor(evidence_positions, dtype=torch.long, device=positions.device)
    evidence = evidence.flatten()
    selected_batch_size = 1 if positions.dim() == 1 else int(positions.shape[0])
    selected_batch_indices = last_layer.get("selected_batch_indices")
    if isinstance(selected_batch_indices, torch.Tensor) and evidence.numel() > selected_batch_size:
        selected_indices = selected_batch_indices.to(device=evidence.device, dtype=torch.long).flatten()
        if selected_indices.numel() != selected_batch_size:
            metrics["unavailable_reason"] = "selected_index_count_mismatch"
            return zero, metrics
        if bool(((selected_indices < 0) | (selected_indices >= evidence.numel())).any().item()):
            metrics["unavailable_reason"] = "selected_index_out_of_range"
            return zero, metrics
        evidence = evidence[selected_indices]
    elif evidence.numel() == 1 and selected_batch_size > 1:
        evidence = evidence.expand(selected_batch_size)
    if mask.dim() != positions.dim():
        metrics["unavailable_reason"] = "selected_mask_rank_mismatch"
        return zero, metrics
    if positions.dim() == 2 and mask.shape[0] != selected_batch_size:
        metrics["unavailable_reason"] = "selected_mask_batch_mismatch"
        return zero, metrics
    if positions.shape[-1] != mask.shape[-1]:
        metrics["unavailable_reason"] = "selected_mask_width_mismatch"
        return zero, metrics
    if evidence.numel() != selected_batch_size:
        metrics["unavailable_reason"] = "evidence_batch_mismatch"
        return zero, metrics
    if gate_by_sample.numel() != selected_batch_size:
        metrics["unavailable_reason"] = "gate_batch_mismatch"
        return zero, metrics
    if positions.dim() == 1:
        hit = ((positions.unsqueeze(0) == evidence.view(-1, 1)) & mask.view(1, -1)).any(dim=1)
    elif positions.dim() == 2:
        hit = ((positions == evidence.view(-1, 1)) & mask).any(dim=1)
    else:
        metrics["unavailable_reason"] = "selected_positions_rank_unsupported"
        return zero, metrics
    gate_prob = gate_by_sample.to(device=device, dtype=torch.float32).flatten()
    target = hit.to(device=device, dtype=torch.float32).flatten()
    if gate_prob.numel() != target.numel():
        metrics["unavailable_reason"] = "gate_target_count_mismatch"
        return zero, metrics
    gate_loss = F.binary_cross_entropy(gate_prob.clamp(1e-6, 1.0 - 1e-6), target)
    ranking_loss = torch.tensor(0.0, device=device)
    margin_loss = torch.tensor(0.0, device=device)
    score_margin_loss = torch.tensor(0.0, device=device)
    evidence_weight_mean = torch.tensor(0.0, device=device)
    best_negative_weight_mean = torch.tensor(0.0, device=device)
    evidence_margin_mean = torch.tensor(0.0, device=device)
    evidence_score_mean = torch.tensor(0.0, device=device)
    best_negative_score_mean = torch.tensor(0.0, device=device)
    evidence_score_margin_mean = torch.tensor(0.0, device=device)
    target_rank_mean = None
    top1_rate = None
    score_target_rank_mean = None
    score_top1_rate = None
    token_weights = last_layer.get(
        "selected_retrieval_token_weight_by_sample_for_loss",
        last_layer.get(
            "selected_retrieval_token_weight_by_sample",
            None,
        ),
    )
    if isinstance(token_weights, torch.Tensor) and bool(hit.any().item()):
        weights = token_weights.to(device=device, dtype=torch.float32)
        if weights.dim() == 1:
            weights = weights.view(1, -1)
        if (
            weights.dim() == 2
            and weights.shape[0] == selected_batch_size
            and weights.shape[0] == target.numel()
        ):
            if positions.dim() == 1:
                evidence_mask = (
                    positions.to(device=weights.device).view(1, -1)
                    == evidence.to(device=weights.device).view(-1, 1)
                ) & mask.to(device=weights.device, dtype=torch.bool).view(1, -1)
            else:
                evidence_mask = (
                    positions.to(device=weights.device)
                    == evidence.to(device=weights.device).view(-1, 1)
                ) & mask.to(device=weights.device, dtype=torch.bool)
            positive_rows = evidence_mask.any(dim=1)
            if bool(positive_rows.any().item()):
                valid_mask = mask.to(device=weights.device, dtype=torch.bool)
                evidence_weights_raw = weights.masked_fill(~evidence_mask, 0.0).sum(dim=1)
                evidence_weights = evidence_weights_raw.clamp_min(1e-6)
                negative_mask = valid_mask & ~evidence_mask
                has_negative = negative_mask.any(dim=1)
                negative_weights = weights.masked_fill(
                    ~negative_mask,
                    torch.finfo(weights.dtype).min,
                )
                best_negative_weights = torch.where(
                    has_negative,
                    negative_weights.max(dim=1).values,
                    torch.zeros_like(evidence_weights_raw),
                )
                evidence_margins = evidence_weights_raw - best_negative_weights
                target_ranks = (
                    (
                        negative_mask
                        & (weights > evidence_weights_raw.unsqueeze(1))
                    )
                    .to(dtype=torch.long)
                    .sum(dim=1)
                    + 1
                )
                top1_rows = evidence_weights_raw >= best_negative_weights

                evidence_weight_mean = evidence_weights[positive_rows].mean()
                ranking_loss = -torch.log(evidence_weights[positive_rows]).mean()
                best_negative_weight_mean = best_negative_weights[positive_rows].mean()
                evidence_margin_mean = evidence_margins[positive_rows].mean()
                target_rank_mean = float(
                    target_ranks[positive_rows].to(dtype=torch.float32).mean().detach().cpu().item()
                )
                top1_rate = float(
                    top1_rows[positive_rows].to(dtype=torch.float32).mean().detach().cpu().item()
                )
                margin_rows = positive_rows & has_negative
                if rank_margin > 0.0 and bool(margin_rows.any().item()):
                    margin_target = torch.tensor(
                        float(rank_margin),
                        device=weights.device,
                        dtype=weights.dtype,
                    )
                    margin_loss = F.relu(
                        best_negative_weights[margin_rows]
                        + margin_target
                        - evidence_weights_raw[margin_rows]
                    ).mean()
    token_scores = last_layer.get(
        "selected_retrieval_token_score_by_sample_for_loss",
        last_layer.get(
            "selected_retrieval_token_score_by_sample",
            None,
        ),
    )
    if score_margin > 0.0 and isinstance(token_scores, torch.Tensor) and bool(hit.any().item()):
        scores = token_scores.to(device=device, dtype=torch.float32)
        if scores.dim() == 1:
            scores = scores.view(1, -1)
        if (
            scores.dim() == 2
            and scores.shape[0] == selected_batch_size
            and scores.shape[0] == target.numel()
        ):
            if positions.dim() == 1:
                evidence_mask = (
                    positions.to(device=scores.device).view(1, -1)
                    == evidence.to(device=scores.device).view(-1, 1)
                ) & mask.to(device=scores.device, dtype=torch.bool).view(1, -1)
                valid_mask = mask.to(device=scores.device, dtype=torch.bool).view(1, -1).expand_as(
                    scores
                )
            else:
                evidence_mask = (
                    positions.to(device=scores.device)
                    == evidence.to(device=scores.device).view(-1, 1)
                ) & mask.to(device=scores.device, dtype=torch.bool)
                valid_mask = mask.to(device=scores.device, dtype=torch.bool)
            positive_rows = evidence_mask.any(dim=1)
            if bool(positive_rows.any().item()):
                evidence_scores = scores.masked_fill(
                    ~evidence_mask,
                    torch.finfo(scores.dtype).min,
                ).max(dim=1).values
                negative_mask = valid_mask & ~evidence_mask
                has_negative = negative_mask.any(dim=1)
                negative_scores = scores.masked_fill(
                    ~negative_mask,
                    torch.finfo(scores.dtype).min,
                )
                best_negative_scores = torch.where(
                    has_negative,
                    negative_scores.max(dim=1).values,
                    torch.zeros_like(evidence_scores),
                )
                evidence_score_margins = evidence_scores - best_negative_scores
                score_target_ranks = (
                    (
                        negative_mask
                        & (scores > evidence_scores.unsqueeze(1))
                    )
                    .to(dtype=torch.long)
                    .sum(dim=1)
                    + 1
                )
                score_top1_rows = evidence_scores >= best_negative_scores
                evidence_score_mean = evidence_scores[positive_rows].mean()
                best_negative_score_mean = best_negative_scores[positive_rows].mean()
                evidence_score_margin_mean = evidence_score_margins[positive_rows].mean()
                score_target_rank_mean = float(
                    score_target_ranks[positive_rows]
                    .to(dtype=torch.float32)
                    .mean()
                    .detach()
                    .cpu()
                    .item()
                )
                score_top1_rate = float(
                    score_top1_rows[positive_rows]
                    .to(dtype=torch.float32)
                    .mean()
                    .detach()
                    .cpu()
                    .item()
                )
                score_margin_rows = positive_rows & has_negative
                if bool(score_margin_rows.any().item()):
                    score_margin_target = torch.tensor(
                        float(score_margin),
                        device=scores.device,
                        dtype=scores.dtype,
                    )
                    score_margin_loss = F.relu(
                        best_negative_scores[score_margin_rows]
                        + score_margin_target
                        - evidence_scores[score_margin_rows]
                    ).mean()
    loss = gate_loss + ranking_loss + margin_loss + score_margin_loss
    metrics.update(
        {
            "available": True,
            "unavailable_reason": None,
            "hit_rate": float(target.mean().detach().cpu().item()) if target.numel() else 0.0,
            "gate_mean": float(gate_prob.mean().detach().cpu().item()) if gate_prob.numel() else 0.0,
            "evidence_weight_mean": float(evidence_weight_mean.detach().cpu().item()),
            "best_negative_weight_mean": float(best_negative_weight_mean.detach().cpu().item()),
            "evidence_margin_mean": float(evidence_margin_mean.detach().cpu().item()),
            "target_rank_mean": target_rank_mean,
            "top1_rate": top1_rate,
            "ranking_loss": float(ranking_loss.detach().cpu().item()),
            "margin_loss": float(margin_loss.detach().cpu().item()),
            "score_margin_loss": float(score_margin_loss.detach().cpu().item()),
            "evidence_score_mean": float(evidence_score_mean.detach().cpu().item()),
            "best_negative_score_mean": float(best_negative_score_mean.detach().cpu().item()),
            "evidence_score_margin_mean": float(
                evidence_score_margin_mean.detach().cpu().item()
            ),
            "score_target_rank_mean": score_target_rank_mean,
            "score_top1_rate": score_top1_rate,
            "gate_loss": float(gate_loss.detach().cpu().item()),
            "positive_count": int(target.sum().detach().cpu().item()) if target.numel() else 0,
        }
    )
    return loss, metrics


def compute_query_evidence_alignment_loss(
    hidden_query,
    evidence_tokens,
    embedding,
    *,
    detach_evidence: bool = True,
):
    """Align query hidden states with known NIAH evidence token embeddings.

    中文说明:
    - 调用方 / Called by: `run_niah_verification_case` when the experimental
      `query_evidence_alignment_alpha` switch is enabled; unit tests call it directly.
    - 调用对象 / Calls: `embedding`, `F.normalize`, `F.mse_loss`
    - 作用 / Purpose: 默认训练路径不使用本函数；显式打开时，把 query 位置的 hidden state
      拉近已知 evidence token 的 embedding，用于验证“最终 query 表征是否能靠近证据 token”。
    - 参数 / Parameters: `hidden_query` 是 `[B,D]` query hidden；`evidence_tokens` 是 `[B]`
      证据 token id；`embedding` 是模型 embedding 模块；`detach_evidence` 默认 True，避免把
      辅助目标退化成只移动 token embedding。
    - 返回 / Returns: `(loss, metrics)`，metrics 包含 cosine、mse 和 available 状态。
    - 错误处理 / Error handling: 形状不匹配时抛 `ValueError`，避免静默广播。
    - 副作用 / Side effects: 无；不修改模型或输入。
    - 关键词 / Keywords:
      niah|query|evidence|alignment|hidden|embedding|auxiliary_loss|表征对齐

    English documentation:
    Function name:
        compute_query_evidence_alignment_loss
    Purpose:
        Add an explicit, opt-in representation-alignment auxiliary loss between
        selected query hidden states and known evidence token embeddings.
    """
    if hidden_query.dim() != 2:
        raise ValueError(f"hidden_query must be [B,D], got {tuple(hidden_query.shape)}")
    evidence_tokens = torch.as_tensor(
        evidence_tokens,
        dtype=torch.long,
        device=hidden_query.device,
    ).flatten()
    if evidence_tokens.numel() != hidden_query.shape[0]:
        raise ValueError(
            "evidence token count must match hidden_query batch size "
            f"({evidence_tokens.numel()} vs {hidden_query.shape[0]})"
        )
    evidence_embed = embedding(evidence_tokens)
    if detach_evidence:
        evidence_embed = evidence_embed.detach()
    query_norm = F.normalize(hidden_query.to(dtype=torch.float32), dim=-1)
    evidence_norm = F.normalize(evidence_embed.to(dtype=torch.float32), dim=-1)
    cosine = (query_norm * evidence_norm).sum(dim=-1)
    cosine_loss = (1.0 - cosine).mean()
    mse_loss = F.mse_loss(query_norm, evidence_norm)
    metrics = {
        "available": True,
        "loss": float(cosine_loss.detach().cpu().item()),
        "mean_cosine": float(cosine.mean().detach().cpu().item()),
        "mean_mse": float(mse_loss.detach().cpu().item()),
        "detach_evidence": bool(detach_evidence),
    }
    return cosine_loss, metrics


def compute_retrieval_projection_contrastive_loss(
    aux,
    evidence_positions,
    *,
    device,
    temperature: float = 0.10,
):
    """Supervise selected retrieval query/key projections with known evidence positions.

    中文说明:
    - 调用方 / Called by: `run_niah_verification_case` when the experimental
      `retrieval_projection_contrastive_alpha` switch is enabled.
    - 调用对象 / Calls: `F.normalize`, `F.cross_entropy`, tensor indexing.
    - 作用 / Purpose: 默认训练路径不使用本函数；显式打开时，只读取 selected query
      位置对应的 retrieval query projection 和候选 key projection，让正确 evidence
      位置在 query-key 相似度里排到前面。它不读取 validation/test 指标。
    - 参数 / Parameters:
      `aux` 来自 `forward_selected_logits(..., return_aux=True)`;
      `evidence_positions` 是每个训练样本的 gold evidence 全局位置；
      `temperature` 控制 contrastive softmax 锐度，必须为正数。
    - 返回 / Returns: `(loss, metrics)`；aux 不完整或 evidence 未进入候选时返回 0 loss
      和 unavailable/diagnostic 指标。
    - 错误处理 / Error handling: 形状不匹配返回 unavailable，非法 temperature 抛 `ValueError`。
    - 副作用 / Side effects: 无；只产生训练期辅助梯度。
    - 关键词 / Keywords:
      retrieval|projection|contrastive|query_key|niah|auxiliary_loss|表征

    English documentation:
    Function name:
        compute_retrieval_projection_contrastive_loss
    Purpose:
        Add an opt-in contrastive loss directly on selected retrieval query/key
        projections for train-only synthetic evidence.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    zero = torch.tensor(0.0, device=device)
    metrics = {
        "available": False,
        "unavailable_reason": None,
        "hit_rate": 0.0,
        "positive_count": 0,
        "loss": 0.0,
        "evidence_score_mean": 0.0,
        "best_negative_score_mean": 0.0,
        "score_margin_mean": 0.0,
        "target_rank_mean": None,
        "top1_rate": None,
        "temperature": float(temperature),
    }
    if not isinstance(aux, dict):
        metrics["unavailable_reason"] = "missing_aux"
        return zero, metrics
    last_layer = aux.get("last_layer")
    if not isinstance(last_layer, dict):
        metrics["unavailable_reason"] = "missing_last_layer_aux"
        return zero, metrics
    metadata = last_layer.get("selected_retrieval_metadata")
    query_projection = last_layer.get(
        "selected_retrieval_query_projection_for_loss",
        last_layer.get("selected_retrieval_query_projection"),
    )
    key_projection = last_layer.get(
        "selected_retrieval_key_projection_for_loss",
        last_layer.get("selected_retrieval_key_projection"),
    )
    if (
        not isinstance(metadata, dict)
        or not isinstance(query_projection, torch.Tensor)
        or not isinstance(key_projection, torch.Tensor)
    ):
        metrics["unavailable_reason"] = "missing_selected_projection_aux"
        return zero, metrics

    positions = metadata.get("positions")
    mask = metadata.get("mask")
    if not isinstance(positions, torch.Tensor) or not isinstance(mask, torch.Tensor):
        metrics["unavailable_reason"] = "missing_selected_positions_or_mask"
        return zero, metrics

    evidence = torch.as_tensor(evidence_positions, dtype=torch.long, device=positions.device)
    evidence = evidence.flatten()
    selected_batch_size = 1 if positions.dim() == 1 else int(positions.shape[0])
    selected_batch_indices = last_layer.get("selected_batch_indices")
    if isinstance(selected_batch_indices, torch.Tensor) and evidence.numel() > selected_batch_size:
        selected_indices = selected_batch_indices.to(device=evidence.device, dtype=torch.long).flatten()
        if selected_indices.numel() != selected_batch_size:
            metrics["unavailable_reason"] = "selected_index_count_mismatch"
            return zero, metrics
        if bool(((selected_indices < 0) | (selected_indices >= evidence.numel())).any().item()):
            metrics["unavailable_reason"] = "selected_index_out_of_range"
            return zero, metrics
        evidence = evidence[selected_indices]
    elif evidence.numel() == 1 and selected_batch_size > 1:
        evidence = evidence.expand(selected_batch_size)

    if mask.dim() != positions.dim():
        metrics["unavailable_reason"] = "selected_mask_rank_mismatch"
        return zero, metrics
    if positions.dim() == 1:
        positions_2d = positions.view(1, -1)
        mask_2d = mask.view(1, -1)
    elif positions.dim() == 2:
        positions_2d = positions
        mask_2d = mask
    else:
        metrics["unavailable_reason"] = "selected_positions_rank_unsupported"
        return zero, metrics
    if positions_2d.shape != mask_2d.shape:
        metrics["unavailable_reason"] = "selected_positions_mask_shape_mismatch"
        return zero, metrics
    if positions_2d.shape[0] != selected_batch_size:
        metrics["unavailable_reason"] = "selected_positions_batch_mismatch"
        return zero, metrics
    if evidence.numel() != selected_batch_size:
        metrics["unavailable_reason"] = "evidence_batch_mismatch"
        return zero, metrics

    query = query_projection.to(device=device, dtype=torch.float32)
    keys = key_projection.to(device=device, dtype=torch.float32)
    if query.dim() != 3:
        metrics["unavailable_reason"] = "query_projection_rank_mismatch"
        return zero, metrics
    if keys.dim() != 4:
        metrics["unavailable_reason"] = "key_projection_rank_mismatch"
        return zero, metrics
    if query.shape[0] != selected_batch_size or keys.shape[0] != selected_batch_size:
        metrics["unavailable_reason"] = "projection_batch_mismatch"
        return zero, metrics
    if query.shape[1] != keys.shape[1] or query.shape[2] != keys.shape[3]:
        metrics["unavailable_reason"] = "projection_head_dim_mismatch"
        return zero, metrics
    if keys.shape[2] != positions_2d.shape[1]:
        metrics["unavailable_reason"] = "projection_candidate_width_mismatch"
        return zero, metrics

    positions_2d = positions_2d.to(device=device, dtype=torch.long)
    mask_2d = mask_2d.to(device=device, dtype=torch.bool)
    evidence = evidence.to(device=device, dtype=torch.long)
    evidence_mask = (positions_2d == evidence.view(-1, 1)) & mask_2d
    hit = evidence_mask.any(dim=1)
    valid_rows = hit & mask_2d.any(dim=1)
    metrics["hit_rate"] = float(hit.to(dtype=torch.float32).mean().detach().cpu().item())
    metrics["positive_count"] = int(hit.to(dtype=torch.long).sum().detach().cpu().item())
    if not bool(valid_rows.any().item()):
        metrics["unavailable_reason"] = "evidence_not_in_selected_candidates"
        return zero, metrics

    query_norm = F.normalize(query, dim=-1)
    key_norm = F.normalize(keys, dim=-1)
    scores = torch.einsum("bhd,bhrd->br", query_norm, key_norm) / query.shape[1]
    scores = scores.masked_fill(~mask_2d, torch.finfo(scores.dtype).min)
    positive_indices = evidence_mask.to(dtype=torch.long).argmax(dim=1)
    row_indices = valid_rows.nonzero(as_tuple=True)[0]
    logits = scores[row_indices] / float(temperature)
    targets = positive_indices[row_indices]
    loss = F.cross_entropy(logits, targets)

    evidence_scores = scores.gather(1, positive_indices.view(-1, 1)).squeeze(1)
    negative_mask = mask_2d & ~evidence_mask
    has_negative = negative_mask.any(dim=1)
    negative_scores = scores.masked_fill(
        ~negative_mask,
        torch.finfo(scores.dtype).min,
    )
    best_negative_scores = torch.where(
        has_negative,
        negative_scores.max(dim=1).values,
        torch.zeros_like(evidence_scores),
    )
    score_margins = evidence_scores - best_negative_scores
    target_ranks = (
        (negative_mask & (scores > evidence_scores.unsqueeze(1)))
        .to(dtype=torch.long)
        .sum(dim=1)
        + 1
    )
    top1_rows = evidence_scores >= best_negative_scores
    metrics.update(
        {
            "available": True,
            "unavailable_reason": None,
            "loss": float(loss.detach().cpu().item()),
            "evidence_score_mean": float(evidence_scores[valid_rows].mean().detach().cpu().item()),
            "best_negative_score_mean": float(
                best_negative_scores[valid_rows].mean().detach().cpu().item()
            ),
            "score_margin_mean": float(score_margins[valid_rows].mean().detach().cpu().item()),
            "target_rank_mean": float(
                target_ranks[valid_rows].to(dtype=torch.float32).mean().detach().cpu().item()
            ),
            "top1_rate": float(
                top1_rows[valid_rows].to(dtype=torch.float32).mean().detach().cpu().item()
            ),
        }
    )
    return loss, metrics


def _default_retrieval_span_metrics(device):
    """Return a fresh unavailable metrics payload for span predictor diagnostics."""
    return {
        "available": False,
        "unavailable_reason": None,
        "loss": 0.0,
        "hit_rate": 0.0,
        "positive_count": 0,
        "target_rank_mean": None,
        "top1_rate": None,
        "evidence_logit_mean": None,
        "best_negative_logit_mean": None,
        "logit_margin_mean": None,
        "candidate_filter": DEFAULT_NIAH_SPAN_CANDIDATE_FILTER,
        "loss_mode": DEFAULT_NIAH_SPAN_LOSS_MODE,
        "device": str(device),
    }


def _metadata_position_lists(
    metadata,
    *,
    batch_size,
    list_field,
    tensor_field,
):
    """Extract per-sample retrieval locality positions from selected metadata.

    中文说明:
    - 调用方 / Called by: `_prepare_niah_span_candidate_tensors`。
    - 调用对象 / Calls: `torch.empty`, tensor detach/cpu conversion。
    - 作用 / Purpose: 将分页检索返回的 page-locality 诊断整理成每个样本一组
      CPU position tensor。batch=1 时兼容旧式单 tensor 字段；batch>1 时优先读取
      `*_by_sample` 列表，避免把不同样本的候选页混在一起。
    - 参数 / Parameters: `metadata` 是 selected retrieval metadata；`list_field` 是
      batch 隔离列表字段；`tensor_field` 是 batch=1 兼容字段。
    - 返回 / Returns: 长度为 `batch_size` 的 CPU long tensor 列表。
    - 错误处理 / Error handling: 字段缺失或形状不匹配时返回空 tensor，不抛错影响评估。
    - 副作用 / Side effects: 无。
    - 关键词 / Keywords:
      niah|retrieval_metadata|locality|page_candidate|top_token|diagnostics|位置

    English documentation:
    Function name:
        _metadata_position_lists
    Purpose:
        Normalize optional retrieval locality metadata into per-sample CPU
        position lists for diagnostic-only NIAH metrics.
    """
    empty = [torch.empty(0, dtype=torch.long) for _ in range(batch_size)]
    if not isinstance(metadata, dict):
        return empty
    list_value = metadata.get(list_field)
    if isinstance(list_value, list) and len(list_value) == batch_size:
        rows = []
        for value in list_value:
            if isinstance(value, torch.Tensor):
                rows.append(value.detach().cpu().long().flatten())
            else:
                rows.append(torch.empty(0, dtype=torch.long))
        return rows
    tensor_value = metadata.get(tensor_field)
    if isinstance(tensor_value, torch.Tensor) and batch_size == 1:
        return [tensor_value.detach().cpu().long().flatten()]
    return empty


def _prepare_niah_span_candidate_tensors(
    X,
    targets,
    query_positions,
    aux,
    *,
    device,
    candidate_filter=DEFAULT_NIAH_SPAN_CANDIDATE_FILTER,
    loss_mode=DEFAULT_NIAH_SPAN_LOSS_MODE,
):
    """Build candidate/source token tensors from selected retrieval metadata.

    中文说明:
    - 调用方 / Called by: span predictor loss/readout helpers.
    - 调用对象 / Calls: tensor indexing only.
    - 作用 / Purpose: 将 selected retrieval metadata 转成 `[B,R]` 候选位置、候选 token、
      可复制来源 token、mask、检索权重和分数。默认不过滤旧候选；显式
      `key_value_pair` 时只保留 selected retrieval 中 key/value 成对同时出现的候选；
      `prefer_key_value_pair` 有成对候选时收窄，无成对候选时回退。
      函数不读取 gold evidence 位置，只用 targets 在训练 loss 或评估指标里判断哪个候选预测正确。
    - 参数 / Parameters: `X` 是 CPU NIAH token，`targets/query_positions` 是当前 batch 标签和查询位置，
      `aux` 是 `forward_selected_logits(..., return_aux=True)` 返回的 selected 诊断；
      `candidate_filter` 控制是否收窄候选池。
    - 返回 / Returns: `(candidate_payload, unavailable_reason)`。
    - 错误处理 / Error handling: metadata 缺失或形状不一致时返回 reason，不抛错影响默认路径。
    - 副作用 / Side effects: 无。
    - 关键词 / Keywords:
      niah|span_predictor|candidate|metadata|retrieval|readout|证据候选
    """
    candidate_filter = normalize_niah_span_candidate_filter(candidate_filter)
    if X.dim() != 2:
        return None, "tokens_rank_mismatch"
    if targets.dim() != 1:
        return None, "targets_rank_mismatch"
    batch_size, sequence_width = X.shape
    if targets.numel() != batch_size:
        return None, "target_batch_mismatch"
    last_layer = aux.get("last_layer") if isinstance(aux, dict) else None
    if not isinstance(last_layer, dict):
        return None, "missing_last_layer_aux"
    metadata = last_layer.get("selected_retrieval_metadata")
    if not isinstance(metadata, dict):
        return None, "missing_selected_retrieval_metadata"
    positions = metadata.get("positions")
    mask = metadata.get("mask")
    if not isinstance(positions, torch.Tensor) or not isinstance(mask, torch.Tensor):
        return None, "missing_selected_positions_or_mask"

    positions_cpu = positions.detach().cpu().long()
    mask_cpu = mask.detach().cpu().bool()
    if positions_cpu.dim() == 1:
        positions_cpu = positions_cpu.view(1, -1)
        mask_cpu = mask_cpu.view(1, -1)
    if positions_cpu.dim() != 2 or mask_cpu.dim() != 2:
        return None, "selected_positions_rank_unsupported"
    if positions_cpu.shape != mask_cpu.shape:
        return None, "selected_positions_mask_shape_mismatch"
    if positions_cpu.shape[0] != batch_size:
        return None, "selected_retrieval_batch_mismatch"

    candidate_count = int(positions_cpu.shape[1])
    source_positions = positions_cpu.clone()
    candidate_tokens = torch.zeros(batch_size, candidate_count, dtype=torch.long)
    source_tokens = torch.zeros(batch_size, candidate_count, dtype=torch.long)
    copyable_mask = mask_cpu.clone()
    pair_mask = torch.zeros_like(copyable_mask)
    query_positions_cpu = query_positions.detach().cpu().long().flatten()
    if query_positions_cpu.numel() == 1 and batch_size > 1:
        query_positions_cpu = query_positions_cpu.expand(batch_size)
    if query_positions_cpu.numel() != batch_size:
        return None, "query_position_batch_mismatch"
    page_candidate_positions_by_sample = _metadata_position_lists(
        metadata,
        batch_size=batch_size,
        list_field="page_candidate_positions_by_sample",
        tensor_field="page_candidate_positions",
    )
    top_token_positions_by_sample = _metadata_position_lists(
        metadata,
        batch_size=batch_size,
        list_field="top_token_positions_by_sample",
        tensor_field="top_token_positions",
    )
    seed_token_positions_by_sample = _metadata_position_lists(
        metadata,
        batch_size=batch_size,
        list_field="seed_token_positions_by_sample",
        tensor_field="seed_token_positions",
    )
    target_cpu = targets.detach().cpu().long()
    target_value_in_page_candidates = torch.zeros(batch_size, dtype=torch.bool)
    target_pair_in_page_candidates = torch.zeros(batch_size, dtype=torch.bool)
    target_value_in_top_tokens = torch.zeros(batch_size, dtype=torch.bool)
    target_pair_in_top_tokens = torch.zeros(batch_size, dtype=torch.bool)
    target_value_in_seed_tokens = torch.zeros(batch_size, dtype=torch.bool)
    target_pair_in_seed_tokens = torch.zeros(batch_size, dtype=torch.bool)
    for sample_idx in range(batch_size):
        query_position = int(query_positions_cpu[sample_idx].item())
        target_token = int(target_cpu[sample_idx].item())
        page_position_set = {
            int(position)
            for position in page_candidate_positions_by_sample[sample_idx].tolist()
            if int(position) < query_position
        }
        top_token_position_set = {
            int(position)
            for position in top_token_positions_by_sample[sample_idx].tolist()
            if int(position) < query_position
        }
        seed_token_position_set = {
            int(position)
            for position in seed_token_positions_by_sample[sample_idx].tolist()
            if int(position) < query_position
        }
        for position in page_position_set:
            if 0 <= position < sequence_width and int(X[sample_idx, position].item()) == target_token:
                target_value_in_page_candidates[sample_idx] = True
                left_position = position - 1
                if (
                    left_position in page_position_set
                    and int(X[sample_idx, left_position].item()) == NEEDLE_KEY_TOKEN_ID
                ):
                    target_pair_in_page_candidates[sample_idx] = True
                break
        for position in top_token_position_set:
            if 0 <= position < sequence_width and int(X[sample_idx, position].item()) == target_token:
                target_value_in_top_tokens[sample_idx] = True
                left_position = position - 1
                if (
                    left_position in top_token_position_set
                    and int(X[sample_idx, left_position].item()) == NEEDLE_KEY_TOKEN_ID
                ):
                    target_pair_in_top_tokens[sample_idx] = True
                break
        for position in seed_token_position_set:
            if 0 <= position < sequence_width and int(X[sample_idx, position].item()) == target_token:
                target_value_in_seed_tokens[sample_idx] = True
                left_position = position - 1
                if (
                    left_position in seed_token_position_set
                    and int(X[sample_idx, left_position].item()) == NEEDLE_KEY_TOKEN_ID
                ):
                    target_pair_in_seed_tokens[sample_idx] = True
                break
        valid_candidate_positions = {
            int(positions_cpu[sample_idx, candidate_idx].item())
            for candidate_idx in range(candidate_count)
            if bool(mask_cpu[sample_idx, candidate_idx].item())
        }
        for candidate_idx in range(candidate_count):
            if not bool(mask_cpu[sample_idx, candidate_idx].item()):
                copyable_mask[sample_idx, candidate_idx] = False
                continue
            candidate_position = int(positions_cpu[sample_idx, candidate_idx].item())
            if candidate_position < 0 or candidate_position >= sequence_width:
                copyable_mask[sample_idx, candidate_idx] = False
                continue
            candidate_token = int(X[sample_idx, candidate_position].item())
            candidate_tokens[sample_idx, candidate_idx] = candidate_token
            if candidate_token == NEEDLE_KEY_TOKEN_ID and candidate_position + 1 < sequence_width:
                source_position = candidate_position + 1
                source_token = int(X[sample_idx, source_position].item())
                if (
                    source_position >= query_position
                    or source_position not in valid_candidate_positions
                ):
                    copyable_mask[sample_idx, candidate_idx] = False
                    continue
                source_positions[sample_idx, candidate_idx] = source_position
                source_tokens[sample_idx, candidate_idx] = source_token
                if source_token >= FILLER_TOKEN_START:
                    pair_mask[sample_idx, candidate_idx] = True
            elif candidate_token >= FILLER_TOKEN_START:
                if candidate_position >= query_position:
                    copyable_mask[sample_idx, candidate_idx] = False
                    continue
                source_positions[sample_idx, candidate_idx] = candidate_position
                source_tokens[sample_idx, candidate_idx] = candidate_token
                left_position = candidate_position - 1
                if (
                    left_position >= 0
                    and left_position in valid_candidate_positions
                    and int(X[sample_idx, left_position].item()) == NEEDLE_KEY_TOKEN_ID
                ):
                    pair_mask[sample_idx, candidate_idx] = True
            else:
                copyable_mask[sample_idx, candidate_idx] = False

    raw_copyable_mask = copyable_mask.clone()
    pair_mask &= copyable_mask
    if candidate_filter == "key_value_pair":
        copyable_mask &= pair_mask
    elif candidate_filter == "prefer_key_value_pair":
        rows_with_pair = pair_mask.any(dim=1)
        copyable_mask[rows_with_pair] = pair_mask[rows_with_pair]

    weights = last_layer.get("selected_retrieval_token_weight_by_sample_for_loss")
    if not isinstance(weights, torch.Tensor):
        weights = last_layer.get("selected_retrieval_token_weight_by_sample")
    weights_cpu = None
    if isinstance(weights, torch.Tensor):
        weights_cpu = weights.detach().cpu().float()
        if weights_cpu.dim() == 1:
            weights_cpu = weights_cpu.view(1, -1)
        if weights_cpu.shape != positions_cpu.shape:
            weights_cpu = None

    scores = last_layer.get("selected_retrieval_token_score_by_sample_for_loss")
    if not isinstance(scores, torch.Tensor):
        scores = last_layer.get("selected_retrieval_token_score_by_sample")
    scores_cpu = None
    if isinstance(scores, torch.Tensor):
        scores_cpu = scores.detach().cpu().float()
        if scores_cpu.dim() == 1:
            scores_cpu = scores_cpu.view(1, -1)
        if scores_cpu.shape != positions_cpu.shape:
            scores_cpu = None

    target_mask = (source_tokens == target_cpu.view(-1, 1)) & copyable_mask
    return {
        "positions": positions_cpu.to(device=device),
        "source_positions": source_positions.to(device=device),
        "candidate_token_ids": candidate_tokens.to(device=device),
        "source_token_ids": source_tokens.to(device=device),
        "mask": copyable_mask.to(device=device),
        "raw_mask": raw_copyable_mask.to(device=device),
        "pair_mask": pair_mask.to(device=device),
        "target_mask": target_mask.to(device=device),
        "weights": None if weights_cpu is None else weights_cpu.to(device=device),
        "scores": None if scores_cpu is None else scores_cpu.to(device=device),
        "candidate_filter": candidate_filter,
        "target_value_in_page_candidates": target_value_in_page_candidates.to(device=device),
        "target_pair_in_page_candidates": target_pair_in_page_candidates.to(device=device),
        "target_value_in_top_tokens": target_value_in_top_tokens.to(device=device),
        "target_pair_in_top_tokens": target_pair_in_top_tokens.to(device=device),
        "target_value_in_seed_tokens": target_value_in_seed_tokens.to(device=device),
        "target_pair_in_seed_tokens": target_pair_in_seed_tokens.to(device=device),
    }, None


def compute_retrieval_span_predictor_loss(
    model,
    hidden_query,
    X,
    targets,
    query_positions,
    aux,
    *,
    device,
    candidate_filter=DEFAULT_NIAH_SPAN_CANDIDATE_FILTER,
    loss_mode=DEFAULT_NIAH_SPAN_LOSS_MODE,
):
    """Train the opt-in retrieval span predictor from selected candidates only.

    中文说明:
    - 调用方 / Called by: `run_niah_verification_case` when
      `retrieval_span_predictor_alpha > 0`。
    - 调用对象 / Calls: `_prepare_niah_span_candidate_tensors`,
      `model.score_retrieval_span_candidates`, `F.cross_entropy`。
    - 作用 / Purpose: 让默认关闭的小头学习在模型 selected retrieval 候选中选择能复制出答案的
      key/value span；默认 `single_positive` 保持旧行为，显式 `multi_positive`
      时所有能复制正确 token 的候选都作为正例。它不读取 validation/test 选择信号，
      也不把 gold evidence 位置当输入特征。
    - 返回 / Returns: `(loss, metrics)`；没有可监督候选时返回 0 loss 和 unavailable reason。
    - 错误处理 / Error handling: 缺 aux 或无正例时返回 unavailable，避免默认训练路径失败。
    - 副作用 / Side effects: 只产生 predictor 训练梯度。
    - 关键词 / Keywords:
      niah|span_predictor|retrieval|auxiliary_loss|selected_candidates|训练
    """
    zero = torch.tensor(0.0, device=device)
    candidate_filter = normalize_niah_span_candidate_filter(candidate_filter)
    loss_mode = normalize_niah_span_loss_mode(loss_mode)
    metrics = _default_retrieval_span_metrics(device)
    metrics["candidate_filter"] = candidate_filter
    metrics["loss_mode"] = loss_mode
    if not hasattr(model, "score_retrieval_span_candidates"):
        metrics["unavailable_reason"] = "model_missing_span_predictor_method"
        return zero, metrics
    try:
        candidates, reason = _prepare_niah_span_candidate_tensors(
            X,
            targets,
            query_positions,
            aux,
            device=device,
            candidate_filter=candidate_filter,
        )
    except (RuntimeError, ValueError) as exc:
        metrics["unavailable_reason"] = f"candidate_prepare_error:{exc}"
        return zero, metrics
    if candidates is None:
        metrics["unavailable_reason"] = reason
        return zero, metrics
    target_mask = candidates["target_mask"].to(device=device, dtype=torch.bool)
    valid_rows = target_mask.any(dim=1)
    metrics["hit_rate"] = float(
        target_mask.any(dim=1).to(dtype=torch.float32).mean().detach().cpu().item()
    )
    metrics["positive_count"] = int(
        target_mask.any(dim=1).to(dtype=torch.long).sum().detach().cpu().item()
    )
    if not bool(valid_rows.any().item()):
        metrics["unavailable_reason"] = "target_not_in_selected_candidates"
        return zero, metrics

    logits = model.score_retrieval_span_candidates(
        hidden_query,
        candidates["candidate_token_ids"],
        candidates["source_token_ids"],
        candidates["positions"],
        query_positions,
        candidate_mask=candidates["mask"],
        candidate_weights=candidates["weights"],
        candidate_scores=candidates["scores"],
        candidate_pair_mask=candidates["pair_mask"],
        source_positions=candidates["source_positions"],
    )
    row_indices = valid_rows.nonzero(as_tuple=True)[0]
    candidate_mask = candidates["mask"].to(device=device, dtype=torch.bool)
    positive_indices = target_mask.to(dtype=torch.long).argmax(dim=1)
    if loss_mode == "multi_positive":
        valid_logits = logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)
        positive_logits_for_loss = logits.masked_fill(
            ~target_mask,
            torch.finfo(logits.dtype).min,
        )
        losses = -torch.logsumexp(positive_logits_for_loss[row_indices], dim=1)
        losses = losses + torch.logsumexp(valid_logits[row_indices], dim=1)
        loss = losses.mean()
    else:
        loss = F.cross_entropy(logits[row_indices], positive_indices[row_indices])

    if loss_mode == "multi_positive":
        positive_logits = logits.masked_fill(
            ~target_mask,
            torch.finfo(logits.dtype).min,
        )
        evidence_logits = positive_logits.max(dim=1).values
    else:
        evidence_logits = logits.gather(1, positive_indices.view(-1, 1)).squeeze(1)
    negative_mask = candidate_mask & ~target_mask
    has_negative = negative_mask.any(dim=1)
    negative_logits = logits.masked_fill(
        ~negative_mask,
        torch.finfo(logits.dtype).min,
    )
    best_negative_logits = torch.where(
        has_negative,
        negative_logits.max(dim=1).values,
        torch.zeros_like(evidence_logits),
    )
    ranks = (
        (negative_mask & (logits > evidence_logits.unsqueeze(1)))
        .to(dtype=torch.long)
        .sum(dim=1)
        + 1
    )
    top1_rows = evidence_logits >= best_negative_logits
    metrics.update(
        {
            "available": True,
            "unavailable_reason": None,
            "loss": float(loss.detach().cpu().item()),
            "candidate_filter": candidate_filter,
            "loss_mode": loss_mode,
            "target_rank_mean": float(
                ranks[valid_rows].to(dtype=torch.float32).mean().detach().cpu().item()
            ),
            "top1_rate": float(
                top1_rows[valid_rows].to(dtype=torch.float32).mean().detach().cpu().item()
            ),
            "evidence_logit_mean": float(
                evidence_logits[valid_rows].mean().detach().cpu().item()
            ),
            "best_negative_logit_mean": float(
                best_negative_logits[valid_rows].mean().detach().cpu().item()
            ),
            "logit_margin_mean": float(
                (evidence_logits - best_negative_logits)[valid_rows]
                .mean()
                .detach()
                .cpu()
                .item()
            ),
        }
    )
    return loss, metrics


def compute_niah_span_predictor_sample_metrics(
    model,
    hidden_query,
    X,
    targets,
    query_positions,
    aux,
    seq_len,
    depth,
    sample_index_start=0,
    candidate_filter=DEFAULT_NIAH_SPAN_CANDIDATE_FILTER,
):
    """Evaluate NIAH by selecting a retrieved span with the opt-in predictor."""
    candidate_filter = normalize_niah_span_candidate_filter(candidate_filter)
    if X.dim() != 2:
        raise ValueError(f"expected token ids with shape [B, SeqLen], got {tuple(X.shape)}")
    if targets.dim() != 1:
        raise ValueError(f"expected targets with shape [B], got {tuple(targets.shape)}")
    batch_size, sequence_width = X.shape
    targets_cpu = targets.detach().cpu()
    query_positions_cpu = query_positions.detach().cpu()
    denominator = max(seq_len - 1, 1)
    unavailable_reason = None
    if not hasattr(model, "score_retrieval_span_candidates"):
        unavailable_reason = "model_missing_span_predictor_method"
        candidates = None
    else:
        candidates, unavailable_reason = _prepare_niah_span_candidate_tensors(
            X,
            targets,
            query_positions,
            aux,
            device=hidden_query.device,
            candidate_filter=candidate_filter,
        )
    logits = None
    if candidates is not None:
        try:
            logits = model.score_retrieval_span_candidates(
                hidden_query,
                candidates["candidate_token_ids"],
                candidates["source_token_ids"],
                candidates["positions"],
                query_positions,
                candidate_mask=candidates["mask"],
                candidate_weights=candidates["weights"],
                candidate_scores=candidates["scores"],
                candidate_pair_mask=candidates["pair_mask"],
                source_positions=candidates["source_positions"],
            ).detach().cpu()
        except (RuntimeError, ValueError) as exc:
            unavailable_reason = f"span_predictor_error:{exc}"
            logits = None

    sample_rows = []
    for sample_offset in range(batch_size):
        target_token = int(targets_cpu[sample_offset].item())
        query_position = int(query_positions_cpu[sample_offset].item())
        pred_token = PAD_TOKEN_ID
        readout_available = unavailable_reason is None and logits is not None
        row_unavailable_reason = unavailable_reason
        copied_from_position = None
        copied_candidate_position = None
        target_candidate_present = False
        target_candidate_rank = None
        pred_prob = 0.0
        span_valid_candidate_count = 0
        span_raw_candidate_count = 0
        span_pair_candidate_count = 0
        span_pair_available = False
        span_filter_fallback = False
        span_selected_is_pair = False
        span_target_pair_candidate_present = False
        span_target_value_in_page_candidates = False
        span_target_pair_in_page_candidates = False
        span_target_value_in_top_tokens = False
        span_target_pair_in_top_tokens = False
        span_target_value_in_seed_tokens = False
        span_target_pair_in_seed_tokens = False
        if readout_available and candidates is not None:
            mask_row = candidates["mask"][sample_offset].detach().cpu().bool()
            raw_mask_row = candidates.get("raw_mask", candidates["mask"])[
                sample_offset
            ].detach().cpu().bool()
            pair_mask_row = candidates.get("pair_mask", candidates["mask"])[
                sample_offset
            ].detach().cpu().bool()
            span_valid_candidate_count = int(mask_row.to(dtype=torch.long).sum().item())
            span_raw_candidate_count = int(raw_mask_row.to(dtype=torch.long).sum().item())
            span_pair_candidate_count = int(pair_mask_row.to(dtype=torch.long).sum().item())
            span_pair_available = span_pair_candidate_count > 0
            span_filter_fallback = (
                candidate_filter == "prefer_key_value_pair"
                and span_pair_candidate_count == 0
                and span_valid_candidate_count > 0
            )
            span_target_value_in_page_candidates = bool(
                candidates.get("target_value_in_page_candidates")[
                    sample_offset
                ].detach().cpu().item()
            )
            span_target_pair_in_page_candidates = bool(
                candidates.get("target_pair_in_page_candidates")[
                    sample_offset
                ].detach().cpu().item()
            )
            span_target_value_in_top_tokens = bool(
                candidates.get("target_value_in_top_tokens")[
                    sample_offset
                ].detach().cpu().item()
            )
            span_target_pair_in_top_tokens = bool(
                candidates.get("target_pair_in_top_tokens")[
                    sample_offset
                ].detach().cpu().item()
            )
            span_target_value_in_seed_tokens = bool(
                candidates.get("target_value_in_seed_tokens")[
                    sample_offset
                ].detach().cpu().item()
            )
            span_target_pair_in_seed_tokens = bool(
                candidates.get("target_pair_in_seed_tokens")[
                    sample_offset
                ].detach().cpu().item()
            )
            if not bool(mask_row.any().item()):
                readout_available = False
                row_unavailable_reason = "no_valid_retrieval_candidates"
            else:
                logits_row = logits[sample_offset].masked_fill(~mask_row, float("-inf"))
                chosen_index = int(torch.argmax(logits_row).item())
                valid_indices = mask_row.nonzero(as_tuple=True)[0]
                sorted_indices = valid_indices[
                    torch.argsort(logits_row[valid_indices], descending=True)
                ]
                source_tokens = candidates["source_token_ids"][sample_offset].detach().cpu().long()
                source_positions = candidates["source_positions"][sample_offset].detach().cpu().long()
                candidate_positions = candidates["positions"][sample_offset].detach().cpu().long()
                for rank_offset, candidate_index_tensor in enumerate(sorted_indices):
                    candidate_index = int(candidate_index_tensor.item())
                    if int(source_tokens[candidate_index].item()) == target_token:
                        target_candidate_present = True
                        target_candidate_rank = rank_offset + 1
                        span_target_pair_candidate_present = bool(
                            pair_mask_row[candidate_index].item()
                        )
                        break
                copied_candidate_position = int(candidate_positions[chosen_index].item())
                copied_from_position = int(source_positions[chosen_index].item())
                span_selected_is_pair = bool(pair_mask_row[chosen_index].item())
                if 0 <= copied_from_position < sequence_width:
                    pred_token = int(X[sample_offset, copied_from_position].item())
                    probs = torch.softmax(logits_row[valid_indices], dim=0)
                    chosen_valid_rank = (valid_indices == chosen_index).nonzero(as_tuple=True)[0]
                    pred_prob = (
                        float(probs[int(chosen_valid_rank[0].item())].item())
                        if chosen_valid_rank.numel()
                        else 0.0
                    )
                else:
                    readout_available = False
                    row_unavailable_reason = "source_position_out_of_range"

        correct = readout_available and pred_token == target_token
        sample_rows.append(
            {
                "sample_index": sample_index_start + sample_offset,
                "depth": float(depth),
                "query_position": query_position,
                "query_position_ratio": float(query_position / denominator),
                "target_token": target_token,
                "pred_token": int(pred_token),
                "correct": bool(correct),
                "top3_correct": bool(correct),
                "top5_correct": bool(correct),
                "target_rank": 1 if correct else 2,
                "target_prob": 1.0 if correct else 0.0,
                "pred_prob": pred_prob if readout_available else 0.0,
                "loss": 0.0 if correct else 1.0,
                "logit_margin": 1.0 if correct else -1.0,
                "prob_margin": 1.0 if correct else 0.0,
                "entropy": 0.0,
                "readout_mode": "span_predictor",
                "readout_available": bool(readout_available),
                "readout_unavailable_reason": row_unavailable_reason,
                "copied_from_position": copied_from_position,
                "copied_candidate_position": copied_candidate_position,
                "target_candidate_present": bool(target_candidate_present),
                "target_candidate_rank": target_candidate_rank,
                "span_candidate_filter": candidate_filter,
                "span_valid_candidate_count": span_valid_candidate_count,
                "span_raw_candidate_count": span_raw_candidate_count,
                "span_pair_candidate_count": span_pair_candidate_count,
                "span_pair_available": bool(span_pair_available),
                "span_filter_fallback": bool(span_filter_fallback),
                "span_selected_is_pair": bool(span_selected_is_pair),
                "span_target_pair_candidate_present": bool(
                    span_target_pair_candidate_present
                ),
                "span_target_value_in_page_candidates": bool(
                    span_target_value_in_page_candidates
                ),
                "span_target_pair_in_page_candidates": bool(
                    span_target_pair_in_page_candidates
                ),
                "span_target_value_in_top_tokens": bool(
                    span_target_value_in_top_tokens
                ),
                "span_target_pair_in_top_tokens": bool(
                    span_target_pair_in_top_tokens
                ),
                "span_target_value_in_seed_tokens": bool(
                    span_target_value_in_seed_tokens
                ),
                "span_target_pair_in_seed_tokens": bool(
                    span_target_pair_in_seed_tokens
                ),
            }
        )
    return sample_rows


def summarize_retrieval_evidence_step_metrics(step_rows):
    """Summarize train-time retrieval evidence diagnostics across observed steps.

    中文说明:
    - 调用方 / Called by: `run_niah_verification_case`
    - 调用对象 / Calls: local numeric helpers only
    - 作用 / Purpose: final step 的 evidence hit/rank 可能刚好没有命中；本函数聚合整段训练
      的 evidence 诊断，帮助判断辅助监督是否曾经拿到可学习信号。
    - 参数 / Parameters: `step_rows` 是训练循环每步写入报告的行。
    - 返回 / Returns: dict，包含 available step 数、hit/top1/rank/margin/loss 均值。
    - 错误处理 / Error handling: 缺字段或 None 会被跳过，不抛错影响训练报告。
    - 副作用 / Side effects: 无。
    - 关键词 / Keywords:
      niah|retrieval_evidence|summary|hit_rate|rank|margin|training|诊断

    English documentation:
    Function name:
        summarize_retrieval_evidence_step_metrics
    Purpose:
        Aggregate train-time evidence diagnostics so reports are not interpreted
        only from the final optimizer step.
    """
    rows = list(step_rows or [])
    available_rows = [
        row for row in rows if bool(row.get("retrieval_evidence_available"))
    ]

    def _mean_field(field):
        values = [
            float(row[field])
            for row in available_rows
            if row.get(field) is not None
        ]
        return None if not values else sum(values) / len(values)

    return {
        "steps": len(rows),
        "available_steps": len(available_rows),
        "positive_steps": sum(
            1
            for row in available_rows
            if float(row.get("retrieval_evidence_positive_count") or 0.0) > 0.0
        ),
        "mean_hit_rate": _mean_field("retrieval_evidence_hit_rate"),
        "mean_evidence_weight": _mean_field("retrieval_evidence_weight_mean"),
        "mean_best_negative_weight": _mean_field(
            "retrieval_evidence_best_negative_weight_mean"
        ),
        "mean_evidence_margin": _mean_field("retrieval_evidence_margin_mean"),
        "mean_target_rank": _mean_field("retrieval_evidence_target_rank_mean"),
        "mean_top1_rate": _mean_field("retrieval_evidence_top1_rate"),
        "mean_ranking_loss": _mean_field("retrieval_evidence_ranking_loss"),
        "mean_margin_loss": _mean_field("retrieval_evidence_margin_loss"),
        "mean_score_margin_loss": _mean_field(
            "retrieval_evidence_score_margin_loss"
        ),
        "mean_evidence_score": _mean_field("retrieval_evidence_score_mean"),
        "mean_best_negative_score": _mean_field(
            "retrieval_evidence_best_negative_score_mean"
        ),
        "mean_evidence_score_margin": _mean_field(
            "retrieval_evidence_score_margin_mean"
        ),
        "mean_score_target_rank": _mean_field(
            "retrieval_evidence_score_target_rank_mean"
        ),
        "mean_score_top1_rate": _mean_field("retrieval_evidence_score_top1_rate"),
        "mean_gate_loss": _mean_field("retrieval_evidence_gate_loss"),
    }


def summarize_query_evidence_alignment_step_metrics(step_rows):
    """Summarize train-time query/evidence representation alignment diagnostics.

    中文说明:
    - 调用方 / Called by: `run_niah_verification_case`
    - 调用对象 / Calls: local numeric helpers only
    - 作用 / Purpose: 将显式开启的 query/evidence hidden 对齐辅助目标汇总到报告，
      便于判断该目标是否真的推动 query 表征靠近证据 embedding。
    - 参数 / Parameters: `step_rows` 是训练循环每步写入报告的行。
    - 返回 / Returns: dict，包含 available step 数、loss/cosine/mse 均值。
    - 错误处理 / Error handling: 缺字段或 None 会被跳过。
    - 副作用 / Side effects: 无。
    - 关键词 / Keywords:
      niah|query_evidence_alignment|hidden|embedding|cosine|mse|summary|表征

    English documentation:
    Function name:
        summarize_query_evidence_alignment_step_metrics
    Purpose:
        Aggregate optional query/evidence representation-alignment diagnostics.
    """
    rows = list(step_rows or [])
    available_rows = [
        row for row in rows if bool(row.get("query_evidence_alignment_available"))
    ]

    def _mean_field(field):
        values = [
            float(row[field])
            for row in available_rows
            if row.get(field) is not None
        ]
        return None if not values else sum(values) / len(values)

    return {
        "steps": len(rows),
        "available_steps": len(available_rows),
        "mean_loss": _mean_field("query_evidence_alignment_loss"),
        "mean_cosine": _mean_field("query_evidence_alignment_mean_cosine"),
        "mean_mse": _mean_field("query_evidence_alignment_mean_mse"),
    }


def summarize_retrieval_span_predictor_step_metrics(step_rows):
    """Summarize train-time retrieval span predictor diagnostics.

    中文说明:
    - 调用方 / Called by: `run_niah_verification_case`.
    - 调用对象 / Calls: local numeric helpers only.
    - 作用 / Purpose: 聚合默认关闭的 span predictor 辅助目标，判断它是否真的学会
      在 selected retrieval 候选里选择可复制答案的 span。
    - 参数 / Parameters: `step_rows` 是训练循环每步写入报告的行。
    - 返回 / Returns: dict，包含可用步数、命中率、rank、top1、loss 和 logit margin。
    - 错误处理 / Error handling: 缺失字段会被跳过，不影响实验报告生成。
    - 副作用 / Side effects: 无。
    - 关键词 / Keywords:
      niah|retrieval_span_predictor|summary|rank|top1|候选选择
    """
    rows = list(step_rows or [])
    available_rows = [
        row for row in rows if bool(row.get("retrieval_span_predictor_available"))
    ]

    def _mean_field(field):
        values = [
            float(row[field])
            for row in available_rows
            if row.get(field) is not None
        ]
        return None if not values else sum(values) / len(values)

    return {
        "steps": len(rows),
        "available_steps": len(available_rows),
        "positive_steps": sum(
            1
            for row in available_rows
            if float(row.get("retrieval_span_predictor_positive_count") or 0.0) > 0.0
        ),
        "mean_loss": _mean_field("retrieval_span_predictor_loss"),
        "mean_hit_rate": _mean_field("retrieval_span_predictor_hit_rate"),
        "mean_target_rank": _mean_field("retrieval_span_predictor_target_rank_mean"),
        "mean_top1_rate": _mean_field("retrieval_span_predictor_top1_rate"),
        "mean_evidence_logit": _mean_field("retrieval_span_predictor_evidence_logit_mean"),
        "mean_best_negative_logit": _mean_field(
            "retrieval_span_predictor_best_negative_logit_mean"
        ),
        "mean_logit_margin": _mean_field("retrieval_span_predictor_logit_margin_mean"),
    }


def build_niah_model(
    device,
    vocab_size=100,
    dim=64,
    num_layers=2,
    K=64,
    kr=8,
    chunk_size=256,
    model_type="mhdsra2",
    mhdsra2_config_override=None,
    use_retrieval: bool = False,
    use_retrieval_span_predictor: bool = False,
    retrieval_span_structure_features: bool = False,
):
    """Build the long-context Needle-In-A-Haystack benchmark model.

    中文说明:
    - 调用方 / Called by: `run_single_niah_test`, `run_single_niah_capacity_test`,
      `scripts.next_round_benchmark_runner.run_next_round_benchmark`
    - 调用对象 / Calls: `normalize_model_type`, `MultiLayerMHDSRA2Model`
    - 作用 / Purpose: 统一构造 NIAH 基准模型；归档的 `dsra` 名称会经领域层别名映射到
      当前 active 架构 `mhdsra2`
    - 变量 / Variables:
      `model_type` 支持 `dsra/mhdsra2`, 其余参数为维度、层数、槽位与 chunk 配置；
      `use_retrieval` 显式决定是否启用外部分页召回分支，默认 False 保持旧 NIAH 行为；
      `use_retrieval_span_predictor` 只给显式实验组创建候选 span 打分头；
      `retrieval_span_structure_features` 默认关闭，只让显式结构组把 pair/source 提示
      送入 span predictor。
    - 接入 / Integration: 新增长上下文模型时优先扩展本函数，避免分散在多个训练入口
    - 错误处理 / Error handling: 未知 `model_type` 抛出 `ValueError`
    - 关键词 / Keywords:
      niah|build_model|dsra|mhdsra2|factory|benchmark|needle|haystack|long_context|构建
    """
    active_model_type = normalize_model_type(model_type)
    if active_model_type == "mhdsra2":
        return MultiLayerMHDSRA2Model(
            vocab_size,
            dim,
            num_layers,
            K,
            kr,
            chunk_size,
            use_retrieval=use_retrieval,
            use_retrieval_span_predictor=use_retrieval_span_predictor,
            retrieval_span_structure_features=retrieval_span_structure_features,
            mhdsra2_config_override=mhdsra2_config_override,
        ).to(device)
    raise ValueError(f"Unsupported model_type: {model_type} (normalized: {active_model_type})")


def _mean(values):
    """Return the arithmetic mean for a non-empty numeric sequence.

    中文说明:
    - 调用方 / Called by: `summarize_niah_sample_metrics`
    - 调用对象 / Calls: `sum`, `len`
    - 作用 / Purpose: 为 NIAH 诊断指标提供统一均值口径
    - 参数 / Parameters: `values` 是非空数值序列
    - 返回 / Returns: float 均值
    - 错误处理 / Error handling: 空序列由调用方避免；若为空会触发除零错误
    - 副作用 / Side effects: 无
    - 事务边界 / Transaction boundary: 不涉及事务
    - 并发与幂等 / Concurrency and idempotency: 纯函数
    - 关键词 / Keywords:
      mean|metric|niah|summary|sample|aggregate|float|helper|均值|指标

    English documentation:
    Function name:
        _mean
    Purpose:
        Compute a shared arithmetic mean for NIAH metric summaries.
    Called by:
        `summarize_niah_sample_metrics`.
    Calls:
        `sum` and `len`.
    Parameters:
        - values: non-empty numeric sequence.
    Returns:
        Mean value as float.
    Error handling:
        Empty inputs are prevented by callers.
    Side effects:
        None.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Pure and deterministic.
    English keywords:
        mean, metric, niah, summary, sample, aggregate, float, helper, average, diagnostics
    """
    return float(sum(values) / len(values))


def _median(values):
    """Return the median for a non-empty numeric sequence.

    中文说明:
    - 调用方 / Called by: `summarize_niah_sample_metrics`
    - 调用对象 / Calls: `sorted`, `len`
    - 作用 / Purpose: 为 target rank 等偏态指标提供比均值更稳健的摘要
    - 参数 / Parameters: `values` 是非空数值序列
    - 返回 / Returns: float 中位数
    - 错误处理 / Error handling: 空序列由调用方避免；若为空会触发索引错误
    - 副作用 / Side effects: 无
    - 事务边界 / Transaction boundary: 不涉及事务
    - 并发与幂等 / Concurrency and idempotency: 纯函数
    - 关键词 / Keywords:
      median|rank|metric|niah|summary|sample|aggregate|helper|中位数|指标

    English documentation:
    Function name:
        _median
    Purpose:
        Compute a median summary for rank-like NIAH diagnostic metrics.
    Called by:
        `summarize_niah_sample_metrics`.
    Calls:
        `sorted` and `len`.
    Parameters:
        - values: non-empty numeric sequence.
    Returns:
        Median value as float.
    Error handling:
        Empty inputs are prevented by callers.
    Side effects:
        None.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Pure and deterministic.
    English keywords:
        median, rank, metric, niah, summary, sample, aggregate, helper, robust, diagnostics
    """
    ordered_values = sorted(float(value) for value in values)
    midpoint = len(ordered_values) // 2
    if len(ordered_values) % 2 == 1:
        return ordered_values[midpoint]
    return (ordered_values[midpoint - 1] + ordered_values[midpoint]) / 2.0


def compute_selected_logits_sample_metrics(
    logits_target,
    targets,
    query_positions,
    seq_len,
    depth,
    sample_index_start=0,
):
    """Compute per-sample NIAH diagnostics from selected query logits.

    中文说明:
    - 调用方 / Called by: `evaluate_niah_depths`, unit tests
    - 调用对象 / Calls: `torch.softmax`, `torch.topk`, `F.cross_entropy`
    - 作用 / Purpose: 在不额外 forward、不保存完整 2M token 或完整 logits 的前提下，
      从 query 位置 logits 计算样本级 rank、top-k、置信度、margin、entropy 与 loss
    - 参数 / Parameters:
      `logits_target` 是 `[batch, vocab]` query logits；`targets` 是 `[batch]` 正确 token；
      `query_positions` 是 CPU query 位置；`seq_len` 是上下文长度；`depth` 是 NIAH depth；
      `sample_index_start` 是本次 eval 内样本编号起点
    - 返回 / Returns: list[dict]，每个元素是可写 JSON/SwanLab 的样本级标量指标
    - 内部关键变量 / Internal variables:
      `target_ranks` 是 1-based 正确 token 排名；`logit_margins` 衡量目标 token 相对最佳非目标 token；
      `prob_margins` 衡量 top1 与 top2 预测概率差
    - 接入 / Integration: 仅接收 selected logits，避免长上下文评估回退到全序列 logits
    - 错误处理 / Error handling: logits/targets 形状不匹配时抛出 `ValueError`
    - 副作用 / Side effects: 无；不写文件、不访问网络、不修改输入张量
    - 事务边界 / Transaction boundary: 不涉及 Unit of Work、数据库或文件事务
    - 并发与幂等 / Concurrency and idempotency: 对同一 logits/targets 可重复调用
    - 关键词 / Keywords:
      sample_metric|target_rank|topk|confidence|margin|entropy|niah|logits|诊断|样本

    English documentation:
    Function name:
        compute_selected_logits_sample_metrics
    Purpose:
        Compute sample-level NIAH diagnostics from selected query logits without
        storing full contexts or full-sequence logits.
    Called by:
        `evaluate_niah_depths` and tests.
    Calls:
        `torch.softmax`, `torch.topk`, and `F.cross_entropy`.
    Parameters:
        - logits_target: `[batch, vocab]` query logits.
        - targets: `[batch]` target token ids.
        - query_positions: CPU query positions.
        - seq_len: sequence length.
        - depth: NIAH depth ratio.
        - sample_index_start: first sample index for this eval batch.
    Returns:
        List of JSON/SwanLab-safe scalar metric dictionaries.
    Error handling:
        Raises `ValueError` for incompatible shapes.
    Side effects:
        None.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Pure tensor-derived metrics for the same inputs.
    English keywords:
        sample_metric, target_rank, topk, confidence, margin, entropy, niah, logits, diagnostics, sample
    """
    if logits_target.dim() != 2:
        raise ValueError(f"expected selected logits with shape [B, V], got {tuple(logits_target.shape)}")
    if targets.dim() != 1:
        raise ValueError(f"expected targets with shape [B], got {tuple(targets.shape)}")
    if logits_target.shape[0] != targets.shape[0]:
        raise ValueError("logits batch size must match targets")

    batch_size, vocab_size = logits_target.shape
    safe_topk = min(5, vocab_size)
    probs = torch.softmax(logits_target, dim=-1)
    per_sample_losses = F.cross_entropy(logits_target, targets, ignore_index=PAD_TOKEN_ID, reduction="none")
    topk_probs, topk_indices = torch.topk(probs, k=safe_topk, dim=-1)
    pred_tokens = topk_indices[:, 0]
    pred_probs = topk_probs[:, 0]
    target_column = targets.view(-1, 1)
    target_logits = logits_target.gather(1, target_column).squeeze(1)
    target_probs = probs.gather(1, target_column).squeeze(1)
    target_ranks = (logits_target > target_logits.unsqueeze(1)).sum(dim=-1) + 1
    entropy_values = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)

    if vocab_size > 1:
        non_target_logits = logits_target.clone()
        non_target_logits.scatter_(1, target_column, float("-inf"))
        best_non_target_logits = non_target_logits.max(dim=-1).values
        top2_probs = topk_probs[:, 1] if safe_topk > 1 else torch.zeros_like(pred_probs)
    else:
        best_non_target_logits = torch.zeros_like(target_logits)
        top2_probs = torch.zeros_like(pred_probs)
    logit_margins = target_logits - best_non_target_logits
    prob_margins = pred_probs - top2_probs
    query_positions_cpu = query_positions.detach().cpu()
    denominator = max(seq_len - 1, 1)

    sample_rows = []
    for sample_offset in range(batch_size):
        target_token = int(targets[sample_offset].item())
        top3_limit = min(3, safe_topk)
        top5_limit = min(5, safe_topk)
        top3_correct = bool((topk_indices[sample_offset, :top3_limit] == target_token).any().item())
        top5_correct = bool((topk_indices[sample_offset, :top5_limit] == target_token).any().item())
        query_position = int(query_positions_cpu[sample_offset].item())
        sample_rows.append(
            {
                "sample_index": sample_index_start + sample_offset,
                "depth": float(depth),
                "query_position": query_position,
                "query_position_ratio": float(query_position / denominator),
                "target_token": target_token,
                "pred_token": int(pred_tokens[sample_offset].item()),
                "correct": bool(pred_tokens[sample_offset].item() == target_token),
                "top3_correct": top3_correct,
                "top5_correct": top5_correct,
                "target_rank": int(target_ranks[sample_offset].item()),
                "target_prob": float(target_probs[sample_offset].item()),
                "pred_prob": float(pred_probs[sample_offset].item()),
                "loss": float(per_sample_losses[sample_offset].item()),
                "logit_margin": float(logit_margins[sample_offset].item()),
                "prob_margin": float(prob_margins[sample_offset].item()),
                "entropy": float(entropy_values[sample_offset].item()),
            }
        )
    return sample_rows


def compute_niah_needle_copy_readout_sample_metrics(
    X,
    targets,
    query_positions,
    aux,
    seq_len,
    depth,
    sample_index_start=0,
    readout_mode="needle_copy",
):
    """Compute NIAH sample metrics by copying from predicted retrieval evidence.

    中文说明:
    - 调用方 / Called by: `evaluate_niah_depths` when `niah_readout_mode` is
      `needle_copy` or `needle_pair_copy`
    - 调用对象 / Calls: tensor indexing, `torch.argmax`
    - 作用 / Purpose: 只使用模型在 query 位置暴露的 selected retrieval 候选和权重，
      选出最高权重候选；若候选是 needle key，则复制后一个 token；若候选本身是非保留
      value token，则直接复制它。`needle_pair_copy` 会先在候选中寻找 key/right-value 成对
      结构，再退回 `needle_copy`，用于验证“候选覆盖 + 结构化复制”是否足够回答 NIAH。
    - 参数 / Parameters:
      `X` 是 CPU token tensor；`targets` 是正确答案 token，仅用于评价；`aux` 是
      `forward_selected_logits(..., return_aux=True)` 返回的轻量诊断。
    - 返回 / Returns: 与 `compute_selected_logits_sample_metrics` 兼容的样本级指标列表。
    - 错误处理 / Error handling: 缺少 selected retrieval metadata/weights 或形状不匹配时
      返回 unavailable 样本指标，而不是猜测或读取 gold answer。
    - 副作用 / Side effects: 无；不修改模型、不写文件。
    - 关键词 / Keywords:
      niah|needle_copy|retrieval_metadata|selected_aux|copy_readout|证据复制

    English documentation:
    Function name:
        compute_niah_needle_copy_readout_sample_metrics
    Purpose:
        Evaluate explicit NIAH copy readouts from predicted selected retrieval
        candidates without using labels to construct predictions.
    """
    readout_mode = normalize_niah_readout_mode(readout_mode)
    if readout_mode not in {"needle_copy", "needle_pair_copy"}:
        raise ValueError(f"copy readout metrics do not support mode: {readout_mode!r}")
    if X.dim() != 2:
        raise ValueError(f"expected token ids with shape [B, SeqLen], got {tuple(X.shape)}")
    if targets.dim() != 1:
        raise ValueError(f"expected targets with shape [B], got {tuple(targets.shape)}")
    if X.shape[0] != targets.shape[0]:
        raise ValueError("token batch size must match targets")

    batch_size, sequence_width = X.shape
    query_positions_cpu = query_positions.detach().cpu()
    targets_cpu = targets.detach().cpu()
    denominator = max(seq_len - 1, 1)

    last_layer = aux.get("last_layer") if isinstance(aux, dict) else None
    selected_metadata = (
        last_layer.get("selected_retrieval_metadata")
        if isinstance(last_layer, dict)
        else None
    )
    weights = (
        last_layer.get("selected_retrieval_token_weight_by_sample")
        if isinstance(last_layer, dict)
        else None
    )

    unavailable_reason = None
    if not isinstance(selected_metadata, dict):
        unavailable_reason = "missing_selected_retrieval_metadata"
    elif not isinstance(weights, torch.Tensor):
        unavailable_reason = "missing_selected_retrieval_weights"

    positions = None
    mask = None
    if unavailable_reason is None:
        positions = selected_metadata.get("positions")
        mask = selected_metadata.get("mask")
        if not isinstance(positions, torch.Tensor) or not isinstance(mask, torch.Tensor):
            unavailable_reason = "missing_selected_positions_or_mask"

    if unavailable_reason is None:
        positions = positions.detach().cpu().long()
        mask = mask.detach().cpu().bool()
        weights = weights.detach().cpu().float()
        if positions.dim() == 1:
            positions = positions.view(1, -1)
            mask = mask.view(1, -1)
        if weights.dim() == 1:
            weights = weights.view(1, -1)
        if positions.shape != mask.shape or positions.shape != weights.shape:
            unavailable_reason = "selected_retrieval_shape_mismatch"
        elif positions.shape[0] != batch_size:
            unavailable_reason = "selected_retrieval_batch_mismatch"

    sample_rows = []
    for sample_offset in range(batch_size):
        target_token = int(targets_cpu[sample_offset].item())
        query_position = int(query_positions_cpu[sample_offset].item())
        copied_token = PAD_TOKEN_ID
        copied_from_position = None
        copied_candidate_position = None
        copied_candidate_token = None
        target_candidate_present = False
        target_candidate_rank = None
        pair_candidate_position = None
        pair_neighbor_position = None
        pair_neighbor_present = False
        readout_available = unavailable_reason is None
        row_unavailable_reason = unavailable_reason

        if readout_available:
            row_mask = mask[sample_offset]
            if not bool(row_mask.any().item()):
                readout_available = False
                row_unavailable_reason = "no_valid_retrieval_candidates"
            else:
                row_weights = weights[sample_offset].masked_fill(~row_mask, float("-inf"))
                best_candidate_index = int(torch.argmax(row_weights).item())
                valid_candidate_indices = row_mask.nonzero(as_tuple=True)[0]
                sorted_candidate_indices = valid_candidate_indices[
                    torch.argsort(row_weights[valid_candidate_indices], descending=True)
                ]
                valid_candidate_positions = {
                    int(positions[sample_offset, index_tensor].item())
                    for index_tensor in valid_candidate_indices
                }
                for rank_offset, candidate_index_tensor in enumerate(sorted_candidate_indices):
                    candidate_index = int(candidate_index_tensor.item())
                    candidate_pos_for_diag = int(positions[sample_offset, candidate_index].item())
                    if 0 <= candidate_pos_for_diag < sequence_width:
                        candidate_token_for_diag = int(
                            X[sample_offset, candidate_pos_for_diag].item()
                        )
                        if candidate_token_for_diag == target_token:
                            target_candidate_present = True
                            target_candidate_rank = rank_offset + 1
                            break
                        if (
                            candidate_token_for_diag == NEEDLE_KEY_TOKEN_ID
                            and candidate_pos_for_diag + 1 < sequence_width
                            and int(X[sample_offset, candidate_pos_for_diag + 1].item())
                            == target_token
                        ):
                            target_candidate_present = True
                            target_candidate_rank = rank_offset + 1
                            break
                selected_candidate_index = best_candidate_index
                if readout_mode == "needle_pair_copy":
                    preferred_pair_key_index = None
                    fallback_pair_key_index = None
                    preferred_pair_value_index = None
                    for candidate_index_tensor in sorted_candidate_indices:
                        candidate_index = int(candidate_index_tensor.item())
                        candidate_position_for_pair = int(
                            positions[sample_offset, candidate_index].item()
                        )
                        if not 0 <= candidate_position_for_pair < sequence_width:
                            continue
                        candidate_token_for_pair = int(
                            X[sample_offset, candidate_position_for_pair].item()
                        )
                        if (
                            candidate_position_for_pair > 0
                            and candidate_token_for_pair >= FILLER_TOKEN_START
                            and int(X[sample_offset, candidate_position_for_pair - 1].item())
                            == NEEDLE_KEY_TOKEN_ID
                        ):
                            preferred_pair_value_index = candidate_index
                            break
                        if candidate_position_for_pair >= sequence_width - 1:
                            continue
                        right_neighbor_position = candidate_position_for_pair + 1
                        right_neighbor_token = int(
                            X[sample_offset, right_neighbor_position].item()
                        )
                        if (
                            candidate_token_for_pair == NEEDLE_KEY_TOKEN_ID
                            and right_neighbor_token >= FILLER_TOKEN_START
                        ):
                            if fallback_pair_key_index is None:
                                fallback_pair_key_index = candidate_index
                            if right_neighbor_position in valid_candidate_positions:
                                preferred_pair_key_index = candidate_index
                                break
                    selected_pair_candidate_index = (
                        preferred_pair_value_index
                        if preferred_pair_value_index is not None
                        else (
                            preferred_pair_key_index
                            if preferred_pair_key_index is not None
                            else fallback_pair_key_index
                        )
                    )
                    if selected_pair_candidate_index is not None:
                        selected_candidate_index = selected_pair_candidate_index
                        pair_candidate_position = int(
                            positions[sample_offset, selected_candidate_index].item()
                        )
                        if (
                            pair_candidate_position > 0
                            and int(X[sample_offset, pair_candidate_position].item())
                            >= FILLER_TOKEN_START
                            and int(X[sample_offset, pair_candidate_position - 1].item())
                            == NEEDLE_KEY_TOKEN_ID
                        ):
                            pair_neighbor_position = pair_candidate_position - 1
                        else:
                            pair_neighbor_position = pair_candidate_position + 1
                        pair_neighbor_present = pair_neighbor_position in valid_candidate_positions

                candidate_position = int(
                    positions[sample_offset, selected_candidate_index].item()
                )
                copied_candidate_position = candidate_position
                if 0 <= candidate_position < sequence_width:
                    candidate_token = int(X[sample_offset, candidate_position].item())
                    copied_candidate_token = candidate_token
                    if (
                        candidate_token == NEEDLE_KEY_TOKEN_ID
                        and candidate_position + 1 < sequence_width
                    ):
                        copied_from_position = candidate_position + 1
                        copied_token = int(X[sample_offset, copied_from_position].item())
                    elif candidate_token >= FILLER_TOKEN_START:
                        copied_from_position = candidate_position
                        copied_token = candidate_token
                    else:
                        readout_available = False
                        row_unavailable_reason = "candidate_is_not_copyable"
                else:
                    readout_available = False
                    row_unavailable_reason = "candidate_position_out_of_range"

        correct = readout_available and copied_token == target_token
        loss = 0.0 if correct else 1.0
        sample_rows.append(
            {
                "sample_index": sample_index_start + sample_offset,
                "depth": float(depth),
                "query_position": query_position,
                "query_position_ratio": float(query_position / denominator),
                "target_token": target_token,
                "pred_token": int(copied_token),
                "correct": bool(correct),
                "top3_correct": bool(correct),
                "top5_correct": bool(correct),
                "target_rank": 1 if correct else 2,
                "target_prob": 1.0 if correct else 0.0,
                "pred_prob": 1.0 if readout_available else 0.0,
                "loss": loss,
                "logit_margin": 1.0 if correct else -1.0,
                "prob_margin": 1.0 if correct else 0.0,
                "entropy": 0.0,
                "readout_mode": readout_mode,
                "readout_available": bool(readout_available),
                "readout_unavailable_reason": row_unavailable_reason,
                "copied_from_position": copied_from_position,
                "copied_candidate_position": copied_candidate_position,
                "copied_candidate_token": copied_candidate_token,
                "pair_candidate_position": pair_candidate_position,
                "pair_neighbor_position": pair_neighbor_position,
                "pair_neighbor_present": bool(pair_neighbor_present),
                "target_candidate_present": bool(target_candidate_present),
                "target_candidate_rank": target_candidate_rank,
            }
        )
    return sample_rows


def summarize_niah_sample_metrics(sample_metrics):
    """Summarize sample-level NIAH diagnostics into aggregate scalar metrics.

    中文说明:
    - 调用方 / Called by: `evaluate_niah_depths`, report and SwanLab payload builders
    - 调用对象 / Calls: `_mean`, `_median`
    - 作用 / Purpose: 将样本级 top-k、rank、confidence、margin、entropy 汇总为稳定的
      depth 级和全局评估指标，补足单一 accuracy/loss 不能表达的模型能力
    - 参数 / Parameters: `sample_metrics` 是 `compute_selected_logits_sample_metrics` 返回的非空列表
    - 返回 / Returns: dict，包含 top-k accuracy、rank、概率、margin、entropy、loss 和样本数
    - 内部关键变量 / Internal variables: `rows` 是只读样本行引用
    - 接入 / Integration: `evaluate_niah_depths` 的统一汇总入口；不要在训练循环重复实现汇总
    - 错误处理 / Error handling: 空列表抛出 `ValueError`
    - 副作用 / Side effects: 无
    - 事务边界 / Transaction boundary: 不涉及事务
    - 并发与幂等 / Concurrency and idempotency: 纯函数
    - 关键词 / Keywords:
      aggregate|topk|rank|confidence|margin|entropy|niah|metrics|汇总|能力

    English documentation:
    Function name:
        summarize_niah_sample_metrics
    Purpose:
        Aggregate sample-level NIAH diagnostics into scalar evaluation metrics.
    Called by:
        `evaluate_niah_depths`, report builders, and SwanLab payload builders.
    Calls:
        `_mean` and `_median`.
    Parameters:
        - sample_metrics: non-empty list from `compute_selected_logits_sample_metrics`.
    Returns:
        Dictionary with top-k accuracy, rank, probability, margin, entropy, loss, and sample count.
    Error handling:
        Raises `ValueError` when no sample metrics are provided.
    Side effects:
        None.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Pure aggregation.
    English keywords:
        aggregate, topk, rank, confidence, margin, entropy, niah, metrics, summary, capability
    """
    if not sample_metrics:
        raise ValueError("sample_metrics must not be empty")
    summary = {
        "total_samples": len(sample_metrics),
        "top1_accuracy": _mean([1.0 if row["correct"] else 0.0 for row in sample_metrics]),
        "top3_accuracy": _mean([1.0 if row["top3_correct"] else 0.0 for row in sample_metrics]),
        "top5_accuracy": _mean([1.0 if row["top5_correct"] else 0.0 for row in sample_metrics]),
        "mean_loss": _mean([row["loss"] for row in sample_metrics]),
        "mean_target_rank": _mean([row["target_rank"] for row in sample_metrics]),
        "median_target_rank": _median([row["target_rank"] for row in sample_metrics]),
        "mean_target_prob": _mean([row["target_prob"] for row in sample_metrics]),
        "mean_pred_prob": _mean([row["pred_prob"] for row in sample_metrics]),
        "mean_logit_margin": _mean([row["logit_margin"] for row in sample_metrics]),
        "mean_prob_margin": _mean([row["prob_margin"] for row in sample_metrics]),
        "mean_entropy": _mean([row["entropy"] for row in sample_metrics]),
    }
    if any("readout_available" in row for row in sample_metrics):
        summary["readout_available_rate"] = _mean(
            [1.0 if row.get("readout_available") else 0.0 for row in sample_metrics]
        )
    if any("target_candidate_present" in row for row in sample_metrics):
        summary["target_candidate_hit_rate"] = _mean(
            [1.0 if row.get("target_candidate_present") else 0.0 for row in sample_metrics]
        )
        target_candidate_ranks = [
            row.get("target_candidate_rank")
            for row in sample_metrics
            if row.get("target_candidate_rank") is not None
        ]
        summary["mean_target_candidate_rank"] = (
            None if not target_candidate_ranks else _mean(target_candidate_ranks)
        )
    if any("span_valid_candidate_count" in row for row in sample_metrics):
        summary["mean_span_valid_candidate_count"] = _mean(
            [float(row.get("span_valid_candidate_count") or 0.0) for row in sample_metrics]
        )
        summary["mean_span_raw_candidate_count"] = _mean(
            [float(row.get("span_raw_candidate_count") or 0.0) for row in sample_metrics]
        )
        summary["mean_span_pair_candidate_count"] = _mean(
            [float(row.get("span_pair_candidate_count") or 0.0) for row in sample_metrics]
        )
        summary["span_pair_available_rate"] = _mean(
            [1.0 if row.get("span_pair_available") else 0.0 for row in sample_metrics]
        )
        summary["span_filter_fallback_rate"] = _mean(
            [1.0 if row.get("span_filter_fallback") else 0.0 for row in sample_metrics]
        )
        summary["span_selected_pair_rate"] = _mean(
            [1.0 if row.get("span_selected_is_pair") else 0.0 for row in sample_metrics]
        )
        summary["span_target_pair_candidate_rate"] = _mean(
            [
                1.0 if row.get("span_target_pair_candidate_present") else 0.0
                for row in sample_metrics
            ]
        )
        summary["span_target_value_page_candidate_rate"] = _mean(
            [
                1.0 if row.get("span_target_value_in_page_candidates") else 0.0
                for row in sample_metrics
            ]
        )
        summary["span_target_pair_page_candidate_rate"] = _mean(
            [
                1.0 if row.get("span_target_pair_in_page_candidates") else 0.0
                for row in sample_metrics
            ]
        )
        summary["span_target_value_top_token_rate"] = _mean(
            [
                1.0 if row.get("span_target_value_in_top_tokens") else 0.0
                for row in sample_metrics
            ]
        )
        summary["span_target_pair_top_token_rate"] = _mean(
            [
                1.0 if row.get("span_target_pair_in_top_tokens") else 0.0
                for row in sample_metrics
            ]
        )
        summary["span_target_value_seed_token_rate"] = _mean(
            [
                1.0 if row.get("span_target_value_in_seed_tokens") else 0.0
                for row in sample_metrics
            ]
        )
        summary["span_target_pair_seed_token_rate"] = _mean(
            [
                1.0 if row.get("span_target_pair_in_seed_tokens") else 0.0
                for row in sample_metrics
            ]
        )
    return summary


def _span_candidate_diagnostics_from_eval(eval_result):
    """Extract span candidate diagnostics from an eval result.

    中文说明:
    - 调用方 / Called by: `run_niah_verification_case`。
    - 调用对象 / Calls: 无。
    - 作用 / Purpose: 用统一字段表从 light/robust/re-eval 结果抽取 span predictor
      候选结构诊断，避免三个分支维护不同字段。
    - 参数 / Parameters: `eval_result` 是 `evaluate_niah_depths` 返回值或 None。
    - 返回 / Returns: dict，字段缺失时值为 None。
    - 错误处理 / Error handling: None 或非 dict 输入返回全 None。
    - 副作用 / Side effects: 无。
    """
    if not isinstance(eval_result, dict):
        return {field: None for field in NIAH_SPAN_CANDIDATE_DIAGNOSTIC_FIELDS}
    return {
        field: eval_result.get(field)
        for field in NIAH_SPAN_CANDIDATE_DIAGNOSTIC_FIELDS
    }


def _depth_key(depth):
    """Convert a numeric NIAH depth into a stable metric-key fragment.

    中文说明:
    - 调用方 / Called by: `add_niah_eval_metrics_to_swanlab_payload`
    - 调用对象 / Calls: string formatting and `str.replace`
    - 作用 / Purpose: 将 `0.1` 这类 depth 转换成 SwanLab metric key 安全片段
    - 参数 / Parameters: `depth` 是 NIAH depth ratio
    - 返回 / Returns: string，例如 `0p1`
    - 错误处理 / Error handling: 非数字 depth 会按 Python 格式化规则抛出或转换
    - 副作用 / Side effects: 无
    - 事务边界 / Transaction boundary: 不涉及事务
    - 并发与幂等 / Concurrency and idempotency: 纯函数
    - 关键词 / Keywords:
      depth|metric_key|swanlab|niah|format|safe|fragment|helper|键|深度

    English documentation:
    Function name:
        _depth_key
    Purpose:
        Convert a NIAH depth ratio into a stable SwanLab metric-key fragment.
    Called by:
        `add_niah_eval_metrics_to_swanlab_payload`.
    Calls:
        String formatting and `str.replace`.
    Parameters:
        - depth: NIAH depth ratio.
    Returns:
        Safe string fragment such as `0p1`.
    Error handling:
        Propagates normal formatting errors.
    Side effects:
        None.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Pure and deterministic.
    English keywords:
        depth, metric_key, swanlab, niah, format, safe, fragment, helper, key, diagnostics
    """
    return f"{float(depth):g}".replace(".", "p")


def add_niah_eval_metrics_to_swanlab_payload(payload, prefix, eval_result, include_samples):
    """Add aggregate, per-depth, and optional sample NIAH metrics to one SwanLab payload.

    中文说明:
    - 调用方 / Called by: `run_niah_verification_case`
    - 调用对象 / Calls: `_depth_key`
    - 作用 / Purpose: 保证每个 optimizer step 只调用一次 SwanLab log，同时完整携带
      light eval 或 robust eval 的聚合、分 depth 与可选样本级标量
    - 参数 / Parameters:
      `payload` 是待上传 dict；`prefix` 是 metric 前缀；`eval_result` 是 `evaluate_niah_depths`
      返回值；`include_samples` 控制是否展开 sample-level 明细
    - 返回 / Returns: None，原地更新 payload
    - 内部关键变量 / Internal variables: `scalar_metric_names` 是固定上传的聚合指标白名单
    - 接入 / Integration: 训练循环应复用本函数，避免 SwanLab key 命名分裂
    - 错误处理 / Error handling: 缺少必要 eval 字段时抛出 KeyError，保留真实错误
    - 副作用 / Side effects: 修改传入 payload；不直接访问网络
    - 事务边界 / Transaction boundary: 不涉及 Unit of Work、数据库或文件事务
    - 并发与幂等 / Concurrency and idempotency: 对同一 payload 重复调用会覆盖同名 key
    - 关键词 / Keywords:
      swanlab|payload|sample_metrics|depth_metrics|topk|rank|confidence|niah|上传|指标

    English documentation:
    Function name:
        add_niah_eval_metrics_to_swanlab_payload
    Purpose:
        Add aggregate, per-depth, and optional sample metrics into one SwanLab log payload.
    Called by:
        `run_niah_verification_case`.
    Calls:
        `_depth_key`.
    Parameters:
        - payload: dictionary to mutate.
        - prefix: metric prefix.
        - eval_result: result from `evaluate_niah_depths`.
        - include_samples: whether to flatten sample-level diagnostics.
    Returns:
        None.
    Error handling:
        Propagates missing required metric keys.
    Side effects:
        Mutates `payload`; does not upload by itself.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Repeated calls overwrite matching keys.
    English keywords:
        swanlab, payload, sample_metrics, depth_metrics, topk, rank, confidence, niah, upload, metrics
    """
    scalar_metric_names = (
        "mean_accuracy",
        "min_depth_accuracy",
        "mean_loss",
        "top1_accuracy",
        "top3_accuracy",
        "top5_accuracy",
        "mean_target_rank",
        "median_target_rank",
        "mean_target_prob",
        "mean_pred_prob",
        "mean_logit_margin",
        "mean_prob_margin",
        "mean_entropy",
        "readout_available_rate",
        "target_candidate_hit_rate",
        "mean_target_candidate_rank",
        "total_samples",
    )
    for metric_name in scalar_metric_names:
        payload[f"{prefix}/{metric_name}"] = eval_result[metric_name]
    for depth_row in eval_result["depth_rows"]:
        depth_prefix = f"{prefix}/depth_{_depth_key(depth_row['depth'])}"
        for metric_name in scalar_metric_names:
            if metric_name in depth_row:
                payload[f"{depth_prefix}/{metric_name}"] = depth_row[metric_name]
    if include_samples:
        sample_metric_names = (
            "depth",
            "query_position_ratio",
            "target_token",
            "pred_token",
            "correct",
            "top3_correct",
            "top5_correct",
            "target_rank",
            "target_prob",
            "pred_prob",
            "loss",
            "logit_margin",
            "prob_margin",
            "entropy",
        )
        for sample_row in eval_result["sample_metrics"]:
            sample_prefix = f"{prefix}/sample_{sample_row['sample_index']}"
            for metric_name in sample_metric_names:
                value = sample_row[metric_name]
                payload[f"{sample_prefix}/{metric_name}"] = float(value) if isinstance(value, bool) else value


def evaluate_niah_depths(
    model,
    seq_len,
    device,
    vocab_size,
    batch_size,
    criterion,
    depths=NIAH_DEPTHS,
    batches_per_depth=DEFAULT_NIAH_EVAL_BATCHES_PER_DEPTH,
    niah_readout_mode=DEFAULT_NIAH_READOUT_MODE,
    niah_span_candidate_filter=DEFAULT_NIAH_SPAN_CANDIDATE_FILTER,
):
    """Evaluate NIAH retrieval on independent batches across all configured depths.

    中文说明:
    - 调用方 / Called by: `run_single_niah_test`, `run_niah_verification_case`
    - 调用对象 / Calls: `generate_haystack_with_needle`,
      `extract_query_positions_and_targets`, `forward_selected_logits`,
      `compute_selected_logits_sample_metrics`, `summarize_niah_sample_metrics`
    - 作用 / Purpose: 将训练 batch accuracy 与真正检索评估解耦；对每个 depth 生成独立
      eval batch，并返回 mean/min depth accuracy、top-k、rank、confidence、margin、entropy
      和样本级诊断，防止单个幸运训练样本高估模型能力
    - 参数 / Parameters:
      `model` 是 NIAH 模型；`seq_len/vocab_size/batch_size` 控制评估样本规模；
      `criterion` 是 loss 函数；`depths` 是要覆盖的 depth；`batches_per_depth` 是每个 depth
      的独立 batch 数，默认值高于 3，降低小 batch 长上下文评估中的二项抽样噪声
    - 返回 / Returns: dict，包含 mean accuracy、min depth accuracy、mean loss、top-k/rank/confidence
      汇总、depth rows 和 sample_metrics
    - 内部关键变量 / Internal variables:
      `depth_rows` 记录每个 depth 的聚合指标；`sample_metrics` 记录不含原始 token 的样本级标量；
      `was_training` 保存模型原状态
    - 接入 / Integration: 训练循环只用本函数结果决定成功/报告口径，不再用训练 batch 命中率
    - 错误处理 / Error handling: 空 depth 或非法 batch 数抛出 `ValueError`
    - 副作用 / Side effects: 消耗 PyTorch RNG 生成 eval 样本；临时切换 eval/train 模式
    - 事务边界 / Transaction boundary: 不涉及 Unit of Work、数据库或文件事务
    - 并发与幂等 / Concurrency and idempotency: 固定 seed 和调用顺序后可复现；不适合并发共享模型
    - 关键词 / Keywords:
      evaluation|independent_batch|depth|mean_accuracy|min_depth|niah|retrieval|report|泛化|评估

    English documentation:
    Function name:
        evaluate_niah_depths
    Purpose:
        Evaluate retrieval on independent batches for every configured NIAH depth.
    Called by:
        `run_single_niah_test` and `run_niah_verification_case`.
    Calls:
        NIAH sample generation, target extraction, selected-logit forward, and metric helpers.
    Parameters:
        - model: NIAH model.
        - seq_len/vocab_size/batch_size: evaluation sample shape controls.
        - criterion: loss function.
        - depths: non-empty depth ratios.
        - batches_per_depth: positive number of eval batches per depth; defaults to a
          larger sample count than 3 to reduce binomial sampling noise.
    Returns:
        Dictionary with aggregate, per-depth, and sample-level evaluation metrics.
    Error handling:
        Raises `ValueError` for empty depth lists or invalid batch counts.
    Side effects:
        Advances PyTorch RNG and temporarily toggles model eval/train mode.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Reproducible with fixed seed and call order; do not share one model concurrently.
    English keywords:
        evaluation, independent_batch, depth, mean_accuracy, min_depth, niah, retrieval, report, generalization, metrics
    """
    depth_values = tuple(depths)
    if not depth_values:
        raise ValueError("depths must not be empty")
    if batches_per_depth <= 0:
        raise ValueError("batches_per_depth must be positive")

    niah_readout_mode = normalize_niah_readout_mode(niah_readout_mode)
    niah_span_candidate_filter = normalize_niah_span_candidate_filter(
        niah_span_candidate_filter
    )
    was_training = model.training
    depth_rows = []
    sample_metrics = []
    sample_index = 0

    model.eval()
    try:
        with torch.no_grad():
            for depth in depth_values:
                depth_sample_metrics = []
                for _ in range(batches_per_depth):
                    X, Y, _ = generate_haystack_with_needle(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        vocab_size=vocab_size,
                        needle_depth_ratio=depth,
                    )
                    query_positions, targets = extract_query_positions_and_targets(X, Y, device)
                    if niah_readout_mode in {"needle_copy", "needle_pair_copy"}:
                        logits_target, aux = model.forward_selected_logits(
                            X,
                            query_positions,
                            return_aux=True,
                        )
                        batch_sample_metrics = compute_niah_needle_copy_readout_sample_metrics(
                            X=X,
                            targets=targets,
                            query_positions=query_positions,
                            aux=aux,
                            seq_len=seq_len,
                            depth=depth,
                            sample_index_start=sample_index,
                            readout_mode=niah_readout_mode,
                        )
                        del aux
                    elif niah_readout_mode == "span_predictor":
                        logits_target, hidden_query, aux = model.forward_selected_logits(
                            X,
                            query_positions,
                            return_hidden=True,
                            return_aux=True,
                        )
                        batch_sample_metrics = compute_niah_span_predictor_sample_metrics(
                            model=model,
                            hidden_query=hidden_query,
                            X=X,
                            targets=targets,
                            query_positions=query_positions,
                            aux=aux,
                            seq_len=seq_len,
                            depth=depth,
                            sample_index_start=sample_index,
                            candidate_filter=niah_span_candidate_filter,
                        )
                        del hidden_query, aux
                    else:
                        logits_target = model.forward_selected_logits(X, query_positions)
                        batch_sample_metrics = compute_selected_logits_sample_metrics(
                            logits_target=logits_target,
                            targets=targets,
                            query_positions=query_positions,
                            seq_len=seq_len,
                            depth=depth,
                            sample_index_start=sample_index,
                        )
                    sample_index += len(batch_sample_metrics)
                    depth_sample_metrics.extend(batch_sample_metrics)
                    sample_metrics.extend(batch_sample_metrics)
                    del X, Y, query_positions, logits_target, targets

                depth_summary = summarize_niah_sample_metrics(depth_sample_metrics)
                depth_rows.append(
                    {
                        "depth": depth,
                        "accuracy": depth_summary["top1_accuracy"],
                        "loss": depth_summary["mean_loss"],
                        "samples": depth_summary["total_samples"],
                        "top1_accuracy": depth_summary["top1_accuracy"],
                        "top3_accuracy": depth_summary["top3_accuracy"],
                        "top5_accuracy": depth_summary["top5_accuracy"],
                        "mean_target_rank": depth_summary["mean_target_rank"],
                        "median_target_rank": depth_summary["median_target_rank"],
                        "mean_target_prob": depth_summary["mean_target_prob"],
                        "mean_pred_prob": depth_summary["mean_pred_prob"],
                        "mean_logit_margin": depth_summary["mean_logit_margin"],
                        "mean_prob_margin": depth_summary["mean_prob_margin"],
                        "mean_entropy": depth_summary["mean_entropy"],
                        "readout_available_rate": depth_summary.get(
                            "readout_available_rate"
                        ),
                        "target_candidate_hit_rate": depth_summary.get(
                            "target_candidate_hit_rate"
                        ),
                        "mean_target_candidate_rank": depth_summary.get(
                            "mean_target_candidate_rank"
                        ),
                        "mean_span_valid_candidate_count": depth_summary.get(
                            "mean_span_valid_candidate_count"
                        ),
                        "mean_span_raw_candidate_count": depth_summary.get(
                            "mean_span_raw_candidate_count"
                        ),
                        "mean_span_pair_candidate_count": depth_summary.get(
                            "mean_span_pair_candidate_count"
                        ),
                        "span_pair_available_rate": depth_summary.get(
                            "span_pair_available_rate"
                        ),
                        "span_filter_fallback_rate": depth_summary.get(
                            "span_filter_fallback_rate"
                        ),
                        "span_selected_pair_rate": depth_summary.get(
                            "span_selected_pair_rate"
                        ),
                        "span_target_pair_candidate_rate": depth_summary.get(
                            "span_target_pair_candidate_rate"
                        ),
                        "span_target_value_page_candidate_rate": depth_summary.get(
                            "span_target_value_page_candidate_rate"
                        ),
                        "span_target_pair_page_candidate_rate": depth_summary.get(
                            "span_target_pair_page_candidate_rate"
                        ),
                        "span_target_value_top_token_rate": depth_summary.get(
                            "span_target_value_top_token_rate"
                        ),
                        "span_target_pair_top_token_rate": depth_summary.get(
                            "span_target_pair_top_token_rate"
                        ),
                        "span_target_value_seed_token_rate": depth_summary.get(
                            "span_target_value_seed_token_rate"
                        ),
                        "span_target_pair_seed_token_rate": depth_summary.get(
                            "span_target_pair_seed_token_rate"
                        ),
                    }
                )
    finally:
        if was_training:
            model.train()

    overall_summary = summarize_niah_sample_metrics(sample_metrics)
    return {
        "mean_accuracy": overall_summary["top1_accuracy"],
        "min_depth_accuracy": min(row["accuracy"] for row in depth_rows),
        "mean_loss": overall_summary["mean_loss"],
        "top1_accuracy": overall_summary["top1_accuracy"],
        "top3_accuracy": overall_summary["top3_accuracy"],
        "top5_accuracy": overall_summary["top5_accuracy"],
        "mean_target_rank": overall_summary["mean_target_rank"],
        "median_target_rank": overall_summary["median_target_rank"],
        "mean_target_prob": overall_summary["mean_target_prob"],
        "mean_pred_prob": overall_summary["mean_pred_prob"],
        "mean_logit_margin": overall_summary["mean_logit_margin"],
        "mean_prob_margin": overall_summary["mean_prob_margin"],
        "mean_entropy": overall_summary["mean_entropy"],
        "readout_mode": niah_readout_mode,
        "readout_available_rate": overall_summary.get("readout_available_rate"),
        "target_candidate_hit_rate": overall_summary.get("target_candidate_hit_rate"),
        "mean_target_candidate_rank": overall_summary.get("mean_target_candidate_rank"),
        "mean_span_valid_candidate_count": overall_summary.get(
            "mean_span_valid_candidate_count"
        ),
        "mean_span_raw_candidate_count": overall_summary.get(
            "mean_span_raw_candidate_count"
        ),
        "mean_span_pair_candidate_count": overall_summary.get(
            "mean_span_pair_candidate_count"
        ),
        "span_pair_available_rate": overall_summary.get("span_pair_available_rate"),
        "span_filter_fallback_rate": overall_summary.get("span_filter_fallback_rate"),
        "span_selected_pair_rate": overall_summary.get("span_selected_pair_rate"),
        "span_target_pair_candidate_rate": overall_summary.get(
            "span_target_pair_candidate_rate"
        ),
        "span_target_value_page_candidate_rate": overall_summary.get(
            "span_target_value_page_candidate_rate"
        ),
        "span_target_pair_page_candidate_rate": overall_summary.get(
            "span_target_pair_page_candidate_rate"
        ),
        "span_target_value_top_token_rate": overall_summary.get(
            "span_target_value_top_token_rate"
        ),
        "span_target_pair_top_token_rate": overall_summary.get(
            "span_target_pair_top_token_rate"
        ),
        "span_target_value_seed_token_rate": overall_summary.get(
            "span_target_value_seed_token_rate"
        ),
        "span_target_pair_seed_token_rate": overall_summary.get(
            "span_target_pair_seed_token_rate"
        ),
        "depth_rows": depth_rows,
        "sample_metrics": sample_metrics,
        "total_samples": overall_summary["total_samples"],
    }


def summarize_retrieval_projection_step_metrics(step_rows):
    """Summarize train-time retrieval projection contrastive diagnostics.

    中文说明:
    - 调用方 / Called by: `run_niah_verification_case`
    - 调用对象 / Calls: local numeric helpers only
    - 作用 / Purpose: 聚合每个 optimizer step 的 projection contrastive 诊断，
      帮助判断 q/k 表征监督是否真的接通并改善 evidence 排名。
    - 参数 / Parameters: `step_rows` 是训练循环每步写入报告的行。
    - 返回 / Returns: dict，包含 available step 数、hit/top1/rank/margin/loss 均值。
    - 错误处理 / Error handling: 缺字段或 None 会被跳过。
    - 副作用 / Side effects: 无。
    - 关键词 / Keywords:
      retrieval|projection|contrastive|summary|rank|niah|诊断

    English documentation:
    Function name:
        summarize_retrieval_projection_step_metrics
    Purpose:
        Aggregate opt-in retrieval q/k projection loss diagnostics over training steps.
    """
    rows = list(step_rows or [])
    available_rows = [
        row for row in rows if bool(row.get("retrieval_projection_available"))
    ]

    def _mean_field(field_name):
        values = [
            float(row[field_name])
            for row in available_rows
            if row.get(field_name) is not None
        ]
        return None if not values else sum(values) / len(values)

    positive_steps = sum(
        1
        for row in available_rows
        if float(row.get("retrieval_projection_positive_count") or 0.0) > 0.0
    )
    return {
        "steps": len(rows),
        "available_steps": len(available_rows),
        "positive_steps": positive_steps,
        "mean_loss": _mean_field("retrieval_projection_loss"),
        "mean_hit_rate": _mean_field("retrieval_projection_hit_rate"),
        "mean_evidence_score": _mean_field("retrieval_projection_evidence_score_mean"),
        "mean_best_negative_score": _mean_field(
            "retrieval_projection_best_negative_score_mean"
        ),
        "mean_score_margin": _mean_field("retrieval_projection_score_margin_mean"),
        "mean_target_rank": _mean_field("retrieval_projection_target_rank_mean"),
        "mean_top1_rate": _mean_field("retrieval_projection_top1_rate"),
    }


def run_single_niah_test(
    seq_len,
    device,
    vocab_size=100,
    dim=64,
    num_layers=2,
    K=64,
    kr=8,
    model_type="mhdsra2",
    mhdsra2_config_override=None,
    eval_batches_per_depth=DEFAULT_NIAH_EVAL_BATCHES_PER_DEPTH,
    return_metrics=False,
    swanlab_run: SwanLabRunProxy | None = None,
    use_retrieval: bool = False,
):
    """Train one NIAH context length using selected-query logits to bound GPU memory.

    中文说明:
    - 调用方 / Called by: `run_niah_test`, ad-hoc 2M verification commands
    - 调用对象 / Calls: `get_niah_runtime_config`, `build_niah_model`,
      `generate_haystack_with_needle`, `extract_query_positions_and_targets`,
      `evaluate_niah_depths`, `MultiLayerMHDSRA2Model.forward_selected_logits`, Adam optimizer
    - 作用 / Purpose: 对指定上下文长度执行 NIAH 训练验证；训练 depth 使用确定性轮询，
      成功/默认返回口径来自最后一次独立 eval batch 的所有 depth 覆盖；`return_metrics=True`
      时同时返回 final/best/train 口径，避免将历史最高点误读为最终稳定能力；2M 场景只保留
      查询位置 logits，不再生成全序列 logits，目标是避免训练 OOM
    - 参数 / Parameters: `seq_len/device/vocab_size/dim/num_layers/K/kr/model_type`
      控制实验规模；`mhdsra2_config_override` 可覆盖 MHDSRA2 配置；`eval_batches_per_depth`
      控制每个 depth 的 eval 样本数；`return_metrics` 控制是否返回结构化指标
    - 返回 / Returns: 默认返回最后一次独立 depth 评估 mean accuracy，范围 `[0.0, 1.0]`；
      `return_metrics=True` 时返回 dict
    - 错误处理 / Error handling: OOM 由上游 sweep 捕获；其他异常向上抛出
    - 副作用 / Side effects: 打印训练进度；更新模型参数；不写文件
    - 事务边界 / Transaction boundary: 不涉及 Unit of Work 或持久化事务
    - 并发与幂等 / Concurrency and idempotency: 依赖随机 depth 与 PyTorch RNG；固定 seed 后可复现
    - 关键词 / Keywords:
      niah|train|2m|selected_logits|oom|gpu_memory|mhdsra2|accuracy|needle|训练

    English documentation:
    Function name:
        run_single_niah_test
    Purpose:
        Train a single NIAH context length with memory-bounded selected query logits.
    Called by:
        `run_niah_test` and direct verification commands.
    Calls:
        NIAH data generation, target extraction, independent depth evaluation,
        selected-logit model forward, and Adam.
    Parameters:
        Model scale, MHDSRA2 override, evaluation sample count, and return-shape arguments.
    Returns:
        Final independent depth-evaluation mean accuracy by default, or a metrics dictionary
        containing final and best accuracy when `return_metrics=True`.
    Error handling:
        Propagates non-OOM errors; sweep callers handle OOM.
    Side effects:
        Prints progress and mutates model parameters only.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Reproducible under fixed RNG seeds.
    English keywords:
        niah, train, 2m, selected_logits, oom, gpu_memory, mhdsra2, accuracy, needle, training
    Note:
        Evaluation sample count is eval_batches_per_depth * batch_size * len(depths),
        defaulting to ~96 samples. Binomial standard error is approximately ±4%;
        consider this statistical fluctuation when comparing across runs.
    """
    runtime_cfg = get_niah_runtime_config(seq_len)
    batch_size = runtime_cfg["batch_size"]
    epochs = runtime_cfg["epochs"]
    chunk_size = runtime_cfg["chunk_size"]

    print(
        f"\n--- Running Needle-In-A-Haystack Test ({seq_len} tokens) on {device} "
        f"| model_type={model_type} ---"
    )
    print(
        f"Config | batch_size={batch_size} | epochs={epochs} | chunk_size={chunk_size}"
    )

    model = build_niah_model(
        device=device,
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        K=K,
        kr=kr,
        chunk_size=chunk_size,
        model_type=model_type,
        mhdsra2_config_override=mhdsra2_config_override,
        use_retrieval=use_retrieval,
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    depths_to_test = NIAH_DEPTHS
    model.train()
    best_train_acc = 0.0
    best_eval_mean_acc = 0.0
    final_eval_mean_acc = 0.0
    final_eval_min_depth_acc = 0.0

    for epoch in range(epochs):
        current_depth = get_niah_depth_for_epoch(epoch, depths_to_test)
        X, Y, _ = generate_haystack_with_needle(batch_size, seq_len, vocab_size, current_depth)
        query_positions, targets = extract_query_positions_and_targets(X, Y, device)

        optimizer.zero_grad()
        logits_target = model.forward_selected_logits(X, query_positions)
        loss = criterion(logits_target, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Momentum-QKV: update slow QKV for all layers
        if hasattr(model, 'layers'):
            for layer in model.layers:
                if hasattr(layer, 'update_momentum'):
                    layer.update_momentum()

        preds = logits_target.argmax(dim=-1)
        correct = (preds == targets).sum().item()
        step_train_acc = correct / batch_size
        best_train_acc = max(best_train_acc, step_train_acc)
        loss_value = loss.item()
        should_stop = False
        should_evaluate = epoch % NIAH_EVAL_INTERVAL == 0 or epoch == epochs - 1
        if should_evaluate:
            eval_result = evaluate_niah_depths(
                model=model,
                seq_len=seq_len,
                device=device,
                vocab_size=vocab_size,
                batch_size=batch_size,
                criterion=criterion,
                depths=depths_to_test,
                batches_per_depth=eval_batches_per_depth,
            )
            final_eval_mean_acc = eval_result["mean_accuracy"]
            final_eval_min_depth_acc = eval_result["min_depth_accuracy"]
            best_eval_mean_acc = max(best_eval_mean_acc, final_eval_mean_acc)
            print(
                f"Epoch {epoch:3d} | Train Depth: {current_depth:.1f} | "
                f"Train Loss: {loss_value:.4f} | Step Train Accuracy: {step_train_acc*100:5.1f}% | "
                f"Eval Mean Accuracy: {final_eval_mean_acc*100:5.1f}% | "
                f"Eval Min-Depth Accuracy: {final_eval_min_depth_acc*100:5.1f}%"
            )
            if swanlab_run is not None and swanlab_run.enabled:
                swanlab_run.log({
                    "train/loss": loss_value,
                    "train/accuracy": step_train_acc,
                    "train/depth": current_depth,
                    "eval/mean_accuracy": final_eval_mean_acc,
                    "eval/min_depth_accuracy": final_eval_min_depth_acc,
                }, step=epoch)

            should_stop = (
                final_eval_min_depth_acc >= 1.0
                and eval_result["mean_loss"] < 0.1
                and eval_result["total_samples"] >= NIAH_MIN_EVAL_SAMPLES_FOR_EARLY_STOP
            )
        if should_stop:
            print(f"\nSUCCESS! The model successfully found the needle in {seq_len} context!")

        del X, Y, query_positions, logits_target, targets, loss
        if should_stop:
            break

    print(
        f"\nFinal Eval Mean Accuracy @ {seq_len}: {final_eval_mean_acc*100:.1f}% | "
        f"Final Eval Min-Depth Accuracy: {final_eval_min_depth_acc*100:.1f}% | "
        f"Best Eval Mean Accuracy: {best_eval_mean_acc*100:.1f}% | "
        f"Best Step Train Accuracy: {best_train_acc*100:.1f}%"
    )
    metrics = {
        "status": "ok",
        "final_eval_mean_accuracy": final_eval_mean_acc,
        "final_eval_min_depth_accuracy": final_eval_min_depth_acc,
        "best_eval_mean_accuracy": best_eval_mean_acc,
        "best_step_train_accuracy": best_train_acc,
        "eval_batches_per_depth": eval_batches_per_depth,
        "seq_len": seq_len,
        "model_type": model_type,
    }
    del model, optimizer, criterion
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics if return_metrics else final_eval_mean_acc


def run_single_niah_capacity_test(
    seq_len,
    device,
    mode="train_step",
    vocab_size=100,
    dim=64,
    num_layers=2,
    K=64,
    kr=8,
    chunk_size=256,
    batch_size=1,
    batches_per_depth=DEFAULT_NIAH_CAPACITY_BATCHES_PER_DEPTH,
):
    """Run one NIAH capacity probe with bounded selected-query logits.

    中文说明:
    - 调用方 / Called by: `run_niah_capacity_test`, manual 2M memory verification
    - 调用对象 / Calls: `build_niah_model`, `generate_haystack_with_needle`,
      `extract_query_positions_and_targets`, `forward_selected_logits`, CUDA memory counters
    - 作用 / Purpose: 验证指定上下文长度在 forward-only 或 train-step 模式下是否 OOM，
      并覆盖全部标准 depth 记录 peak memory、mean accuracy 与 min-depth accuracy；每个 depth
      默认评估多个 batch，避免单 batch 命中率被误读为稳定能力
    - 参数 / Parameters:
      `mode` 支持 `forward_only/train_step`；其余参数控制模型、batch、chunk、每 depth batch 数和序列长度
    - 返回 / Returns: dict，包含 `status/accuracy/min_depth_accuracy/peak_mem_mb/depth_results`
    - 错误处理 / Error handling: 非法 mode 抛出 `ValueError`；OOM 由上游容量 sweep 捕获
    - 副作用 / Side effects: train_step 会按 `batches_per_depth * len(NIAH_DEPTHS)` 更新模型参数；
      CUDA 场景重置 peak memory 计数器
    - 事务边界 / Transaction boundary: 不涉及 Unit of Work 或持久化事务
    - 并发与幂等 / Concurrency and idempotency: 依赖随机样本和模型初始化；固定 seed 后可复现
    - 关键词 / Keywords:
      capacity|train_step|forward_only|2m|peak_memory|selected_logits|niah|oom|cuda|容量

    English documentation:
    Function name:
        run_single_niah_capacity_test
    Purpose:
        Probe NIAH forward or one training step across all standard depths while
        avoiding full-sequence logits.
    Called by:
        `run_niah_capacity_test` and manual memory checks.
    Calls:
        Model factory, vectorized data generation, selected-logit forward, CUDA counters.
    Parameters:
        Context length, mode, model scale, chunk size, batch size, and eval batches per depth.
    Returns:
        Result dictionary with status, mean/min-depth accuracy, per-depth rows, and peak memory in MB.
    Error handling:
        Raises `ValueError` for unsupported modes; callers convert OOM to result status.
    Side effects:
        Optimizer steps in train mode according to the requested sample count; resets CUDA peak memory stats.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Reproducible under fixed seeds.
    English keywords:
        capacity, train_step, forward_only, 2m, peak_memory, selected_logits, niah, oom, cuda, probe
    """
    if mode not in {"forward_only", "train_step"}:
        raise ValueError(f"Unsupported NIAH capacity mode: {mode}")
    if batches_per_depth <= 0:
        raise ValueError("batches_per_depth must be positive")

    model = build_niah_model(
        device=device,
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        K=K,
        kr=kr,
        chunk_size=chunk_size,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3) if mode == "train_step" else None
    depth_results = []
    total_correct = 0
    total_samples = 0
    overall_peak_mem_mb = 0.0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model.eval()
    for depth in NIAH_DEPTHS:
        depth_correct = 0
        depth_samples = 0
        depth_losses = []
        for _ in range(batches_per_depth):
            X, Y, _ = generate_haystack_with_needle(batch_size, seq_len, vocab_size, depth)
            query_positions, targets = extract_query_positions_and_targets(X, Y, device)
            with torch.no_grad():
                logits_target = model.forward_selected_logits(X, query_positions)
                loss = criterion(logits_target, targets)
            preds = logits_target.argmax(dim=-1)
            correct = int((preds == targets).sum().item())
            sample_count = int(targets.numel())
            depth_correct += correct
            depth_samples += sample_count
            depth_losses.append(float(loss.item()))
            del X, Y, query_positions, logits_target, targets, loss

        depth_accuracy = depth_correct / depth_samples
        depth_peak_mem_mb = 0.0
        if torch.cuda.is_available():
            depth_peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            overall_peak_mem_mb = max(overall_peak_mem_mb, depth_peak_mem_mb)
        depth_results.append(
            {
                "depth": depth,
                "accuracy": depth_accuracy,
                "loss": sum(depth_losses) / len(depth_losses),
                "samples": depth_samples,
                "peak_mem_mb": depth_peak_mem_mb,
            }
        )
        total_correct += depth_correct
        total_samples += depth_samples

    if mode == "train_step":
        model.train()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        train_depth = NIAH_DEPTHS[len(NIAH_DEPTHS) // 2]
        X, Y, _ = generate_haystack_with_needle(batch_size, seq_len, vocab_size, train_depth)
        query_positions, targets = extract_query_positions_and_targets(X, Y, device)
        optimizer.zero_grad()
        logits_target = model.forward_selected_logits(X, query_positions)
        loss = criterion(logits_target, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        del X, Y, query_positions, logits_target, targets, loss

    acc = total_correct / total_samples
    min_depth_accuracy = min(row["accuracy"] for row in depth_results)
    peak_mem_mb = overall_peak_mem_mb
    if mode == "train_step" and torch.cuda.is_available():
        train_step_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_mem_mb = max(overall_peak_mem_mb, train_step_peak)

    del model
    if optimizer is not None:
        del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "status": "ok",
        "accuracy": acc,
        "min_depth_accuracy": min_depth_accuracy,
        "depth_results": depth_results,
        "total_samples": total_samples,
        "batches_per_depth": batches_per_depth,
        "peak_mem_mb": peak_mem_mb,
    }


# Short sequences where detach_state=False (full BPTT) is affordable on 8GB GPU.
NIAH_FULL_BPTT_MAX_SEQ_LEN = 16384


def _expand_vocab_state_dict(ckpt_state_dict, old_vocab, new_vocab, dim):
    """Expand embedding and out_proj weights from old_vocab to new_vocab.

    Used by vocab-size curriculum learning: when loading a checkpoint trained
    with a smaller vocabulary, the embedding and out_proj layers need to be
    padded with random-initialized rows/columns for the new token ids.
    """
    new_state = {}
    for key, value in ckpt_state_dict.items():
        if key == "embedding.weight":
            new = torch.empty(new_vocab, dim)
            nn.init.normal_(new, std=dim**-0.5)
            new[:old_vocab] = value.to(new.device, new.dtype)
            new_state[key] = new
        elif key == "out_proj.weight":
            new = torch.empty(new_vocab, dim)
            nn.init.normal_(new, std=dim**-0.5)
            new[:old_vocab] = value.to(new.device, new.dtype)
            new_state[key] = new
        elif key == "out_proj.bias":
            new = torch.zeros(new_vocab)
            new[:old_vocab] = value.to(new.device, new.dtype)
            new_state[key] = new
        else:
            new_state[key] = value
    return new_state


def run_niah_verification_case(
    seq_len,
    device,
    *,
    vocab_size=100,
    data_vocab_size: int | None = None,
    dim=64,
    num_layers=2,
    K=64,
    kr=8,
    chunk_size=1024,
    batch_size=1,
    epochs=60,
    learning_rate=1e-3,
    seed=20260506,
    target_accuracy=1.0,
    stop_loss=0.1,
    log_interval=20,
    eval_batches_per_depth=DEFAULT_NIAH_LIGHT_EVAL_BATCHES_PER_DEPTH,
    robust_eval_interval=None,
    robust_eval_batches_per_depth=DEFAULT_NIAH_ROBUST_EVAL_BATCHES_PER_DEPTH,
    niah_test_batches_per_depth: int = DEFAULT_NIAH_TEST_BATCHES_PER_DEPTH,
    cudnn_benchmark=False,
    model_type="mhdsra2",
    mhdsra2_config_override=None,
    swanlab_mode: str = "disabled",
    load_checkpoint: str | None = None,
    save_checkpoint: str | None = None,
    needle_loss_alpha: float = 0.5,
    hidden_mse_alpha: float = 0.0,
    retrieval_evidence_loss_alpha: float = 0.0,
    retrieval_evidence_rank_margin: float = 0.0,
    retrieval_evidence_score_margin: float = 0.0,
    retrieval_evidence_target_offset: int = 1,
    query_evidence_alignment_alpha: float = 0.0,
    retrieval_projection_contrastive_alpha: float = 0.0,
    retrieval_projection_temperature: float = (
        DEFAULT_RETRIEVAL_PROJECTION_CONTRASTIVE_TEMPERATURE
    ),
    retrieval_span_predictor_alpha: float = 0.0,
    retrieval_span_structure_features: bool = False,
    use_retrieval: bool = False,
    niah_readout_mode: str = DEFAULT_NIAH_READOUT_MODE,
    niah_span_candidate_filter: str = DEFAULT_NIAH_SPAN_CANDIDATE_FILTER,
    niah_span_loss_mode: str = DEFAULT_NIAH_SPAN_LOSS_MODE,
):
    """Run a reproducible NIAH verification case and return report-ready metrics.

    中文说明:
    - 调用方 / Called by: CLI `verify-2m`, CLI `benchmark-scale`, tests/manual reports
    - 调用对象 / Calls: `seed_all`, `get_niah_depth_for_epoch`, `build_niah_model`,
      `generate_haystack_with_needle`, `extract_query_positions_and_targets`,
      `evaluate_niah_depths`, `forward_selected_logits`, Adam optimizer, CUDA peak memory counters
    - 作用 / Purpose: 将 2M NIAH 和更大参数规模 benchmark 固化为可复现、可写入 reports/
      的统一执行函数；训练 depth 按 optimizer step 轮询，每步 light eval 用于 SwanLab 诊断，
      periodic robust eval 用于正式报告和成功条件
    - 参数 / Parameters:
      `seq_len` 是上下文长度；`dim/num_layers/K/kr/chunk_size` 控制模型规模；
      `epochs/learning_rate/seed` 控制训练过程；`target_accuracy/stop_loss` 控制提前停止；
      `eval_batches_per_depth` 控制每 step light eval 样本数；`robust_eval_interval` 与
      `robust_eval_batches_per_depth` 控制正式 robust eval；`niah_test_batches_per_depth`
      默认 0，显式大于 0 时只在训练和 best-state 选择结束后运行 held-out test eval，
      不参与训练、best 选择或 final validation 口径；`cudnn_benchmark` 控制 cuDNN 自动调优；
      `retrieval_evidence_target_offset` 控制 evidence loss 监督 needle key(offset=0)
      还是 needle value(offset=1)，默认 1 保持旧行为；`query_evidence_alignment_alpha`
      默认 0，仅显式开启时把 query hidden 拉近 evidence token embedding；
      `retrieval_projection_contrastive_alpha` 默认 0，仅显式开启时直接监督 selected retrieval
      query/key projection 的 evidence 排名；`retrieval_span_predictor_alpha` 默认 0，仅显式
      开启时训练一个候选 span 打分头；`retrieval_span_structure_features` 默认 False，
      只让显式结构组把 pair/source 提示送入 span predictor；`niah_span_candidate_filter` 默认 `all`
      保持旧候选池，显式 `key_value_pair` 时只保留成对召回的 key/value 候选；
      `niah_span_loss_mode` 默认 `single_positive` 保持旧 loss，显式 `multi_positive`
      时不再把其它正确候选当成负例。
      `use_retrieval` 显式打开外部分页召回。
    - 返回 / Returns: dict，包含模型配置、训练日志、最终/最佳 eval 准确率、loss、耗时和显存指标；
      `passed_success_criteria` 只在同一 eval 点同时满足 min-depth accuracy 与 loss 阈值时为 true
    - 内部关键变量 / Internal variables:
      `step_rows` 记录报告用训练/light eval 曲线；`robust_eval_rows` 记录正式 eval 曲线；
      `best_accuracy` 跟踪当前正式口径下的最佳 eval mean accuracy；
      `final_accuracy` 跟踪最后一个 eval 点的 mean accuracy；`final_min_depth_accuracy`
      跟踪最弱 depth；`peak_allocated_mb/peak_reserved_mb` 记录 CUDA 显存口径
    - 接入 / Integration: 新增长上下文报告应调用本函数，而不是复制训练循环
    - 错误处理 / Error handling: OOM 和底层异常直接向上抛出，由 CLI 或调用方决定是否捕获
    - 副作用 / Side effects: 更新模型参数、打印进度、重置 CUDA peak memory 计数器
    - 事务边界 / Transaction boundary: 不涉及 Unit of Work、数据库或文件事务
    - 并发与幂等 / Concurrency and idempotency: 固定 seed 和环境后可复现；不适合并发共享同一 GPU
    - 关键词 / Keywords:
      niah|2m|benchmark_scale|reports|selected_logits|cuda_memory|accuracy|mhdsra2|cli|验证

    English documentation:
    Function name:
        run_niah_verification_case
    Purpose:
        Run one reproducible NIAH verification case and return report-ready metrics.
    Called by:
        CLI `verify-2m`, CLI `benchmark-scale`, and tests/manual report flows.
    Calls:
        Seeding, deterministic depth scheduling, model factory, vectorized data generation,
        selected-logit forward, independent depth evaluation, Adam optimizer, and CUDA memory counters.
    Parameters:
        Context length, model scale, training controls, light/robust evaluation sample counts,
        cuDNN benchmark flag, and stopping thresholds.
    Returns:
        Dictionary containing config, epoch logs, final/best eval accuracy, loss,
        elapsed time, success criteria, and memory metrics.
    Error handling:
        Propagates OOM and lower-level errors to the caller.
    Side effects:
        Trains the model and resets CUDA peak-memory counters.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Reproducible under fixed seed and runtime; do not share one GPU concurrently.
    English keywords:
        niah, 2m, benchmark_scale, reports, selected_logits, cuda_memory, accuracy, mhdsra2, cli, verification
    Note:
        The warmup step consumes RNG state, so the same seed produces different
        training data sequences compared to run_single_niah_test.
    """
    niah_readout_mode = normalize_niah_readout_mode(niah_readout_mode)
    niah_span_candidate_filter = normalize_niah_span_candidate_filter(
        niah_span_candidate_filter
    )
    niah_span_loss_mode = normalize_niah_span_loss_mode(niah_span_loss_mode)
    retrieval_evidence_target_offset = int(retrieval_evidence_target_offset)
    if retrieval_evidence_target_offset not in (0, 1):
        raise ValueError("retrieval_evidence_target_offset must be 0 (key) or 1 (value)")
    if retrieval_projection_temperature <= 0.0:
        raise ValueError("retrieval_projection_temperature must be positive")
    if retrieval_span_predictor_alpha < 0.0:
        raise ValueError("retrieval_span_predictor_alpha must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if eval_batches_per_depth <= 0:
        raise ValueError("eval_batches_per_depth must be positive")
    resolved_robust_eval_interval = log_interval if robust_eval_interval is None else robust_eval_interval
    if resolved_robust_eval_interval <= 0:
        raise ValueError("robust_eval_interval must be positive")
    if robust_eval_batches_per_depth <= 0:
        raise ValueError("robust_eval_batches_per_depth must be positive")
    niah_test_batches_per_depth = int(niah_test_batches_per_depth)
    if niah_test_batches_per_depth < 0:
        raise ValueError("niah_test_batches_per_depth must be non-negative")

    seed_all(seed, cudnn_benchmark=cudnn_benchmark)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()

    # Auto-select detach_state: False for short seqs (full BPTT, learnable),
    # True for long seqs (memory-safe). User can override via mhdsra2_config_override.
    resolved_config_override = dict(mhdsra2_config_override) if mhdsra2_config_override else {}
    if "detach_state" not in resolved_config_override:
        auto_detach_state = seq_len > NIAH_FULL_BPTT_MAX_SEQ_LEN
        resolved_config_override["detach_state"] = auto_detach_state
        print(
            f"Auto-detach_state={auto_detach_state} for seq_len={seq_len} "
            f"(threshold={NIAH_FULL_BPTT_MAX_SEQ_LEN})"
        )

    resolved_data_vocab = vocab_size if data_vocab_size is None else data_vocab_size
    if data_vocab_size is not None and data_vocab_size != vocab_size:
        print(
            f"Model vocab_size={vocab_size}, data vocab_size={data_vocab_size}"
        )

    swanlab_run = init_swanlab(
        project="MHDSRA2",
        experiment_name=f"niah_d{dim}_l{num_layers}_s{K}_kr{kr}_seq{seq_len}",
        config={
            "vocab_size": vocab_size, "dim": dim, "num_layers": num_layers,
            "slots": K, "read_topk": kr, "chunk_size": chunk_size,
            "batch_size": batch_size, "epochs": epochs, "learning_rate": learning_rate,
            "seed": seed, "seq_len": seq_len,
            "optimizer_steps": epochs,
            "detach_state": resolved_config_override["detach_state"],
            "needle_loss_alpha": needle_loss_alpha,
            "hidden_mse_alpha": hidden_mse_alpha,
            "retrieval_evidence_loss_alpha": retrieval_evidence_loss_alpha,
            "retrieval_evidence_rank_margin": retrieval_evidence_rank_margin,
            "retrieval_evidence_score_margin": retrieval_evidence_score_margin,
            "retrieval_evidence_target_offset": retrieval_evidence_target_offset,
            "query_evidence_alignment_alpha": query_evidence_alignment_alpha,
            "retrieval_projection_contrastive_alpha": retrieval_projection_contrastive_alpha,
            "retrieval_projection_temperature": retrieval_projection_temperature,
            "retrieval_span_predictor_alpha": retrieval_span_predictor_alpha,
            "retrieval_span_structure_features": bool(retrieval_span_structure_features),
            "use_retrieval": use_retrieval,
            "niah_readout_mode": niah_readout_mode,
            "niah_span_candidate_filter": niah_span_candidate_filter,
            "niah_span_loss_mode": niah_span_loss_mode,
            "light_eval_batches_per_depth": eval_batches_per_depth,
            "robust_eval_interval": resolved_robust_eval_interval,
            "robust_eval_batches_per_depth": robust_eval_batches_per_depth,
            "niah_test_batches_per_depth": niah_test_batches_per_depth,
        },
        mode=swanlab_mode,
        description=f"NIAH verification seq_len={seq_len}",
        tags=["niah", "verification"],
    )

    model = build_niah_model(
        device=device,
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        K=K,
        kr=kr,
        chunk_size=chunk_size,
        model_type=model_type,
        mhdsra2_config_override=resolved_config_override,
        use_retrieval=use_retrieval,
        use_retrieval_span_predictor=retrieval_span_predictor_alpha > 0.0
        or niah_readout_mode == "span_predictor",
        retrieval_span_structure_features=bool(retrieval_span_structure_features),
    )

    if load_checkpoint is not None:
        resolved_load_checkpoint = resolve_niah_checkpoint_path(load_checkpoint)
        checkpoint = torch.load(resolved_load_checkpoint, map_location=device, weights_only=True)
        ckpt_state = checkpoint["model_state_dict"]
        ckpt_vocab = checkpoint.get("vocab_size", vocab_size)
        if ckpt_vocab != vocab_size:
            print(
                f"Expanding checkpoint vocab {ckpt_vocab} -> {vocab_size}"
            )
            ckpt_state = _expand_vocab_state_dict(
                ckpt_state, ckpt_vocab, vocab_size, dim
            )
        missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
        if missing:
            print(f"Checkpoint load: missing keys: {missing}")
        if unexpected:
            print(f"Checkpoint load: unexpected keys: {unexpected}")
        if not missing and not unexpected:
            print(
                f"Loaded checkpoint from {resolved_load_checkpoint} "
                f"(source step {checkpoint.get('step', '?')})"
            )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    depths_to_test = NIAH_DEPTHS
    step_rows = []
    robust_eval_rows = []
    sample_metrics = []
    best_accuracy = 0.0
    best_min_depth_accuracy = 0.0
    best_accuracy_step = None
    best_accuracy_loss = None
    best_eval_source = "light"
    best_state_dict = None
    final_accuracy = 0.0
    final_min_depth_accuracy = 0.0
    final_eval_loss = None
    final_eval_source = "light"
    final_light_eval = None
    final_robust_eval = None
    best_step_train_accuracy = 0.0
    final_step_train_accuracy = 0.0
    final_train_loss = None
    final_loss = None
    final_retrieval_evidence_metrics = {
        "available": False,
        "unavailable_reason": None,
        "hit_rate": None,
        "gate_mean": None,
        "evidence_weight_mean": None,
        "best_negative_weight_mean": None,
        "evidence_margin_mean": None,
        "target_rank_mean": None,
        "top1_rate": None,
        "ranking_loss": None,
        "margin_loss": None,
        "score_margin_loss": None,
        "evidence_score_mean": None,
        "best_negative_score_mean": None,
        "evidence_score_margin_mean": None,
        "score_target_rank_mean": None,
        "score_top1_rate": None,
        "gate_loss": None,
        "positive_count": 0,
    }
    final_retrieval_projection_metrics = {
        "available": False,
        "unavailable_reason": None,
        "hit_rate": None,
        "positive_count": 0,
        "loss": None,
        "evidence_score_mean": None,
        "best_negative_score_mean": None,
        "score_margin_mean": None,
        "target_rank_mean": None,
        "top1_rate": None,
        "temperature": float(retrieval_projection_temperature),
    }
    final_retrieval_span_predictor_metrics = _default_retrieval_span_metrics(device)
    success_step = None
    status = "completed"
    train_step_sec = 0.0
    has_robust_eval = False

    model.train()
    warmup_depth = get_niah_depth_for_optimizer_step(0, depths_to_test)
    X_w, Y_w, _ = generate_haystack_with_needle(batch_size, seq_len, resolved_data_vocab, warmup_depth)
    qp_w, _ = extract_query_positions_and_targets(X_w, Y_w, device)
    with torch.no_grad():
        _ = model.forward_selected_logits(X_w, qp_w)
    del X_w, Y_w, qp_w
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    started_at = time.perf_counter()
    for optimizer_step in range(epochs):
        current_depth = get_niah_depth_for_optimizer_step(optimizer_step, depths_to_test)
        X, Y, needle_positions = generate_haystack_with_needle(batch_size, seq_len, resolved_data_vocab, current_depth)
        query_positions, targets = extract_query_positions_and_targets(X, Y, device)

        step_start = time.perf_counter()
        optimizer.zero_grad()
        retrieval_aux = None
        retrieval_evidence_positions = needle_positions + retrieval_evidence_target_offset
        retrieval_span_train_positions = torch.stack(
            [needle_positions, needle_positions + 1],
            dim=1,
        )
        use_retrieval_aux = (
            retrieval_evidence_loss_alpha > 0.0
            or retrieval_projection_contrastive_alpha > 0.0
            or retrieval_span_predictor_alpha > 0.0
        )
        if use_retrieval_aux:
            logits_target, hidden_query, retrieval_aux = model.forward_selected_logits(
                X,
                query_positions,
                return_hidden=True,
                return_aux=True,
                return_retrieval_projection_aux=(
                    retrieval_projection_contrastive_alpha > 0.0
                ),
                train_retrieval_evidence_positions=(
                    retrieval_span_train_positions
                    if retrieval_span_predictor_alpha > 0.0
                    else retrieval_evidence_positions
                ),
            )
        else:
            logits_target, hidden_query = model.forward_selected_logits(
                X,
                query_positions,
                return_hidden=True,
            )
        loss = criterion(logits_target, targets)

        # Hidden-state MSE auxiliary loss
        if hidden_mse_alpha > 0.0:
            target_embed = model.embedding(targets).detach()
            loss_hidden = F.mse_loss(hidden_query, target_embed)
            loss = loss + hidden_mse_alpha * loss_hidden

        query_evidence_alignment_loss = torch.tensor(0.0, device=device)
        query_evidence_alignment_metrics = {
            "available": False,
            "loss": 0.0,
            "mean_cosine": None,
            "mean_mse": None,
            "detach_evidence": True,
        }
        if query_evidence_alignment_alpha > 0.0:
            evidence_token_positions = needle_positions + retrieval_evidence_target_offset
            evidence_tokens = X[
                torch.arange(batch_size, device=X.device),
                evidence_token_positions,
            ].to(device)
            (
                query_evidence_alignment_loss,
                query_evidence_alignment_metrics,
            ) = compute_query_evidence_alignment_loss(
                hidden_query,
                evidence_tokens,
                model.embedding,
                detach_evidence=True,
            )
            loss = loss + query_evidence_alignment_alpha * query_evidence_alignment_loss

        # Auxiliary loss at needle value position
        if needle_loss_alpha > 0.0:
            needle_val_positions = needle_positions + 1
            needle_val_targets = X[torch.arange(batch_size, device=X.device), needle_val_positions].to(device)
            logits_needle = model.forward_selected_logits(X, needle_val_positions)
            loss_needle_val = criterion(logits_needle, needle_val_targets)
            loss = loss + needle_loss_alpha * loss_needle_val

        retrieval_evidence_loss = torch.tensor(0.0, device=device)
        retrieval_evidence_metrics = {
            "available": False,
            "unavailable_reason": None,
            "hit_rate": None,
            "gate_mean": None,
            "evidence_weight_mean": None,
            "best_negative_weight_mean": None,
            "evidence_margin_mean": None,
            "target_rank_mean": None,
            "top1_rate": None,
            "ranking_loss": None,
            "margin_loss": None,
            "score_margin_loss": None,
            "evidence_score_mean": None,
            "best_negative_score_mean": None,
            "evidence_score_margin_mean": None,
            "score_target_rank_mean": None,
            "score_top1_rate": None,
            "gate_loss": None,
            "positive_count": 0,
        }
        if retrieval_evidence_loss_alpha > 0.0:
            retrieval_evidence_loss, retrieval_evidence_metrics = compute_retrieval_evidence_gate_loss(
                retrieval_aux,
                retrieval_evidence_positions,
                device=device,
                rank_margin=retrieval_evidence_rank_margin,
                score_margin=retrieval_evidence_score_margin,
            )
            loss = loss + retrieval_evidence_loss_alpha * retrieval_evidence_loss

        retrieval_projection_loss = torch.tensor(0.0, device=device)
        retrieval_projection_metrics = {
            "available": False,
            "unavailable_reason": None,
            "hit_rate": None,
            "positive_count": 0,
            "loss": 0.0,
            "evidence_score_mean": None,
            "best_negative_score_mean": None,
            "score_margin_mean": None,
            "target_rank_mean": None,
            "top1_rate": None,
            "temperature": float(retrieval_projection_temperature),
        }
        if retrieval_projection_contrastive_alpha > 0.0:
            (
                retrieval_projection_loss,
                retrieval_projection_metrics,
            ) = compute_retrieval_projection_contrastive_loss(
                retrieval_aux,
                retrieval_evidence_positions,
                device=device,
                temperature=retrieval_projection_temperature,
            )
            loss = loss + retrieval_projection_contrastive_alpha * retrieval_projection_loss

        retrieval_span_predictor_loss = torch.tensor(0.0, device=device)
        retrieval_span_predictor_metrics = _default_retrieval_span_metrics(device)
        if retrieval_span_predictor_alpha > 0.0:
            (
                retrieval_span_predictor_loss,
                retrieval_span_predictor_metrics,
            ) = compute_retrieval_span_predictor_loss(
                model,
                hidden_query,
                X,
                targets,
                query_positions,
                retrieval_aux,
                device=device,
                candidate_filter=niah_span_candidate_filter,
                loss_mode=niah_span_loss_mode,
            )
            loss = loss + retrieval_span_predictor_alpha * retrieval_span_predictor_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if hasattr(model, "layers"):
            for layer in model.layers:
                if hasattr(layer, "update_momentum"):
                    layer.update_momentum()
        train_step_sec += time.perf_counter() - step_start

        preds = logits_target.argmax(dim=-1)
        step_train_accuracy = (preds == targets).float().mean().item()
        loss_value = float(loss.item())
        best_step_train_accuracy = max(best_step_train_accuracy, step_train_accuracy)
        final_step_train_accuracy = step_train_accuracy
        final_train_loss = loss_value
        final_retrieval_evidence_metrics = retrieval_evidence_metrics
        final_retrieval_projection_metrics = retrieval_projection_metrics
        final_retrieval_span_predictor_metrics = retrieval_span_predictor_metrics
        should_stop = False

        light_eval_result = evaluate_niah_depths(
            model=model,
            seq_len=seq_len,
            device=device,
            vocab_size=resolved_data_vocab,
            batch_size=batch_size,
                criterion=criterion,
                depths=depths_to_test,
                batches_per_depth=eval_batches_per_depth,
                niah_readout_mode=niah_readout_mode,
                niah_span_candidate_filter=niah_span_candidate_filter,
            )
        final_light_eval = light_eval_result
        light_eval_mean_accuracy = light_eval_result["mean_accuracy"]
        light_eval_min_depth_accuracy = light_eval_result["min_depth_accuracy"]
        light_eval_mean_loss = light_eval_result["mean_loss"]
        sample_metrics.extend(
            {
                "optimizer_step": optimizer_step,
                "epoch": optimizer_step,
                **sample_row,
            }
            for sample_row in light_eval_result["sample_metrics"]
        )

        completed_step_count = optimizer_step + 1
        should_run_robust_eval = completed_step_count % resolved_robust_eval_interval == 0
        robust_eval_result = None
        if should_run_robust_eval:
            first_robust_eval = not has_robust_eval
            robust_eval_result = evaluate_niah_depths(
                model=model,
                seq_len=seq_len,
                device=device,
                vocab_size=resolved_data_vocab,
                batch_size=batch_size,
                criterion=criterion,
                depths=depths_to_test,
                batches_per_depth=robust_eval_batches_per_depth,
                niah_readout_mode=niah_readout_mode,
                niah_span_candidate_filter=niah_span_candidate_filter,
            )
            has_robust_eval = True
            final_robust_eval = robust_eval_result
            robust_eval_mean_accuracy = robust_eval_result["mean_accuracy"]
            robust_eval_min_depth_accuracy = robust_eval_result["min_depth_accuracy"]
            robust_eval_mean_loss = robust_eval_result["mean_loss"]
            robust_eval_row = {
                "optimizer_step": optimizer_step,
                "epoch": optimizer_step,
                "eval_source": "robust",
                "eval_mean_accuracy": robust_eval_mean_accuracy,
                "eval_min_depth_accuracy": robust_eval_min_depth_accuracy,
                "eval_mean_loss": robust_eval_mean_loss,
                "eval_depths": robust_eval_result["depth_rows"],
                "top1_accuracy": robust_eval_result["top1_accuracy"],
                "top3_accuracy": robust_eval_result["top3_accuracy"],
                "top5_accuracy": robust_eval_result["top5_accuracy"],
                "mean_target_rank": robust_eval_result["mean_target_rank"],
                "median_target_rank": robust_eval_result["median_target_rank"],
                "mean_target_prob": robust_eval_result["mean_target_prob"],
                "mean_pred_prob": robust_eval_result["mean_pred_prob"],
                "mean_logit_margin": robust_eval_result["mean_logit_margin"],
                "mean_prob_margin": robust_eval_result["mean_prob_margin"],
                "mean_entropy": robust_eval_result["mean_entropy"],
                "readout_mode": robust_eval_result["readout_mode"],
                "readout_available_rate": robust_eval_result.get(
                    "readout_available_rate"
                ),
                "target_candidate_hit_rate": robust_eval_result.get(
                    "target_candidate_hit_rate"
                ),
                "mean_target_candidate_rank": robust_eval_result.get(
                    "mean_target_candidate_rank"
                ),
                "total_samples": robust_eval_result["total_samples"],
            }
            robust_eval_rows.append(robust_eval_row)
            if first_robust_eval or robust_eval_mean_accuracy > best_accuracy:
                best_accuracy = robust_eval_mean_accuracy
                best_min_depth_accuracy = robust_eval_min_depth_accuracy
                best_accuracy_step = optimizer_step
                best_accuracy_loss = robust_eval_mean_loss
                best_eval_source = "robust"
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            final_accuracy = robust_eval_mean_accuracy
            final_min_depth_accuracy = robust_eval_min_depth_accuracy
            final_eval_loss = robust_eval_mean_loss
            final_loss = robust_eval_mean_loss
            final_eval_source = "robust"
            should_stop = (
                robust_eval_min_depth_accuracy >= target_accuracy
                and robust_eval_mean_loss < stop_loss
                and robust_eval_result["total_samples"] >= NIAH_MIN_EVAL_SAMPLES_FOR_EARLY_STOP
            )
        elif not has_robust_eval:
            if best_accuracy_step is None or light_eval_mean_accuracy > best_accuracy:
                best_accuracy = light_eval_mean_accuracy
                best_min_depth_accuracy = light_eval_min_depth_accuracy
                best_accuracy_step = optimizer_step
                best_accuracy_loss = light_eval_mean_loss
                best_eval_source = "light"
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            final_accuracy = light_eval_mean_accuracy
            final_min_depth_accuracy = light_eval_min_depth_accuracy
            final_eval_loss = light_eval_mean_loss
            final_loss = light_eval_mean_loss
            final_eval_source = "light"

        row = {
            "optimizer_step": optimizer_step,
            "epoch": optimizer_step,
            "depth": current_depth,
            "loss": loss_value,
            "accuracy": step_train_accuracy,
            "train_depth": current_depth,
            "train_loss": loss_value,
            "train_accuracy": step_train_accuracy,
            "step_train_accuracy": step_train_accuracy,
            "eval_source": "light",
            "eval_mean_accuracy": light_eval_mean_accuracy,
            "eval_min_depth_accuracy": light_eval_min_depth_accuracy,
            "eval_mean_loss": light_eval_mean_loss,
            "eval_depths": light_eval_result["depth_rows"],
            "top1_accuracy": light_eval_result["top1_accuracy"],
            "top3_accuracy": light_eval_result["top3_accuracy"],
            "top5_accuracy": light_eval_result["top5_accuracy"],
            "mean_target_rank": light_eval_result["mean_target_rank"],
            "median_target_rank": light_eval_result["median_target_rank"],
            "mean_target_prob": light_eval_result["mean_target_prob"],
            "mean_pred_prob": light_eval_result["mean_pred_prob"],
            "mean_logit_margin": light_eval_result["mean_logit_margin"],
            "mean_prob_margin": light_eval_result["mean_prob_margin"],
            "mean_entropy": light_eval_result["mean_entropy"],
            "readout_mode": light_eval_result["readout_mode"],
            "readout_available_rate": light_eval_result.get("readout_available_rate"),
            "target_candidate_hit_rate": light_eval_result.get(
                "target_candidate_hit_rate"
            ),
            "mean_target_candidate_rank": light_eval_result.get(
                "mean_target_candidate_rank"
            ),
            "total_samples": light_eval_result["total_samples"],
            "retrieval_evidence_loss": float(
                retrieval_evidence_loss.detach().cpu().item()
            ),
            "retrieval_evidence_available": retrieval_evidence_metrics["available"],
            "retrieval_evidence_unavailable_reason": retrieval_evidence_metrics[
                "unavailable_reason"
            ],
            "retrieval_evidence_hit_rate": retrieval_evidence_metrics["hit_rate"],
            "retrieval_evidence_gate_mean": retrieval_evidence_metrics["gate_mean"],
            "retrieval_evidence_weight_mean": retrieval_evidence_metrics[
                "evidence_weight_mean"
            ],
            "retrieval_evidence_best_negative_weight_mean": retrieval_evidence_metrics[
                "best_negative_weight_mean"
            ],
            "retrieval_evidence_margin_mean": retrieval_evidence_metrics[
                "evidence_margin_mean"
            ],
            "retrieval_evidence_target_rank_mean": retrieval_evidence_metrics[
                "target_rank_mean"
            ],
            "retrieval_evidence_top1_rate": retrieval_evidence_metrics["top1_rate"],
            "retrieval_evidence_ranking_loss": retrieval_evidence_metrics["ranking_loss"],
            "retrieval_evidence_margin_loss": retrieval_evidence_metrics["margin_loss"],
            "retrieval_evidence_score_margin_loss": retrieval_evidence_metrics[
                "score_margin_loss"
            ],
            "retrieval_evidence_score_mean": retrieval_evidence_metrics[
                "evidence_score_mean"
            ],
            "retrieval_evidence_best_negative_score_mean": retrieval_evidence_metrics[
                "best_negative_score_mean"
            ],
            "retrieval_evidence_score_margin_mean": retrieval_evidence_metrics[
                "evidence_score_margin_mean"
            ],
            "retrieval_evidence_score_target_rank_mean": retrieval_evidence_metrics[
                "score_target_rank_mean"
            ],
            "retrieval_evidence_score_top1_rate": retrieval_evidence_metrics[
                "score_top1_rate"
            ],
            "retrieval_evidence_gate_loss": retrieval_evidence_metrics["gate_loss"],
            "retrieval_evidence_positive_count": retrieval_evidence_metrics["positive_count"],
            "query_evidence_alignment_loss": float(
                query_evidence_alignment_loss.detach().cpu().item()
            ),
            "query_evidence_alignment_available": query_evidence_alignment_metrics[
                "available"
            ],
            "query_evidence_alignment_mean_cosine": query_evidence_alignment_metrics[
                "mean_cosine"
            ],
            "query_evidence_alignment_mean_mse": query_evidence_alignment_metrics[
                "mean_mse"
            ],
            "retrieval_projection_loss": float(
                retrieval_projection_loss.detach().cpu().item()
            ),
            "retrieval_projection_available": retrieval_projection_metrics["available"],
            "retrieval_projection_unavailable_reason": retrieval_projection_metrics[
                "unavailable_reason"
            ],
            "retrieval_projection_hit_rate": retrieval_projection_metrics["hit_rate"],
            "retrieval_projection_positive_count": retrieval_projection_metrics[
                "positive_count"
            ],
            "retrieval_projection_evidence_score_mean": retrieval_projection_metrics[
                "evidence_score_mean"
            ],
            "retrieval_projection_best_negative_score_mean": (
                retrieval_projection_metrics["best_negative_score_mean"]
            ),
            "retrieval_projection_score_margin_mean": retrieval_projection_metrics[
                "score_margin_mean"
            ],
            "retrieval_projection_target_rank_mean": retrieval_projection_metrics[
                "target_rank_mean"
            ],
            "retrieval_projection_top1_rate": retrieval_projection_metrics["top1_rate"],
            "retrieval_span_predictor_loss": float(
                retrieval_span_predictor_loss.detach().cpu().item()
            ),
            "retrieval_span_predictor_available": retrieval_span_predictor_metrics[
                "available"
            ],
            "retrieval_span_predictor_unavailable_reason": (
                retrieval_span_predictor_metrics["unavailable_reason"]
            ),
            "retrieval_span_predictor_hit_rate": retrieval_span_predictor_metrics[
                "hit_rate"
            ],
            "retrieval_span_predictor_positive_count": (
                retrieval_span_predictor_metrics["positive_count"]
            ),
            "retrieval_span_predictor_target_rank_mean": (
                retrieval_span_predictor_metrics["target_rank_mean"]
            ),
            "retrieval_span_predictor_top1_rate": retrieval_span_predictor_metrics[
                "top1_rate"
            ],
            "retrieval_span_predictor_evidence_logit_mean": (
                retrieval_span_predictor_metrics["evidence_logit_mean"]
            ),
            "retrieval_span_predictor_best_negative_logit_mean": (
                retrieval_span_predictor_metrics["best_negative_logit_mean"]
            ),
            "retrieval_span_predictor_logit_margin_mean": (
                retrieval_span_predictor_metrics["logit_margin_mean"]
            ),
        }
        step_rows.append(row)
        print(
            f"Step {optimizer_step:3d} | Train Depth: {current_depth:.1f} | "
            f"Train Loss: {loss_value:.4f} | Step Train Accuracy: {step_train_accuracy*100:5.1f}% | "
            f"Light Eval Mean Accuracy: {light_eval_mean_accuracy*100:5.1f}% | "
            f"Light Eval Min-Depth Accuracy: {light_eval_min_depth_accuracy*100:5.1f}%"
        )
        if swanlab_run.enabled:
            swanlab_payload = {
                "train/loss": loss_value,
                "train/accuracy": step_train_accuracy,
                "train/depth": current_depth,
                "train/retrieval_evidence_loss": float(
                    retrieval_evidence_loss.detach().cpu().item()
                ),
                "train/retrieval_projection_loss": float(
                    retrieval_projection_loss.detach().cpu().item()
                ),
                "train/retrieval_span_predictor_loss": float(
                    retrieval_span_predictor_loss.detach().cpu().item()
                ),
            }
            add_niah_eval_metrics_to_swanlab_payload(
                swanlab_payload,
                "eval/light",
                light_eval_result,
                include_samples=True,
            )
            if robust_eval_result is not None:
                add_niah_eval_metrics_to_swanlab_payload(
                    swanlab_payload,
                    "eval/robust",
                    robust_eval_result,
                    include_samples=False,
                )
            swanlab_run.log(swanlab_payload, step=optimizer_step)
        del X, Y, query_positions, logits_target, targets, loss
        if should_stop:
            status = "success"
            success_step = optimizer_step
            break

    re_eval_mean_accuracy = None
    re_eval_min_depth_accuracy = None
    final_readout_available_rate = (
        None
        if final_light_eval is None
        else final_light_eval.get("readout_available_rate")
    )
    final_target_candidate_hit_rate = (
        None
        if final_light_eval is None
        else final_light_eval.get("target_candidate_hit_rate")
    )
    final_mean_target_candidate_rank = (
        None
        if final_light_eval is None
        else final_light_eval.get("mean_target_candidate_rank")
    )
    final_span_candidate_diagnostics = (
        {}
        if final_light_eval is None
        else _span_candidate_diagnostics_from_eval(final_light_eval)
    )

    if best_state_dict is not None and status != "success":
        model.load_state_dict(best_state_dict)
        re_eval_batches_per_depth = (
            robust_eval_batches_per_depth if has_robust_eval else eval_batches_per_depth
        )
        final_eval = evaluate_niah_depths(
            model=model,
            seq_len=seq_len,
            device=device,
            vocab_size=resolved_data_vocab,
            batch_size=batch_size,
            criterion=criterion,
            depths=depths_to_test,
            batches_per_depth=re_eval_batches_per_depth,
            niah_readout_mode=niah_readout_mode,
            niah_span_candidate_filter=niah_span_candidate_filter,
        )
        re_eval_mean_accuracy = final_eval["mean_accuracy"]
        re_eval_min_depth_accuracy = final_eval["min_depth_accuracy"]
        final_accuracy = final_eval["mean_accuracy"]
        final_min_depth_accuracy = final_eval["min_depth_accuracy"]
        final_eval_loss = final_eval["mean_loss"]
        final_loss = final_eval["mean_loss"]
        final_eval_source = best_eval_source
        final_readout_available_rate = final_eval.get("readout_available_rate")
        final_target_candidate_hit_rate = final_eval.get("target_candidate_hit_rate")
        final_mean_target_candidate_rank = final_eval.get("mean_target_candidate_rank")
        final_span_candidate_diagnostics = _span_candidate_diagnostics_from_eval(
            final_eval
        )
    elif final_robust_eval is not None and final_eval_source == "robust":
        final_readout_available_rate = final_robust_eval.get("readout_available_rate")
        final_target_candidate_hit_rate = final_robust_eval.get(
            "target_candidate_hit_rate"
        )
        final_mean_target_candidate_rank = final_robust_eval.get(
            "mean_target_candidate_rank"
        )
        final_span_candidate_diagnostics = _span_candidate_diagnostics_from_eval(
            final_robust_eval
        )

    test_eval = None
    test_span_candidate_diagnostics = {}
    if niah_test_batches_per_depth > 0:
        rng_state = capture_niah_rng_state(device)
        try:
            seed_all(seed + NIAH_TEST_SEED_OFFSET, cudnn_benchmark=cudnn_benchmark)
            test_eval = evaluate_niah_depths(
                model=model,
                seq_len=seq_len,
                device=device,
                vocab_size=resolved_data_vocab,
                batch_size=batch_size,
                criterion=criterion,
                depths=depths_to_test,
                batches_per_depth=niah_test_batches_per_depth,
                niah_readout_mode=niah_readout_mode,
                niah_span_candidate_filter=niah_span_candidate_filter,
            )
            test_span_candidate_diagnostics = _span_candidate_diagnostics_from_eval(
                test_eval
            )
        finally:
            restore_niah_rng_state(rng_state)

    elapsed_sec = time.perf_counter() - started_at
    train_retrieval_evidence_summary = summarize_retrieval_evidence_step_metrics(
        step_rows
    )
    train_query_evidence_alignment_summary = (
        summarize_query_evidence_alignment_step_metrics(step_rows)
    )
    train_retrieval_projection_summary = summarize_retrieval_projection_step_metrics(
        step_rows
    )
    train_retrieval_span_predictor_summary = (
        summarize_retrieval_span_predictor_step_metrics(step_rows)
    )
    peak_allocated_mb = 0.0
    peak_reserved_mb = 0.0
    if device.type == "cuda":
        peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

    final_aux_diagnostics = None
    if hasattr(model, "forward_selected_logits"):
        try:
            diagnostic_depth = depths_to_test[-1] if depths_to_test else 0.5
            X_diag, Y_diag, _ = generate_haystack_with_needle(
                1,
                seq_len,
                resolved_data_vocab,
                diagnostic_depth,
            )
            query_positions_diag, _ = extract_query_positions_and_targets(
                X_diag,
                Y_diag,
                device,
            )
            model.eval()
            with torch.no_grad():
                diagnostic_result = model.forward_selected_logits(
                    X_diag,
                    query_positions_diag,
                    return_aux=True,
                )
            if isinstance(diagnostic_result, tuple) and len(diagnostic_result) == 2:
                final_aux_diagnostics = diagnostic_result[1]
            del X_diag, Y_diag, query_positions_diag, diagnostic_result
        except (AttributeError, RuntimeError, ValueError) as exc:
            final_aux_diagnostics = {
                "error": str(exc),
            }

    parameter_count = sum(p.numel() for p in model.parameters())
    result = {
        "status": status,
        "seq_len": seq_len,
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "parameter_count": parameter_count,
        "config": {
            "vocab_size": vocab_size,
            "dim": dim,
            "num_layers": num_layers,
            "slots": K,
            "read_topk": kr,
            "chunk_size": chunk_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "optimizer_steps": epochs,
            "learning_rate": learning_rate,
            "seed": seed,
            "target_accuracy": target_accuracy,
            "stop_loss": stop_loss,
            "eval_interval": resolved_robust_eval_interval,
            "eval_batches_per_depth": eval_batches_per_depth,
            "light_eval_batches_per_depth": eval_batches_per_depth,
            "robust_eval_interval": resolved_robust_eval_interval,
            "robust_eval_batches_per_depth": robust_eval_batches_per_depth,
            "niah_test_batches_per_depth": niah_test_batches_per_depth,
            "niah_test_seed": (
                seed + NIAH_TEST_SEED_OFFSET
                if niah_test_batches_per_depth > 0
                else None
            ),
            "eval_depths": list(depths_to_test),
            "cudnn_benchmark": cudnn_benchmark,
            "use_retrieval": use_retrieval,
            "niah_readout_mode": niah_readout_mode,
            "niah_span_candidate_filter": niah_span_candidate_filter,
            "niah_span_loss_mode": niah_span_loss_mode,
            "retrieval_evidence_loss_alpha": retrieval_evidence_loss_alpha,
            "retrieval_evidence_rank_margin": retrieval_evidence_rank_margin,
            "retrieval_evidence_score_margin": retrieval_evidence_score_margin,
            "retrieval_evidence_target_offset": retrieval_evidence_target_offset,
            "query_evidence_alignment_alpha": query_evidence_alignment_alpha,
            "retrieval_projection_contrastive_alpha": retrieval_projection_contrastive_alpha,
            "retrieval_projection_temperature": retrieval_projection_temperature,
            "retrieval_span_predictor_alpha": retrieval_span_predictor_alpha,
            "retrieval_span_structure_features": bool(retrieval_span_structure_features),
        },
        "best_accuracy": best_accuracy,
        "best_min_depth_accuracy": best_min_depth_accuracy,
        "best_accuracy_step": best_accuracy_step,
        "best_accuracy_epoch": best_accuracy_step,
        "best_accuracy_loss": best_accuracy_loss,
        "best_eval_source": best_eval_source,
        "final_accuracy": final_accuracy,
        "final_min_depth_accuracy": final_min_depth_accuracy,
        "final_eval_loss": final_eval_loss,
        "final_eval_source": final_eval_source,
        "final_light_accuracy": None if final_light_eval is None else final_light_eval["mean_accuracy"],
        "final_light_min_depth_accuracy": None if final_light_eval is None else final_light_eval["min_depth_accuracy"],
        "final_light_eval_loss": None if final_light_eval is None else final_light_eval["mean_loss"],
        "final_robust_accuracy": None if final_robust_eval is None else final_robust_eval["mean_accuracy"],
        "final_robust_min_depth_accuracy": None if final_robust_eval is None else final_robust_eval["min_depth_accuracy"],
        "final_robust_eval_loss": None if final_robust_eval is None else final_robust_eval["mean_loss"],
        "best_step_train_accuracy": best_step_train_accuracy,
        "final_step_train_accuracy": final_step_train_accuracy,
        "final_train_loss": final_train_loss,
        "final_loss": final_loss,
        "success_step": success_step,
        "success_epoch": success_step,
        "steps_observed": step_rows,
        "epochs_observed": step_rows,
        "robust_evals_observed": robust_eval_rows,
        "sample_metrics": sample_metrics,
        "elapsed_sec": elapsed_sec,
        "train_step_sec": train_step_sec,
        "peak_memory_allocated_mb": peak_allocated_mb,
        "peak_memory_reserved_mb": peak_reserved_mb,
        "final_aux_diagnostics": final_aux_diagnostics,
        "final_retrieval_evidence_metrics": final_retrieval_evidence_metrics,
        "final_retrieval_projection_metrics": final_retrieval_projection_metrics,
        "final_retrieval_span_predictor_metrics": final_retrieval_span_predictor_metrics,
        "train_retrieval_evidence_summary": train_retrieval_evidence_summary,
        "train_query_evidence_alignment_summary": train_query_evidence_alignment_summary,
        "train_retrieval_projection_summary": train_retrieval_projection_summary,
        "train_retrieval_span_predictor_summary": train_retrieval_span_predictor_summary,
        "final_readout_mode": niah_readout_mode,
        "final_readout_available_rate": final_readout_available_rate,
        "final_target_candidate_hit_rate": final_target_candidate_hit_rate,
        "final_mean_target_candidate_rank": final_mean_target_candidate_rank,
        "final_span_candidate_diagnostics": final_span_candidate_diagnostics,
        "test_accuracy": None if test_eval is None else test_eval["mean_accuracy"],
        "test_min_depth_accuracy": (
            None if test_eval is None else test_eval["min_depth_accuracy"]
        ),
        "test_eval_loss": None if test_eval is None else test_eval["mean_loss"],
        "test_readout_available_rate": (
            None if test_eval is None else test_eval.get("readout_available_rate")
        ),
        "test_target_candidate_hit_rate": (
            None if test_eval is None else test_eval.get("target_candidate_hit_rate")
        ),
        "test_mean_target_candidate_rank": (
            None if test_eval is None else test_eval.get("mean_target_candidate_rank")
        ),
        "test_span_candidate_diagnostics": test_span_candidate_diagnostics,
        "passed_accuracy": final_min_depth_accuracy >= target_accuracy,
        "passed_success_criteria": status == "success",
        "re_eval_mean_accuracy": re_eval_mean_accuracy,
        "re_eval_min_depth_accuracy": re_eval_min_depth_accuracy,
    }

    if save_checkpoint is not None and best_state_dict is not None:
        ckpt_path = resolve_niah_checkpoint_path(save_checkpoint, create_parent=True)
        torch.save({
            "model_state_dict": best_state_dict,
            "dim": dim,
            "num_layers": num_layers,
            "slots": K,
            "read_topk": kr,
            "vocab_size": vocab_size,
            "chunk_size": chunk_size,
            "step": best_accuracy_step,
            "eval_accuracy": best_accuracy,
        }, ckpt_path)
        print(
            f"Saved best checkpoint to {ckpt_path} "
            f"(step {best_accuracy_step}, acc {best_accuracy*100:.2f}%)"
        )

    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    swanlab_run.finish()
    return result


def build_niah_verification_markdown(title, result):
    """Build a Markdown report for one NIAH verification case.

    中文说明:
    - 调用方 / Called by: `save_niah_verification_report`
    - 调用对象 / Calls: string formatting only
    - 作用 / Purpose: 将 2M NIAH 或更大参数 benchmark 结果转换为 reports/ Markdown 交付物；
      同时展示训练准确率、独立 eval mean/min-depth accuracy 和成功条件，避免把一次训练样本
      命中误读为稳定通过
    - 参数 / Parameters: `title` 是报告标题；`result` 是 `run_niah_verification_case` 返回值
    - 返回 / Returns: Markdown 行列表
    - 接入 / Integration: 报告写入统一通过 `write_markdown`
    - 错误处理 / Error handling: 缺少必要字段会由 Python 字典访问抛出异常
    - 副作用 / Side effects: 无
    - 事务边界 / Transaction boundary: 不涉及事务
    - 并发与幂等 / Concurrency and idempotency: 纯格式化函数
    - 关键词 / Keywords:
      markdown|report|niah|2m|benchmark|memory|accuracy|reports|mhdsra2|报告

    English documentation:
    Function name:
        build_niah_verification_markdown
    Purpose:
        Build Markdown lines for a NIAH verification result, separating train
        accuracy, independent eval mean/min-depth accuracy, and full success criteria.
    Called by:
        `save_niah_verification_report`.
    Calls:
        String formatting only.
    Parameters:
        - title: report title.
        - result: result dictionary from the verification runner.
    Returns:
        List of Markdown lines.
    Error handling:
        Missing required fields raise normal Python errors.
    Side effects:
        None.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Pure formatting.
    English keywords:
        markdown, report, niah, 2m, benchmark, memory, accuracy, reports, mhdsra2, summary
    """
    cfg = result["config"]
    steps_observed = result.get("steps_observed", result.get("epochs_observed", []))
    robust_evals_observed = result.get("robust_evals_observed", [])
    final_accuracy = result.get("final_accuracy", result["best_accuracy"])
    final_min_depth_accuracy = result.get("final_min_depth_accuracy")
    final_min_depth_display = (
        "None" if final_min_depth_accuracy is None else f"{final_min_depth_accuracy*100:.2f}%"
    )
    best_min_depth_accuracy = result.get("best_min_depth_accuracy")
    best_min_depth_display = (
        "None" if best_min_depth_accuracy is None else f"{best_min_depth_accuracy*100:.2f}%"
    )
    best_accuracy_loss = result.get("best_accuracy_loss")
    best_accuracy_loss_display = (
        "None" if best_accuracy_loss is None else f"{best_accuracy_loss:.6f}"
    )
    passed_success_criteria = result.get("passed_success_criteria", result["status"] == "success")
    re_eval_mean = result.get("re_eval_mean_accuracy")
    re_eval_min = result.get("re_eval_min_depth_accuracy")
    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        f"- status: `{result['status']}`",
        f"- sequence length: `{result['seq_len']}`",
        f"- final eval source: `{result.get('final_eval_source', 'legacy')}`",
        f"- best eval source: `{result.get('best_eval_source', 'legacy')}`",
        f"- final eval mean accuracy: `{final_accuracy*100:.2f}%`",
        f"- final eval min-depth accuracy: `{final_min_depth_display}`",
        f"- best eval mean accuracy: `{result['best_accuracy']*100:.2f}%`",
        f"- best eval min-depth accuracy: `{best_min_depth_display}`",
        f"- best accuracy step: `{result.get('best_accuracy_step', result.get('best_accuracy_epoch'))}`",
        f"- best accuracy loss: `{best_accuracy_loss_display}`",
        f"- final step train accuracy: `{result.get('final_step_train_accuracy', 0.0)*100:.2f}%`",
        f"- best step train accuracy: `{result.get('best_step_train_accuracy', 0.0)*100:.2f}%`",
        f"- passed target accuracy: `{result['passed_accuracy']}`",
        f"- passed success criteria: `{passed_success_criteria}`",
    ]
    if re_eval_mean is not None:
        lines.extend([
            f"- re-eval mean accuracy: `{re_eval_mean*100:.2f}%`",
            f"- re-eval min-depth accuracy: `{re_eval_min*100:.2f}%`",
        ])
    lines.extend([
        f"- final loss: `{result['final_loss']:.6f}`",
        f"- elapsed seconds: `{result['elapsed_sec']:.3f}`",
        f"- peak allocated memory: `{result['peak_memory_allocated_mb']:.2f} MB`",
        f"- peak reserved memory: `{result['peak_memory_reserved_mb']:.2f} MB`",
        f"- parameter count: `{result['parameter_count']}`",
        f"- device: `{result['device']}`",
        f"- CUDA device: `{result['cuda_device_name']}`",
        f"- torch: `{result['torch_version']}`",
        f"- torch CUDA: `{result['torch_cuda_version']}`",
        "",
        "## Config",
        "",
        "| Field | Value |",
        "|---|---:|",
    ])
    for key in (
        "vocab_size",
        "dim",
        "num_layers",
        "slots",
        "read_topk",
        "chunk_size",
        "batch_size",
        "epochs",
        "optimizer_steps",
        "learning_rate",
        "seed",
        "target_accuracy",
        "stop_loss",
        "eval_interval",
        "eval_batches_per_depth",
        "light_eval_batches_per_depth",
        "robust_eval_interval",
        "robust_eval_batches_per_depth",
        "eval_depths",
        "cudnn_benchmark",
    ):
        if key in cfg:
            lines.append(f"| {key} | `{cfg[key]}` |")

    lines.extend(
        [
            "",
            "## Observed Steps",
            "",
            "| Step | Train Depth | Train Loss | Train Accuracy | Light Mean Accuracy | Light Min-Depth Accuracy | Light Loss | Top-3 | Target Rank | Entropy |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in steps_observed:
        train_depth = row.get("train_depth", row.get("depth"))
        train_loss = row.get("train_loss", row.get("loss"))
        train_accuracy = row.get("train_accuracy", row.get("accuracy"))
        eval_mean_accuracy = row.get("eval_mean_accuracy")
        eval_min_depth_accuracy = row.get("eval_min_depth_accuracy")
        eval_mean_loss = row.get("eval_mean_loss")
        ema_str = f"{eval_mean_accuracy*100:.2f}%" if eval_mean_accuracy is not None else "N/A"
        emda_str = f"{eval_min_depth_accuracy*100:.2f}%" if eval_min_depth_accuracy is not None else "N/A"
        eml_str = f"{eval_mean_loss:.6f}" if eval_mean_loss is not None else "N/A"
        top3_str = f"{row.get('top3_accuracy', 0.0)*100:.2f}%"
        rank_str = f"{row.get('mean_target_rank', 0.0):.2f}"
        entropy_str = f"{row.get('mean_entropy', 0.0):.4f}"
        lines.append(
            f"| {row.get('optimizer_step', row.get('epoch'))} | {train_depth:.1f} | {train_loss:.6f} | "
            f"{train_accuracy*100:.2f}% | {ema_str} | "
            f"{emda_str} | {eml_str} | {top3_str} | {rank_str} | {entropy_str} |"
        )
    if robust_evals_observed:
        lines.extend(
            [
                "",
                "## Robust Evaluations",
                "",
                "| Step | Mean Accuracy | Min-Depth Accuracy | Mean Loss | Top-3 | Target Rank | Entropy | Samples |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in robust_evals_observed:
            lines.append(
                f"| {row.get('optimizer_step', row.get('epoch'))} | "
                f"{row['eval_mean_accuracy']*100:.2f}% | "
                f"{row['eval_min_depth_accuracy']*100:.2f}% | "
                f"{row['eval_mean_loss']:.6f} | "
                f"{row.get('top3_accuracy', 0.0)*100:.2f}% | "
                f"{row.get('mean_target_rank', 0.0):.2f} | "
                f"{row.get('mean_entropy', 0.0):.4f} | "
                f"{row.get('total_samples', 0)} |"
            )
    if steps_observed:
        final_step = steps_observed[-1]
        lines.extend(
            [
                "",
                "## Latest Light Per-Depth Metrics",
                "",
                "| Depth | Accuracy | Loss | Top-3 | Target Rank | Target Prob | Logit Margin | Entropy | Samples |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for depth_row in final_step.get("eval_depths", []):
            lines.append(
                f"| {depth_row['depth']:.1f} | "
                f"{depth_row['accuracy']*100:.2f}% | "
                f"{depth_row['loss']:.6f} | "
                f"{depth_row.get('top3_accuracy', 0.0)*100:.2f}% | "
                f"{depth_row.get('mean_target_rank', 0.0):.2f} | "
                f"{depth_row.get('mean_target_prob', 0.0):.4f} | "
                f"{depth_row.get('mean_logit_margin', 0.0):.4f} | "
                f"{depth_row.get('mean_entropy', 0.0):.4f} | "
                f"{depth_row.get('samples', 0)} |"
            )
    return lines


def sanitize_report_name(report_name):
    """Validate a report file stem before joining it with reports/.

    中文说明:
    - 调用方 / Called by: `save_niah_verification_report`
    - 调用对象 / Calls: `Path`
    - 作用 / Purpose: 防止 `--report-name` 通过 `../`、绝对路径或 Windows 盘符逃出 reports/
    - 参数 / Parameters: `report_name` 是用户传入的报告文件名前缀
    - 返回 / Returns: 校验后的普通文件名前缀字符串
    - 错误处理 / Error handling: 空值、路径片段或非法字符抛 `ValueError`
    - 副作用 / Side effects: 无；只做字符串与路径结构校验
    - 关键词 / Keywords: report_name|path_traversal|reports|sanitize|security|路径逃逸

    English documentation:
    Function name:
        sanitize_report_name
    Purpose:
        Ensure a report stem is a plain filename before it is written under reports/.
    Called by:
        `save_niah_verification_report`.
    Calls:
        `Path`.
    Parameters:
        - report_name: user-provided report stem.
    Returns:
        Sanitized report stem.
    Error handling:
        Raises `ValueError` for empty names, path components, absolute paths, drives, or invalid characters.
    Side effects:
        None.
    """
    raw_name = str(report_name)
    if raw_name.strip() != raw_name or raw_name == "":
        raise ValueError("--report-name must be a non-empty plain filename stem")
    candidate = Path(raw_name)
    if (
        candidate.is_absolute()
        or candidate.drive
        or candidate.root
        or candidate.name != raw_name
        or any(part in {".", ".."} for part in candidate.parts)
        or any(char in INVALID_REPORT_NAME_CHARS for char in raw_name)
    ):
        raise ValueError(
            "--report-name must be a plain filename stem under reports/; "
            "path separators, absolute paths, drive prefixes, and '..' are not allowed"
        )
    return raw_name


def resolve_niah_reports_dir(reports_dir, project_root=None):
    """Resolve the NIAH verification output directory to project reports/.

    中文说明:
    - 调用方 / Called by: `save_niah_verification_report`
    - 调用对象 / Calls: `Path.resolve`, `Path.mkdir`
    - 作用 / Purpose: 防止 `--reports-dir` 指向项目外目录，确保 NIAH 验证报告仍写入项目 reports/
    - 参数 / Parameters: `reports_dir` 是 CLI 传入目录；`project_root` 测试可覆盖，默认项目根
    - 返回 / Returns: 已创建且解析后的项目 reports/ 路径
    - 错误处理 / Error handling: 解析后不是项目 `reports/` 时抛 `ValueError`
    - 副作用 / Side effects: 创建项目 reports/ 目录
    - 关键词 / Keywords: reports_dir|project_reports|path_boundary|niah|security|路径边界

    English documentation:
    Function name:
        resolve_niah_reports_dir
    Purpose:
        Resolve NIAH verification report output to the project's reports/ directory.
    Called by:
        `save_niah_verification_report`.
    Calls:
        `Path.resolve` and `Path.mkdir`.
    Parameters:
        - reports_dir: CLI-provided output directory.
        - project_root: optional root override for tests.
    Returns:
        Resolved project reports directory.
    Error handling:
        Raises `ValueError` when the path resolves outside project reports/.
    Side effects:
        Creates the reports directory.
    """
    root = Path(PROJECT_ROOT if project_root is None else project_root).resolve()
    raw_reports_dir = Path(reports_dir)
    candidate = raw_reports_dir if raw_reports_dir.is_absolute() else root / raw_reports_dir
    if candidate.parts[-2:] != ("docs", "reports"):
        if candidate.name == "docs":
            candidate = candidate / "reports"
        elif candidate.name == "reports":
            candidate = candidate.parent / "docs" / "reports"
        else:
            candidate = candidate / "docs" / "reports"
    resolved_reports_dir = candidate.resolve()
    expected_reports_dir = (root / "docs" / "reports").resolve()
    if resolved_reports_dir != expected_reports_dir:
        raise ValueError(
            "--reports-dir must resolve to the project docs/reports/ directory; "
            f"got {resolved_reports_dir}"
        )
    resolved_reports_dir.mkdir(parents=True, exist_ok=True)
    return resolved_reports_dir



def resolve_niah_checkpoint_path(checkpoint_path, *, project_root=None, create_parent=False):
    """Resolve a NIAH checkpoint path inside reports/checkpoints/.

    中文说明:
    - 调用方 / Called by: `run_niah_verification_case`
    - 调用对象 / Calls: `Path.resolve`, `Path.mkdir`
    - 作用 / Purpose: 防止 `--save-checkpoint` 或 `--load-checkpoint` 读写项目外任意文件
    - 参数 / Parameters: `checkpoint_path` 是 CLI 路径；`project_root` 测试可覆盖；
      `create_parent` 控制是否创建父目录
    - 返回 / Returns: 解析后的 `reports/checkpoints/*.pt|*.pth|*.ckpt` 路径
    - 错误处理 / Error handling: 路径逃逸或后缀非法时抛 `ValueError`
    - 副作用 / Side effects: `create_parent=True` 时创建父目录
    - 关键词 / Keywords: checkpoint|reports|path_boundary|niah|torch_save|security

    English documentation:
    Function name:
        resolve_niah_checkpoint_path
    Purpose:
        Keep NIAH checkpoints inside project reports/checkpoints/.
    Called by:
        `run_niah_verification_case`.
    Calls:
        `Path.resolve` and `Path.mkdir`.
    Parameters:
        - checkpoint_path: CLI-provided path.
        - project_root: optional root override for tests.
        - create_parent: whether to create the parent directory.
    Returns:
        Resolved checkpoint path.
    Error handling:
        Raises `ValueError` for escaped paths or unsupported suffixes.
    Side effects:
        May create the parent directory.
    """
    root = Path(PROJECT_ROOT if project_root is None else project_root).resolve()
    checkpoint_dir = (root / "docs" / "reports" / "checkpoints").resolve()
    raw_path = Path(checkpoint_path)
    if raw_path.is_absolute():
        candidate = raw_path
    elif len(raw_path.parts) == 1:
        candidate = checkpoint_dir / raw_path
    else:
        candidate = root / raw_path
    resolved = candidate.resolve()
    if not resolved.is_relative_to(checkpoint_dir):
        raise ValueError(
            "--load-checkpoint/--save-checkpoint must stay inside docs/reports/checkpoints/; "
            f"got {resolved}"
        )
    if resolved.suffix.lower() not in NIAH_CHECKPOINT_SUFFIXES:
        raise ValueError(
            "NIAH checkpoint path must use one of "
            f"{sorted(NIAH_CHECKPOINT_SUFFIXES)}; got {resolved.suffix!r}"
        )
    if create_parent:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def build_report_output_path(reports_dir, report_name, suffix):
    """Build and verify one report output path under reports/.

    中文说明:
    - 调用方 / Called by: `save_niah_verification_report`
    - 调用对象 / Calls: `sanitize_report_name`, `Path.resolve`, `Path.is_relative_to`
    - 作用 / Purpose: 在最终写文件前确认 JSON/Markdown 输出仍位于 reports_dir 内
    - 参数 / Parameters: `reports_dir` 是已创建目录；`report_name` 是文件名前缀；`suffix` 是扩展名
    - 返回 / Returns: 安全的输出路径
    - 错误处理 / Error handling: 路径不在 reports_dir 内时抛 `ValueError`
    - 副作用 / Side effects: 无；不创建或写入文件
    - 关键词 / Keywords: reports|output_path|resolve|guard|json|markdown|路径校验

    English documentation:
    Function name:
        build_report_output_path
    Purpose:
        Build a report path and assert it remains inside the reports directory.
    Called by:
        `save_niah_verification_report`.
    Calls:
        `sanitize_report_name`, `Path.resolve`, and `Path.is_relative_to`.
    Parameters:
        - reports_dir: resolved reports directory.
        - report_name: report stem.
        - suffix: output suffix such as `.json`.
    Returns:
        Safe output path.
    Error handling:
        Raises `ValueError` if the final path escapes reports_dir.
    Side effects:
        None.
    """
    safe_report_name = sanitize_report_name(report_name)
    resolved_reports_dir = Path(reports_dir).resolve()
    output_path = (resolved_reports_dir / f"{safe_report_name}{suffix}").resolve()
    if not output_path.is_relative_to(resolved_reports_dir):
        raise ValueError(f"Report output escaped reports_dir: {output_path}")
    return output_path


def save_niah_verification_report(result, reports_dir, report_name, title, *, project_root=None):
    """Persist a NIAH verification result to JSON and Markdown under reports/.

    中文说明:
    - 调用方 / Called by: CLI `verify-2m`, CLI `benchmark-scale`
    - 调用对象 / Calls: `resolve_niah_reports_dir`, `write_json`, `write_markdown`,
      `build_niah_verification_markdown`, `Path.exists`, `Path.stat`
    - 作用 / Purpose: 固化 2M NIAH 与更大参数 benchmark 的可复现报告文件
    - 参数 / Parameters:
      `result` 是验证结果；`reports_dir` 是输出目录；`report_name` 是文件名前缀；
      `title` 是 Markdown 标题
    - 返回 / Returns: dict，包含 JSON 和 Markdown 路径
    - 接入 / Integration: CLI 完成训练后调用；报告统一放入 reports/
    - 错误处理 / Error handling: 非法 reports_dir 或 report_name 抛 `ValueError`；
      文件写入错误直接向上抛出；写入后文件不存在或为空时抛出 `IOError`
    - 副作用 / Side effects: 写入 `reports/*.json` 和 `reports/*.md`
    - 事务边界 / Transaction boundary: 文件写入非事务性；失败时保留已写文件需人工处理
    - 并发与幂等 / Concurrency and idempotency: 同名报告会覆盖旧文件；不同 report_name 可并行
    - 关键词 / Keywords:
      save_report|json|markdown|reports|niah|2m|benchmark|mhdsra2|persist|保存

    English documentation:
    Function name:
        save_niah_verification_report
    Purpose:
        Persist a NIAH verification result as JSON and Markdown under reports/.
    Called by:
        CLI `verify-2m` and `benchmark-scale`.
    Calls:
        Project report directory resolver, path sanitizer, and writer helpers, followed by checks.
    Parameters:
        Result payload, output directory, report name, and title.
    Returns:
        Dictionary with JSON and Markdown paths.
    Error handling:
        Invalid report directories or report names raise `ValueError`; file write errors propagate
        to the caller; empty or missing output raises `IOError`.
    Side effects:
        Writes JSON and Markdown files.
    Transaction boundary:
        Non-transactional filesystem writes.
    Concurrency and idempotency:
        Same report name overwrites previous files.
    English keywords:
        save_report, json, markdown, reports, niah, 2m, benchmark, mhdsra2, persist, filesystem
    """
    resolved_reports_dir = resolve_niah_reports_dir(reports_dir, project_root=project_root)
    json_path = build_report_output_path(resolved_reports_dir, report_name, ".json")
    markdown_path = build_report_output_path(resolved_reports_dir, report_name, ".md")
    write_json(json_path, result)
    write_markdown(markdown_path, build_niah_verification_markdown(title, result))
    verify_nonempty_report_outputs(json_path, markdown_path)
    return {"json": str(json_path), "markdown": str(markdown_path)}


def verify_nonempty_report_outputs(*output_paths):
    """Validate that report output files exist and are non-empty.

    中文说明:
    - 调用方 / Called by: `save_niah_verification_report`, `save_niah_capacity_reports`
    - 调用对象 / Calls: `Path.exists`, `Path.stat`
    - 作用 / Purpose: 防止 reports/ 写入静默失败后仍返回成功路径，降低报告证据失真风险
    - 参数 / Parameters: `output_paths` 是一个或多个报告文件路径
    - 返回 / Returns: None
    - 内部关键变量 / Internal variables: `output_path` 是当前检查的单个报告路径
    - 接入 / Integration: 新增 NIAH 报告写入函数后应复用该校验
    - 错误处理 / Error handling: 文件缺失或大小为 0 时抛出 `IOError`
    - 副作用 / Side effects: 只读文件元数据，不写文件
    - 事务边界 / Transaction boundary: 不涉及 Unit of Work、数据库或文件写事务
    - 并发与幂等 / Concurrency and idempotency: 对同一文件状态重复调用结果一致
    - 关键词 / Keywords:
      report|verify|nonempty|json|markdown|niah|filesystem|guard|写入|报告

    English documentation:
    Function name:
        verify_nonempty_report_outputs
    Purpose:
        Ensure generated report files exist and are non-empty.
    Called by:
        `save_niah_verification_report` and `save_niah_capacity_reports`.
    Calls:
        `Path.exists` and `Path.stat`.
    Parameters:
        - output_paths: report file paths to validate.
    Returns:
        None.
    Error handling:
        Raises `IOError` when a report is missing or empty.
    Side effects:
        Reads filesystem metadata only.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Deterministic for the same file state.
    English keywords:
        report, verify, nonempty, json, markdown, niah, filesystem, guard, write, output
    """
    for output_path in output_paths:
        resolved_output_path = Path(output_path)
        if not resolved_output_path.exists() or resolved_output_path.stat().st_size <= 0:
            raise IOError(f"NIAH report write verification failed: {resolved_output_path}")


def run_niah_capacity_test(
    seq_lengths=None,
    mode="train_step",
    seed=None,
    *,
    batches_per_depth=DEFAULT_NIAH_CAPACITY_BATCHES_PER_DEPTH,
    cudnn_benchmark=False,
    swanlab_mode: str = "disabled",
):
    """Run the legacy NIAH capacity sweep with traceable RNG and OOM cleanup.

    中文说明:
    - 调用方 / Called by: `scripts.main.run_needle_capacity_reports`
    - 调用对象 / Calls: `resolve_niah_run_seed`, `seed_all`, `run_single_niah_capacity_test`,
      `cleanup_after_oom`, `is_oom_error`
    - 作用 / Purpose: 对多个上下文长度执行 forward/train capacity probe；本函数固定并打印
      本轮 seed，把每个 depth 的 accuracy 建立在多个 batch 上，并只在 OOM 恢复路径清理 CUDA cache
    - 参数 / Parameters:
      `seq_lengths` 是待测长度；`mode` 是 `forward_only/train_step`；`seed` 为可选随机种子；
      `batches_per_depth` 控制 accuracy 样本量；`cudnn_benchmark` 控制 cuDNN 自动调优
    - 返回 / Returns: dict，key 为 seq_len，value 为带 seed、status、accuracy、peak memory 的结果
    - 内部关键变量 / Internal variables: `resolved_seed` 是本轮 capacity sweep 的可追溯种子
    - 接入 / Integration: 供 legacy 主脚本生成 capacity reports；更严格训练验证应使用 CLI 子命令
    - 错误处理 / Error handling: OOM 记录为 `status=oom`；非 OOM RuntimeError/AcceleratorError 向上抛出
    - 副作用 / Side effects: 打印进度、训练小模型、可能清理 OOM 后 CUDA cache；不写文件
    - 事务边界 / Transaction boundary: 不涉及 Unit of Work、数据库或文件事务
    - 并发与幂等 / Concurrency and idempotency: 显式 seed 可复现；默认 time seed 可追溯但不可复现
    - 关键词 / Keywords:
      capacity|seed|oom|batches_per_depth|cuda|benchmark|niah|accuracy|memory|容量

    English documentation:
    Function name:
        run_niah_capacity_test
    Purpose:
        Run the legacy NIAH capacity sweep with traceable seeding and bounded cleanup.
    Called by:
        `scripts.main.run_needle_capacity_reports`.
    Calls:
        Seed resolver, `seed_all`, per-length capacity probe, OOM cleanup, and OOM classifier.
    Parameters:
        Sequence lengths, probe mode, optional seed, per-depth batch count, and cuDNN benchmark flag.
    Returns:
        Mapping from sequence length to structured capacity results.
    Error handling:
        Converts OOM to result rows and propagates non-OOM runtime errors.
    Side effects:
        Prints progress, may train models, and clears CUDA cache only after OOM or final cleanup.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Explicit seed is reproducible; time-derived seed is traceable only.
    English keywords:
        capacity, seed, oom, batches_per_depth, cuda, benchmark, niah, accuracy, memory, probe
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seq_lengths = seq_lengths or CAPACITY_TEST_LENGTHS
    resolved_seed = resolve_niah_run_seed(seed)
    seed_all(resolved_seed, cudnn_benchmark=cudnn_benchmark)
    print(f"--- Running Needle-In-A-Haystack Capacity Test ({mode}) on {device} ---")
    print(
        f"Seed | seed={resolved_seed} | cudnn_benchmark={cudnn_benchmark} | "
        f"batches_per_depth={batches_per_depth}"
    )

    results = {}
    for seq_len in seq_lengths:
        print(f"\n--- Capacity Test | mode={mode} | seq_len={seq_len} ---")
        swanlab_run = init_swanlab(
            project="MHDSRA2",
            experiment_name=f"capacity_{mode}_seq{seq_len}",
            config={"seq_len": seq_len, "mode": mode},
            mode=swanlab_mode,
            tags=["capacity"],
        )
        try:
            result = run_single_niah_capacity_test(
                seq_len=seq_len,
                device=device,
                mode=mode,
                batches_per_depth=batches_per_depth,
            )
            result["seed"] = resolved_seed
            result["cudnn_benchmark"] = cudnn_benchmark
            results[seq_len] = result
            print(
                f"PASS | seq_len={seq_len} | accuracy={result['accuracy']*100:5.1f}% | "
                f"min_depth_accuracy={result.get('min_depth_accuracy', result['accuracy'])*100:5.1f}% | "
                f"peak_mem={result['peak_mem_mb']:.2f} MB"
            )
            if result["status"] == "ok" and swanlab_run.enabled:
                swanlab_run.log({"accuracy": result["accuracy"], "min_depth_accuracy": result.get("min_depth_accuracy", 0), "peak_mem_mb": result.get("peak_mem_mb", 0)})
        except torch.cuda.OutOfMemoryError:
            results[seq_len] = {"status": "oom", "seed": resolved_seed, "cudnn_benchmark": cudnn_benchmark}
            print(f"OOM | seq_len={seq_len}")
            cleanup_after_oom()
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            results[seq_len] = {"status": "oom", "seed": resolved_seed, "cudnn_benchmark": cudnn_benchmark}
            print(f"OOM | seq_len={seq_len}")
            cleanup_after_oom()
        except torch.AcceleratorError as exc:
            if not is_oom_error(exc):
                raise
            results[seq_len] = {"status": "oom", "seed": resolved_seed, "cudnn_benchmark": cudnn_benchmark}
            print(f"OOM | seq_len={seq_len}")
            cleanup_after_oom()
        swanlab_run.finish()

    print(f"\n=== Needle Capacity Results ({mode}) ===")
    for seq_len in seq_lengths:
        result = results[seq_len]
        if result["status"] != "ok":
            print(f"Context {seq_len:>8} | Status: OOM")
        else:
            print(
                f"Context {seq_len:>8} | Status: PASS | "
                f"Accuracy: {result['accuracy']*100:5.1f}% | "
                f"Min Depth Accuracy: {result.get('min_depth_accuracy', result['accuracy'])*100:5.1f}% | "
                f"Peak Mem: {result['peak_mem_mb']:.2f} MB"
            )

    return results


def save_niah_capacity_reports(forward_results, train_results, reports_dir):
    """Persist NIAH capacity sweep reports and verify non-empty outputs.

    中文说明:
    - 调用方 / Called by: `scripts.main.run_needle_capacity_reports`
    - 调用对象 / Calls: `ensure_reports_dir`, `write_json`, `write_markdown`,
      `build_capacity_markdown`, `verify_nonempty_report_outputs`
    - 作用 / Purpose: 将 forward-only 与 train-step capacity 结果写入 reports/，并在写入后
      校验 JSON/Markdown 文件非空，避免容量报告静默损坏导致证据失真
    - 参数 / Parameters:
      `forward_results` 是 forward-only sweep 结果；`train_results` 是 train-step sweep 结果；
      `reports_dir` 是输出目录或项目根目录
    - 返回 / Returns: dict，包含 JSON 和 Markdown 路径
    - 内部关键变量 / Internal variables: `payload` 是写入 JSON 的容量报告结构
    - 接入 / Integration: 主脚本 capacity report 入口调用；报告统一落在 reports/
    - 错误处理 / Error handling: 写入失败或空文件由底层 writer/校验函数向上抛出
    - 副作用 / Side effects: 写入 `needle_capacity_results.json/.md`
    - 事务边界 / Transaction boundary: 文件写入非事务性；失败时保留部分文件需人工处理
    - 并发与幂等 / Concurrency and idempotency: 同名报告会覆盖；不同目录可并行
    - 关键词 / Keywords:
      capacity|report|json|markdown|verify|nonempty|niah|reports|保存|容量

    English documentation:
    Function name:
        save_niah_capacity_reports
    Purpose:
        Persist NIAH capacity results and verify non-empty JSON/Markdown outputs.
    Called by:
        `scripts.main.run_needle_capacity_reports`.
    Calls:
        Report directory helper, JSON/Markdown writers, capacity Markdown builder, and output verifier.
    Parameters:
        - forward_results: forward-only capacity results.
        - train_results: train-step capacity results.
        - reports_dir: output directory or project root.
    Returns:
        Dictionary with JSON and Markdown paths.
    Error handling:
        Propagates write failures and raises `IOError` for missing or empty outputs.
    Side effects:
        Writes capacity report files.
    Transaction boundary:
        Non-transactional filesystem writes.
    Concurrency and idempotency:
        Same output names overwrite previous files.
    English keywords:
        capacity, report, json, markdown, verify, nonempty, niah, reports, save, output
    """
    reports_dir = ensure_reports_dir(reports_dir)
    payload = {
        "forward_only": forward_results,
        "train_step": train_results,
    }
    json_path = reports_dir / "needle_capacity_results.json"
    markdown_path = reports_dir / "needle_capacity_results.md"
    write_json(json_path, payload)
    write_markdown(
        markdown_path,
        build_capacity_markdown(forward_results, train_results),
    )
    verify_nonempty_report_outputs(json_path, markdown_path)
    return {"json": str(json_path), "markdown": str(markdown_path)}


def run_niah_test(
    seq_lengths=None,
    model_type="mhdsra2",
    seed=None,
    *,
    eval_batches_per_depth=DEFAULT_NIAH_EVAL_BATCHES_PER_DEPTH,
    cudnn_benchmark=False,
    swanlab_mode: str = "disabled",
):
    """Run the legacy NIAH long-context sweep with final-eval reporting.

    中文说明:
    - 调用方 / Called by: `main` legacy fallback, `scripts.main.run_needle_in_haystack`
    - 调用对象 / Calls: `resolve_niah_run_seed`, `seed_all`, `run_single_niah_test`,
      `cleanup_after_oom`, `is_oom_error`
    - 作用 / Purpose: 对多个上下文长度运行 NIAH sweep；本函数固定并打印 seed，返回结构化
      final/best eval 指标，并在汇总输出中以 final eval mean accuracy 作为主报告口径
    - 参数 / Parameters:
      `seq_lengths` 是待测长度；`model_type` 是模型名或归档别名；`seed` 为可选随机种子；
      `eval_batches_per_depth` 控制 eval 样本量；`cudnn_benchmark` 控制 cuDNN 自动调优
    - 返回 / Returns: dict，包含 seed、配置和每个 seq_len 的结构化指标或 OOM 状态
    - 内部关键变量 / Internal variables: `metrics` 是单长度 final/best/train 指标集合
    - 接入 / Integration: 保留 legacy sweep 入口；可复现报告优先使用 `verify-2m` CLI
    - 错误处理 / Error handling: OOM 记录为 `status=oom`；非 OOM RuntimeError/AcceleratorError 向上抛出
    - 副作用 / Side effects: 打印进度、训练模型、OOM 后清理 CUDA cache；不写文件
    - 事务边界 / Transaction boundary: 不涉及 Unit of Work、数据库或文件事务
    - 并发与幂等 / Concurrency and idempotency: 显式 seed 可复现；默认 time seed 可追溯但不可复现
    - 关键词 / Keywords:
      niah|sweep|final_eval|best_eval|seed|oom|benchmark|accuracy|legacy|长上下文

    English documentation:
    Function name:
        run_niah_test
    Purpose:
        Run the legacy long-context NIAH sweep using final eval metrics as the primary result.
    Called by:
        Legacy CLI fallback and `scripts.main.run_needle_in_haystack`.
    Calls:
        Seed resolver, `seed_all`, per-length NIAH trainer, OOM cleanup, and OOM classifier.
    Parameters:
        Sequence lengths, model type, optional seed, eval batches per depth, and cuDNN benchmark flag.
    Returns:
        Structured dictionary with seed metadata and per-length metrics or OOM status.
    Error handling:
        Converts OOM to result rows and propagates non-OOM runtime errors.
    Side effects:
        Prints progress, trains models, and clears CUDA cache only on OOM recovery.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Explicit seed is reproducible; time-derived seed is traceable only.
    English keywords:
        niah, sweep, final_eval, best_eval, seed, oom, benchmark, accuracy, legacy, long_context
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seq_lengths = seq_lengths or DEFAULT_SEQ_LENGTHS
    resolved_seed = resolve_niah_run_seed(seed)
    seed_all(resolved_seed, cudnn_benchmark=cudnn_benchmark)
    print(
        f"--- Running Needle-In-A-Haystack Long-Context Sweep on {device} "
        f"| model_type={model_type} ---"
    )
    print(
        f"Seed | seed={resolved_seed} | cudnn_benchmark={cudnn_benchmark} | "
        f"eval_batches_per_depth={eval_batches_per_depth}"
    )

    vocab_size = 100
    dim = 64
    num_layers = 2
    K = 64
    kr = 8

    results = {}
    for seq_len in seq_lengths:
        swanlab_run = init_swanlab(
            project="MHDSRA2",
            experiment_name=f"niah_d{dim}_l{num_layers}_s{K}_kr{kr}_seq{seq_len}",
            config={"seq_len": seq_len, "dim": dim, "num_layers": num_layers, "slots": K, "read_topk": kr, "model_type": model_type},
            mode=swanlab_mode,
            tags=["niah", "sweep"],
        )
        try:
            metrics = run_single_niah_test(
                seq_len=seq_len,
                device=device,
                vocab_size=vocab_size,
                dim=dim,
                num_layers=num_layers,
                K=K,
                kr=kr,
                model_type=model_type,
                eval_batches_per_depth=eval_batches_per_depth,
                return_metrics=True,
                swanlab_run=swanlab_run,
            )
            metrics["seed"] = resolved_seed
            metrics["cudnn_benchmark"] = cudnn_benchmark
            results[seq_len] = metrics
        except torch.cuda.OutOfMemoryError:
            results[seq_len] = {"status": "oom", "seed": resolved_seed, "cudnn_benchmark": cudnn_benchmark}
            print(f"\nOOM! Skipping Needle-In-A-Haystack at seq_len={seq_len}.")
            cleanup_after_oom()
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            results[seq_len] = {"status": "oom", "seed": resolved_seed, "cudnn_benchmark": cudnn_benchmark}
            print(f"\nOOM! Skipping Needle-In-A-Haystack at seq_len={seq_len}.")
            cleanup_after_oom()
        except torch.AcceleratorError as exc:
            if not is_oom_error(exc):
                raise
            results[seq_len] = {"status": "oom", "seed": resolved_seed, "cudnn_benchmark": cudnn_benchmark}
            print(f"\nOOM! Skipping Needle-In-A-Haystack at seq_len={seq_len}.")
            cleanup_after_oom()
        swanlab_run.finish()

    print("\n=== Needle-In-A-Haystack Sweep Results ===")
    for seq_len in seq_lengths:
        result = results[seq_len]
        if result["status"] == "oom":
            print(f"Context {seq_len:>8} | Final Eval Accuracy: OOM")
        else:
            print(
                f"Context {seq_len:>8} | "
                f"Final Eval Accuracy: {result['final_eval_mean_accuracy']*100:5.1f}% | "
                f"Best Eval Accuracy: {result['best_eval_mean_accuracy']*100:5.1f}%"
            )

    return {
        "seed": resolved_seed,
        "model_type": model_type,
        "cudnn_benchmark": cudnn_benchmark,
        "eval_batches_per_depth": eval_batches_per_depth,
        "results": results,
    }


def build_parser():
    """Build the NIAH command-line parser.

    中文说明:
    - 调用方 / Called by: `main`, tests/manual CLI usage
    - 调用对象 / Calls: `argparse.ArgumentParser`
    - 作用 / Purpose: 将 2M NIAH 验证和更大参数 benchmark 固化为可复现 CLI
    - 参数 / Parameters: 无
    - 返回 / Returns: `argparse.ArgumentParser`
    - 接入 / Integration: 运行 `python scripts/needle_in_haystack_test.py verify-2m ...`
    - 错误处理 / Error handling: argparse 负责非法参数报错和退出码
    - 副作用 / Side effects: 无
    - 事务边界 / Transaction boundary: 不涉及事务
    - 并发与幂等 / Concurrency and idempotency: 纯构造函数
    - 关键词 / Keywords:
      cli|argparse|verify_2m|benchmark_scale|reports|niah|mhdsra2|reproducible|parser|命令行

    English documentation:
    Function name:
        build_parser
    Purpose:
        Build the NIAH CLI parser for reproducible verification and benchmark reports.
    Called by:
        `main` and tests/manual CLI usage.
    Calls:
        `argparse.ArgumentParser`.
    Parameters:
        None.
    Returns:
        Argument parser instance.
    Error handling:
        argparse handles invalid arguments.
    Side effects:
        None.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Pure constructor.
    English keywords:
        cli, argparse, verify_2m, benchmark_scale, reports, niah, mhdsra2, reproducible, parser, command
    """
    parser = argparse.ArgumentParser(description="Run reproducible MHDSRA2 NIAH verification.")
    subparsers = parser.add_subparsers(dest="command")

    def add_common_options(command_parser):
        command_parser.add_argument("--seq-len", type=int, default=2_097_152)
        command_parser.add_argument("--device", default="auto")
        command_parser.add_argument("--vocab-size", type=int, default=100)
        command_parser.add_argument(
            "--data-vocab-size",
            type=int,
            default=None,
            help="Vocabulary size for data generation (default: same as --vocab-size). "
                 "Use this for vocab curriculum learning: model always uses 100, "
                 "but data grows from small to large.",
        )
        command_parser.add_argument("--dim", type=int, default=64)
        command_parser.add_argument("--num-layers", type=int, default=2)
        command_parser.add_argument("--slots", type=int, default=64)
        command_parser.add_argument("--read-topk", type=int, default=8)
        command_parser.add_argument("--chunk-size", type=int, default=1024)
        command_parser.add_argument("--batch-size", type=int, default=1)
        command_parser.add_argument("--epochs", type=int, default=60)
        command_parser.add_argument("--learning-rate", type=float, default=1e-3)
        command_parser.add_argument("--seed", type=int, default=20260506)
        command_parser.add_argument("--target-accuracy", type=float, default=1.0)
        command_parser.add_argument("--stop-loss", type=float, default=0.1)
        command_parser.add_argument(
            "--log-interval",
            type=int,
            default=20,
            help=(
                "Compatibility interval used as the default robust-eval interval; "
                "SwanLab light metrics are logged every optimizer step."
            ),
        )
        command_parser.add_argument(
            "--eval-batches-per-depth",
            type=int,
            default=DEFAULT_NIAH_LIGHT_EVAL_BATCHES_PER_DEPTH,
            help="Light eval batches per depth, executed and uploaded every optimizer step.",
        )
        command_parser.add_argument(
            "--robust-eval-interval",
            type=int,
            default=None,
            help="Optimizer-step interval for robust eval; defaults to --log-interval.",
        )
        command_parser.add_argument(
            "--robust-eval-batches-per-depth",
            type=int,
            default=DEFAULT_NIAH_ROBUST_EVAL_BATCHES_PER_DEPTH,
            help="Robust eval batches per depth, used for report conclusions and early stop.",
        )
        command_parser.add_argument("--cudnn-benchmark", action="store_true")
        command_parser.add_argument("--swanlab-mode", default="disabled", choices=["cloud", "local", "offline", "disabled"])
        command_parser.add_argument("--reports-dir", default="docs/reports")
        command_parser.add_argument("--report-name", default=None)
        command_parser.add_argument(
            "--mhdsra2-config",
            type=str,
            default=None,
            help='JSON string to override MHDSRA2 config, e.g. \'{"detach_state": false}\'',
        )
        command_parser.add_argument(
            "--detach-state",
            type=str,
            default=None,
            choices=["true", "false"],
            help="Override detach_state in MHDSRA2Config (True/False).",
        )
        command_parser.add_argument(
            "--needle-loss-alpha",
            type=float,
            default=0.5,
            help="Weight for auxiliary needle-value prediction loss (0.0 to disable).",
        )
        command_parser.add_argument(
            "--hidden-mse-alpha",
            type=float,
            default=0.0,
            help="Weight for hidden-state MSE auxiliary loss (0.0 to disable). "
                 "Aligns query hidden state with target token embedding.",
        )
        command_parser.add_argument(
            "--retrieval-projection-contrastive-alpha",
            type=float,
            default=0.0,
            help=(
                "Weight for opt-in retrieval q/k projection contrastive loss "
                "(0.0 to disable)."
            ),
        )
        command_parser.add_argument(
            "--retrieval-projection-temperature",
            type=float,
            default=DEFAULT_RETRIEVAL_PROJECTION_CONTRASTIVE_TEMPERATURE,
            help="Positive softmax temperature for retrieval projection contrastive loss.",
        )
        command_parser.add_argument(
            "--retrieval-span-predictor-alpha",
            type=float,
            default=0.0,
            help=(
                "Weight for opt-in retrieval candidate span predictor loss "
                "(0.0 to disable)."
            ),
        )
        command_parser.add_argument(
            "--niah-readout-mode",
            type=str,
            default=DEFAULT_NIAH_READOUT_MODE,
            choices=ALLOWED_NIAH_READOUT_MODES,
            help="Evaluation readout mode; default keeps query-logit model evaluation.",
        )
        command_parser.add_argument(
            "--niah-span-candidate-filter",
            type=str,
            default=DEFAULT_NIAH_SPAN_CANDIDATE_FILTER,
            choices=ALLOWED_NIAH_SPAN_CANDIDATE_FILTERS,
            help=(
                "Candidate filter for span_predictor readout/loss; default keeps "
                "all selected retrieval candidates."
            ),
        )
        command_parser.add_argument(
            "--niah-span-loss-mode",
            type=str,
            default=DEFAULT_NIAH_SPAN_LOSS_MODE,
            choices=ALLOWED_NIAH_SPAN_LOSS_MODES,
            help=(
                "Training loss for retrieval span predictor; default keeps the "
                "single-positive cross entropy behavior."
            ),
        )
        command_parser.add_argument(
            "--load-checkpoint",
            type=str,
            default=None,
            help=(
                "Load model state_dict from reports/checkpoints/. "
                "Plain filenames are resolved inside that directory."
            ),
        )
        command_parser.add_argument(
            "--save-checkpoint",
            type=str,
            default=None,
            help=(
                "Save best model state_dict under reports/checkpoints/. "
                "Plain filenames are resolved inside that directory."
            ),
        )

    verify_parser = subparsers.add_parser(
        "verify-2m",
        help="Run the standard 2M NIAH verification and write reports.",
    )
    add_common_options(verify_parser)

    benchmark_parser = subparsers.add_parser(
        "benchmark-scale",
        help="Run a larger-parameter MHDSRA2 NIAH benchmark and write reports.",
    )
    add_common_options(benchmark_parser)
    benchmark_parser.set_defaults(
        dim=128,
        num_layers=4,
        slots=128,
        read_topk=16,
        epochs=8,
        stop_loss=0.0,
        report_name="mhdsra2_niah_2m_scale_benchmark",
    )
    return parser


def run_cli_command(args):
    """Run one parsed NIAH CLI command and persist its report.

    中文说明:
    - 调用方 / Called by: `main`
    - 调用对象 / Calls: `resolve_device`, `run_niah_verification_case`,
      `save_niah_verification_report`
    - 作用 / Purpose: 将 CLI 参数转换为可复现实验、执行训练并写入 reports/ 交付物
    - 参数 / Parameters: `args` 是 argparse 解析结果
    - 返回 / Returns: 验证结果 dict
    - 接入 / Integration: `verify-2m` 和 `benchmark-scale` 子命令共用本函数
    - 错误处理 / Error handling: 训练、CUDA 或文件写入错误直接向上抛出，保留真实失败
    - 副作用 / Side effects: 训练模型、打印结果、写入报告文件
    - 事务边界 / Transaction boundary: 文件写入非事务性；不涉及数据库事务
    - 并发与幂等 / Concurrency and idempotency: 同名报告会覆盖；固定 seed 可复现
    - 关键词 / Keywords:
      cli|run|reports|verify_2m|benchmark_scale|niah|mhdsra2|json|markdown|执行

    English documentation:
    Function name:
        run_cli_command
    Purpose:
        Execute a parsed NIAH CLI command and persist JSON/Markdown reports.
    Called by:
        `main`.
    Calls:
        Device resolver, verification runner, and report saver.
    Parameters:
        - args: argparse namespace.
    Returns:
        Verification result dictionary.
    Error handling:
        Propagates runtime and file errors.
    Side effects:
        Trains a model, prints status, and writes report files.
    Transaction boundary:
        Non-transactional report writes only.
    Concurrency and idempotency:
        Same report name overwrites previous reports.
    English keywords:
        cli, run, reports, verify_2m, benchmark_scale, niah, mhdsra2, json, markdown, execute
    """
    device = resolve_device(args.device)
    report_name = args.report_name
    if report_name is None:
        report_name = "mhdsra2_niah_2m_verification"
    title = (
        "MHDSRA2 2M NIAH Verification"
        if args.command == "verify-2m"
        else "MHDSRA2 2M Larger-Scale NIAH Benchmark"
    )
    mhdsra2_config_override = None
    if args.mhdsra2_config is not None:
        try:
            mhdsra2_config_override = json.loads(args.mhdsra2_config)
            print(f"MHDSRA2 config override: {mhdsra2_config_override}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid --mhdsra2-config JSON: {e}")
    if args.detach_state is not None:
        if mhdsra2_config_override is None:
            mhdsra2_config_override = {}
        mhdsra2_config_override["detach_state"] = args.detach_state == "true"
        print(f"Overriding detach_state = {mhdsra2_config_override['detach_state']}")
    result = run_niah_verification_case(
        seq_len=args.seq_len,
        device=device,
        vocab_size=args.vocab_size,
        data_vocab_size=args.data_vocab_size,
        dim=args.dim,
        num_layers=args.num_layers,
        K=args.slots,
        kr=args.read_topk,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        target_accuracy=args.target_accuracy,
        stop_loss=args.stop_loss,
        log_interval=args.log_interval,
        eval_batches_per_depth=args.eval_batches_per_depth,
        robust_eval_interval=args.robust_eval_interval,
        robust_eval_batches_per_depth=args.robust_eval_batches_per_depth,
        cudnn_benchmark=args.cudnn_benchmark,
        swanlab_mode=args.swanlab_mode,
        mhdsra2_config_override=mhdsra2_config_override,
        load_checkpoint=args.load_checkpoint,
        save_checkpoint=args.save_checkpoint,
        needle_loss_alpha=args.needle_loss_alpha,
        hidden_mse_alpha=args.hidden_mse_alpha,
        retrieval_projection_contrastive_alpha=args.retrieval_projection_contrastive_alpha,
        retrieval_projection_temperature=args.retrieval_projection_temperature,
        retrieval_span_predictor_alpha=args.retrieval_span_predictor_alpha,
        use_retrieval=args.retrieval_span_predictor_alpha > 0.0
        or args.retrieval_projection_contrastive_alpha > 0.0
        or args.niah_readout_mode != "model",
        niah_readout_mode=args.niah_readout_mode,
        niah_span_candidate_filter=args.niah_span_candidate_filter,
        niah_span_loss_mode=args.niah_span_loss_mode,
    )
    paths = save_niah_verification_report(result, args.reports_dir, report_name, title)
    result["report_paths"] = paths
    print("NIAH_REPORT_RESULT=" + json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main(argv=None):
    """NIAH CLI entrypoint.

    中文说明:
    - 调用方 / Called by: `if __name__ == "__main__"` and command-line users
    - 调用对象 / Calls: `build_parser`, `run_cli_command`, legacy `run_niah_test`
    - 作用 / Purpose: 提供默认 sweep 兼容行为和新的可复现 2M 报告 CLI
    - 参数 / Parameters: `argv` 是可选参数列表；None 时使用 `sys.argv`
    - 返回 / Returns: 命令结果或 legacy sweep 结果
    - 接入 / Integration: 推荐使用 `verify-2m` 或 `benchmark-scale` 子命令
    - 错误处理 / Error handling: argparse 或运行错误向上抛出，不伪造成功
    - 副作用 / Side effects: 可能训练模型并写入 reports/
    - 事务边界 / Transaction boundary: 不涉及数据库事务
    - 并发与幂等 / Concurrency and idempotency: 取决于子命令和 report_name
    - 关键词 / Keywords:
      main|entrypoint|cli|legacy|sweep|verify_2m|benchmark|reports|niah|入口

    English documentation:
    Function name:
        main
    Purpose:
        Provide the NIAH command-line entrypoint.
    Called by:
        `if __name__ == "__main__"` and command-line users.
    Calls:
        Parser builder, CLI command runner, and legacy sweep fallback.
    Parameters:
        - argv: optional CLI argument list.
    Returns:
        Command result or legacy sweep result.
    Error handling:
        Propagates argparse/runtime errors.
    Side effects:
        May train models and write reports.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Depends on subcommand and report name.
    English keywords:
        main, entrypoint, cli, legacy, sweep, verify_2m, benchmark, reports, niah, command
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        return run_niah_test()
    return run_cli_command(args)


if __name__ == '__main__':
    main()
