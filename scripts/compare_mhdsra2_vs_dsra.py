"""Compare MHDSRA2 against baseline DSRA under the same random-token workload.

This module keeps the comparison lightweight so it can run inside tests while
still producing stable report artifacts in the project `reports/` directory.
"""

from __future__ import annotations

import argparse
import itertools
import statistics
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.dsra.dsra_layer import DSRA_Chunk_Layer
from src.dsra.mhdsra2.improved_dsra_mha import MHDSRA2Config, MultiHeadDSRA2
from src.dsra.report_utils import ensure_reports_dir, write_json, write_markdown


def _tensor_bytes(tensor: torch.Tensor | None) -> int:
    """Return tensor memory footprint in bytes.

    中文说明:
    - 调用方 / Called by: `_estimate_dsra_state_bytes`, `_estimate_mhdsra2_state_bytes`
    - 调用对象 / Calls: `torch.Tensor.element_size`, `torch.Tensor.numel`
    - 作用 / Purpose: 统一计算张量占用字节数，便于两个实现的状态内存对比
    - 变量 / Variables: `tensor` 待统计张量，允许为 `None`
    - 接入 / Integration: 新增状态字段时，可直接复用本函数累计字节数
    - 错误处理 / Error handling: 输入为 `None` 时返回 `0`，避免空状态分支报错
    - 关键词 / Keywords:
      tensor_bytes|memory|bytes|state|footprint|torch|compare|utility|size|统计
    """
    if tensor is None:
        return 0
    return tensor.element_size() * tensor.numel()


def _estimate_dsra_state_bytes(state: torch.Tensor | None, bypass_kv) -> int:
    """Estimate baseline DSRA streaming state memory in bytes.

    中文说明:
    - 调用方 / Called by: `run_dsra_baseline_once`
    - 调用对象 / Calls: `_tensor_bytes`
    - 作用 / Purpose: 统计 DSRA 的槽状态与局部 bypass KV 缓存开销
    - 变量 / Variables: `state` 为 `S_prev`，`bypass_kv` 为 `(K_cache, V_cache)`
    - 接入 / Integration: 若 DSRA 后续新增流式状态，可在此函数扩展统计项
    - 错误处理 / Error handling: 空状态或空缓存按 `0` 处理
    - 关键词 / Keywords:
      dsra|state_bytes|bypass|cache|memory|slots|streaming|baseline|measure|统计
    """
    total = _tensor_bytes(state)
    if bypass_kv is None:
        return total
    return total + _tensor_bytes(bypass_kv[0]) + _tensor_bytes(bypass_kv[1])


def _estimate_mhdsra2_state_bytes(state) -> int:
    """Estimate MHDSRA2 streaming state memory in bytes.

    中文说明:
    - 调用方 / Called by: `run_mhdsra2_once`
    - 调用对象 / Calls: `_tensor_bytes`
    - 作用 / Purpose: 统计 MHDSRA2 slot/local/state metadata 的内存占用
    - 变量 / Variables: `state` 为 `MHDSRA2State`
    - 接入 / Integration: 若 MHDSRA2 新增状态字段，可继续在此追加统计
    - 错误处理 / Error handling: 各字段为空时自动按 `0` 处理
    - 关键词 / Keywords:
      mhdsra2|state_bytes|local_cache|slot|memory|streaming|measure|compare|usage|统计
    """
    return sum(
        [
            _tensor_bytes(state.slot_k),
            _tensor_bytes(state.slot_v),
            _tensor_bytes(state.age),
            _tensor_bytes(state.usage),
            _tensor_bytes(state.confidence),
            _tensor_bytes(state.local_k),
            _tensor_bytes(state.local_v),
        ]
    )


def _synchronize_device(device: torch.device) -> None:
    """Synchronize device before or after GPU timing windows.

    中文说明:
    - 调用方 / Called by: `_measure_repeated_passes`
    - 调用对象 / Calls: `torch.cuda.synchronize`
    - 作用 / Purpose: 在 CUDA 场景中显式同步，避免异步 kernel 导致计时偏小
    - 变量 / Variables: `device` 为当前执行设备
    - 接入 / Integration: 其他 GPU 基准脚本也可复用本函数保证计时一致性
    - 错误处理 / Error handling: CPU 或非 CUDA 设备直接跳过，不抛出异常
    - 关键词 / Keywords:
      synchronize|cuda|timing|gpu|benchmark|latency|device|async|measure|同步
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summarize_samples(samples: list[float]) -> tuple[float, float]:
    """Return mean and standard deviation for timing samples.

    中文说明:
    - 调用方 / Called by: `_measure_repeated_passes`
    - 调用对象 / Calls: `statistics.fmean`, `statistics.stdev`
    - 作用 / Purpose: 统一把重复测量结果转换为均值和标准差
    - 变量 / Variables: `samples` 为毫秒级重复计时结果列表
    - 接入 / Integration: 后续新增吞吐或显存采样统计时可沿用同类模式
    - 错误处理 / Error handling: 单样本标准差返回 `0.0`，避免 `statistics.stdev` 抛错
    - 关键词 / Keywords:
      samples|mean|std|statistics|timing|repeat|latency|benchmark|aggregate|标准差
    """
    if not samples:
        return 0.0, 0.0
    mean_value = float(statistics.fmean(samples))
    std_value = float(statistics.stdev(samples)) if len(samples) > 1 else 0.0
    return mean_value, std_value


def _measure_repeated_passes(
    run_once,
    warmup_runs: int,
    repeat_runs: int,
    device: torch.device,
) -> tuple[torch.Tensor, int, dict | None, list[float]]:
    """Measure one runner with warmup, repeat, mean and std semantics.

    中文说明:
    - 调用方 / Called by: `run_dsra_baseline_once`, `run_mhdsra2_once`
    - 调用对象 / Calls: `_synchronize_device`, `time.perf_counter`
    - 作用 / Purpose: 对单模型前向执行 warmup 与重复测量，输出最终结果和全部样本
    - 变量 / Variables:
      `run_once` 单次完整前向闭包, `warmup_runs` 预热次数, `repeat_runs` 正式测量次数, `device` 执行设备
    - 接入 / Integration: 新增第三种模型时只需提供单次前向闭包即可复用计时逻辑
    - 错误处理 / Error handling: `repeat_runs` 至少按 `1` 执行；底层前向异常直接向上抛出
    - 关键词 / Keywords:
      warmup|repeat|timing|gpu|benchmark|latency|std|mean|runner|计时
    """
    effective_repeats = max(1, repeat_runs)
    last_output = None
    last_state_bytes = 0
    last_aux = None
    for _ in range(max(0, warmup_runs)):
        last_output, last_state_bytes, last_aux = run_once()
        _synchronize_device(device)

    elapsed_samples = []
    for _ in range(effective_repeats):
        _synchronize_device(device)
        start_time = time.perf_counter()
        last_output, last_state_bytes, last_aux = run_once()
        _synchronize_device(device)
        elapsed_samples.append((time.perf_counter() - start_time) * 1000.0)

    return last_output, last_state_bytes, last_aux, elapsed_samples


def _resolve_device(device_name: str) -> torch.device:
    """Resolve CLI device setting to an executable torch device.

    中文说明:
    - 调用方 / Called by: `run_comparison`
    - 调用对象 / Calls: `torch.cuda.is_available`, `torch.device`
    - 作用 / Purpose: 支持 `auto/cpu/cuda` 三类入口，并优先选择可用 GPU
    - 变量 / Variables: `device_name` 为命令行输入设备字符串
    - 接入 / Integration: 主入口或其他脚本若想复用自动设备选择，可直接调用本函数
    - 错误处理 / Error handling: `auto` 在无 CUDA 时自动回退到 `cpu`
    - 关键词 / Keywords:
      device|auto|cuda|cpu|resolve|runtime|torch|gpu|fallback|设备
    """
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def run_dsra_baseline_once(
    x: torch.Tensor,
    chunk_size: int,
    slots: int,
    read_topk: int,
    device: torch.device,
    warmup_runs: int,
    repeat_runs: int,
) -> dict:
    """Run baseline DSRA on one shared random workload and collect metrics.

    中文说明:
    - 调用方 / Called by: `run_comparison`
    - 调用对象 / Calls:
      `DSRA_Chunk_Layer.forward`, `_estimate_dsra_state_bytes`, `_measure_repeated_passes`
    - 作用 / Purpose: 对原始 DSRA 执行同输入分块前向，输出稳定计时均值、标准差与状态大小
    - 变量 / Variables:
      `x` 共享输入, `chunk_size` 分块长度, `slots` 槽位数, `read_topk` 稀疏读取 top-k,
      `warmup_runs/repeat_runs` 为预热与正式测量次数
    - 接入 / Integration: 作为 MHDSRA2 对比基线，供 CLI 和测试复用
    - 错误处理 / Error handling: 依赖张量维度一致性；若底层前向异常则向上抛出
    - 关键词 / Keywords:
      dsra|baseline|forward|chunk|latency|memory|shape|state|benchmark|稳定计时
    """
    layer = DSRA_Chunk_Layer(
        dim=x.shape[-1],
        K=slots,
        kr=min(read_topk, slots),
        use_orthogonal_update=True,
        use_bypass=True,
    ).to(device)
    layer.eval()

    def _run_once():
        state = None
        bypass_kv = None
        max_state_bytes = 0
        outputs = []
        with torch.no_grad():
            chunk_idx = 0
            for start in range(0, x.shape[1], chunk_size):
                chunk = x[:, start : start + chunk_size, :]
                out_chunk, state, bypass_kv, _ = layer(
                    chunk, S_prev=state, bypass_kv=bypass_kv, chunk_idx=chunk_idx
                )
                outputs.append(out_chunk)
                max_state_bytes = max(
                    max_state_bytes,
                    _estimate_dsra_state_bytes(state, bypass_kv),
                )
                chunk_idx += 1
        return torch.cat(outputs, dim=1), max_state_bytes, None

    output, max_state_bytes, _, elapsed_samples = _measure_repeated_passes(
        _run_once,
        warmup_runs=warmup_runs,
        repeat_runs=repeat_runs,
        device=device,
    )
    elapsed_ms, elapsed_std_ms = _summarize_samples(elapsed_samples)
    return {
        "model": "dsra",
        "elapsed_ms": elapsed_ms,
        "elapsed_ms_std": elapsed_std_ms,
        "elapsed_ms_samples": elapsed_samples,
        "warmup_runs": warmup_runs,
        "repeat_runs": max(1, repeat_runs),
        "tokens_per_second": (x.shape[0] * x.shape[1]) / max(elapsed_ms / 1000.0, 1e-9),
        "max_state_bytes": max_state_bytes,
        "output_shape": list(output.shape),
        "output_mean_abs": float(output.abs().mean().item()),
        "is_finite": bool(torch.isfinite(output).all().item()),
    }


def run_mhdsra2_once(
    x: torch.Tensor,
    chunk_size: int,
    slots: int,
    read_topk: int,
    device: torch.device,
    warmup_runs: int,
    repeat_runs: int,
    use_retrieval: bool,
    retrieval_tokens: int,
) -> dict:
    """Run MHDSRA2 on one shared random workload and collect metrics.

    中文说明:
    - 调用方 / Called by: `run_comparison`
    - 调用对象 / Calls:
      `MultiHeadDSRA2.forward`, `_estimate_mhdsra2_state_bytes`, `_measure_repeated_passes`
    - 作用 / Purpose: 对 MHDSRA2 执行同输入分块前向，输出稳定计时均值、标准差与状态大小
    - 变量 / Variables:
      `x` 共享输入, `chunk_size` 分块长度, `slots` 槽位数, `read_topk` 读取 top-k,
      `warmup_runs/repeat_runs` 为预热与正式测量次数,
      `use_retrieval` 控制是否启用检索分支, `retrieval_tokens` 为每次读回 token 数
    - 接入 / Integration: 作为对比方案主模型，供 CLI 和测试复用
    - 错误处理 / Error handling: 依赖配置合法性；若维度不整除等问题则由配置抛错
    - 关键词 / Keywords:
      mhdsra2|forward|chunk|latency|memory|shape|state|benchmark|multihead|稳定计时
    """
    heads = 4
    cfg = MHDSRA2Config(
        dim=x.shape[-1],
        heads=heads,
        slots=slots,
        read_topk=min(read_topk, slots),
        write_topk=max(1, min(2, slots)),
        local_window=chunk_size,
        use_local=True,
        use_retrieval=use_retrieval,
        detach_state=True,
    )
    layer = MultiHeadDSRA2(cfg).to(device)
    layer.eval()
    retrieved_k = None
    retrieved_v = None
    if use_retrieval and retrieval_tokens > 0:
        head_dim = cfg.dim // cfg.heads
        retrieved_k = torch.randn(
            x.shape[0],
            cfg.heads,
            retrieval_tokens,
            head_dim,
            device=device,
        )
        retrieved_v = torch.randn(
            x.shape[0],
            cfg.heads,
            retrieval_tokens,
            head_dim,
            device=device,
        )

    def _run_once():
        state = None
        max_state_bytes = 0
        outputs = []
        gates_mean = None
        with torch.no_grad():
            for start in range(0, x.shape[1], chunk_size):
                chunk = x[:, start : start + chunk_size, :]
                out_chunk, state, aux = layer(
                    chunk,
                    state=state,
                    retrieved_k=retrieved_k,
                    retrieved_v=retrieved_v,
                    return_aux=True,
                )
                outputs.append(out_chunk)
                gates_mean = aux["gates_mean"]
                max_state_bytes = max(max_state_bytes, _estimate_mhdsra2_state_bytes(state))
        return torch.cat(outputs, dim=1), max_state_bytes, {"gates_mean": gates_mean}

    output, max_state_bytes, aux_payload, elapsed_samples = _measure_repeated_passes(
        _run_once,
        warmup_runs=warmup_runs,
        repeat_runs=repeat_runs,
        device=device,
    )
    elapsed_ms, elapsed_std_ms = _summarize_samples(elapsed_samples)
    gates_mean = None if aux_payload is None else aux_payload["gates_mean"]
    return {
        "model": "mhdsra2",
        "elapsed_ms": elapsed_ms,
        "elapsed_ms_std": elapsed_std_ms,
        "elapsed_ms_samples": elapsed_samples,
        "warmup_runs": warmup_runs,
        "repeat_runs": max(1, repeat_runs),
        "tokens_per_second": (x.shape[0] * x.shape[1]) / max(elapsed_ms / 1000.0, 1e-9),
        "max_state_bytes": max_state_bytes,
        "output_shape": list(output.shape),
        "output_mean_abs": float(output.abs().mean().item()),
        "is_finite": bool(torch.isfinite(output).all().item()),
        "use_retrieval": use_retrieval,
        "retrieval_tokens": retrieval_tokens if use_retrieval else 0,
        "gates_mean": gates_mean.detach().cpu().tolist() if gates_mean is not None else None,
    }


def run_comparison(args: argparse.Namespace) -> dict:
    """Execute DSRA vs MHDSRA2 comparison with shared random seeds and shapes.

    中文说明:
    - 调用方 / Called by: `main`, `tests.test_mhdsra2_vs_dsra_compare`
    - 调用对象 / Calls:
      `run_dsra_baseline_once`, `run_mhdsra2_once`, `torch.manual_seed`, `_resolve_device`
    - 作用 / Purpose: 统一组织多参数网格对比，生成结构化结果载荷
    - 变量 / Variables:
      `args` CLI 参数集合，控制 batch、dim、slots/read_topk/chunk_size、warmup/repeat、
      retrieval 与序列长度
    - 接入 / Integration: 新增更多模型时，可在本函数中继续扩展比较矩阵
    - 错误处理 / Error handling: 统一使用确定性种子；若单轮失败则直接抛出，便于测试快速失败
    - 关键词 / Keywords:
      comparison|grid|shared_input|seed|payload|benchmark|dsra|mhdsra2|reports|稳定计时
    """
    device = _resolve_device(args.device)
    results = []
    for batch_size, dim, seq_len, slots, read_topk, chunk_size in itertools.product(
        args.batch_size, args.dim, args.seq_lengths, args.slots, args.read_topk, args.chunk_sizes
    ):
        torch.manual_seed(
            args.seed + batch_size + dim + seq_len + slots + read_topk + chunk_size
        )
        x = torch.randn(batch_size, seq_len, dim, device=device)
        dsra_metrics = run_dsra_baseline_once(
            x,
            chunk_size,
            slots,
            read_topk,
            device,
            args.warmup_runs,
            args.repeat_runs,
        )
        mhdsra2_metrics = run_mhdsra2_once(
            x,
            chunk_size,
            slots,
            read_topk,
            device,
            args.warmup_runs,
            args.repeat_runs,
            args.use_retrieval,
            args.retrieval_tokens,
        )
        results.append(
            {
                "batch_size": batch_size,
                "dim": dim,
                "seq_len": seq_len,
                "slots": slots,
                "read_topk": read_topk,
                "chunk_size": chunk_size,
                "use_retrieval": bool(args.use_retrieval),
                "retrieval_tokens": args.retrieval_tokens if args.use_retrieval else 0,
                "shared_input_shape": list(x.shape),
                "dsra": dsra_metrics,
                "mhdsra2": mhdsra2_metrics,
                "speedup_ratio": dsra_metrics["elapsed_ms"]
                / max(mhdsra2_metrics["elapsed_ms"], 1e-9),
                "dsra_to_mhdsra2_state_bytes_ratio": dsra_metrics["max_state_bytes"]
                / max(mhdsra2_metrics["max_state_bytes"], 1),
                "mhdsra2_to_dsra_state_bytes_ratio": mhdsra2_metrics["max_state_bytes"]
                / max(dsra_metrics["max_state_bytes"], 1),
                "state_bytes_ratio": dsra_metrics["max_state_bytes"]
                / max(mhdsra2_metrics["max_state_bytes"], 1),
            }
        )

    return {
        "config": {
            "device": str(device),
            "seed": args.seed,
            "batch_size": list(args.batch_size),
            "dim": list(args.dim),
            "warmup_runs": args.warmup_runs,
            "repeat_runs": args.repeat_runs,
            "use_retrieval": bool(args.use_retrieval),
            "retrieval_tokens": args.retrieval_tokens if args.use_retrieval else 0,
            "slots": list(args.slots),
            "read_topk": list(args.read_topk),
            "chunk_sizes": list(args.chunk_sizes),
            "seq_lengths": list(args.seq_lengths),
        },
        "results": results,
        "summary": build_summary(results),
    }


def _case_label(item: dict) -> str:
    """Build a compact identifier for one comparison case.

    中文说明:
    - 调用方 / Called by: `build_summary`, `save_reports`
    - 调用对象 / Calls: 无
    - 作用 / Purpose: 把单条结果格式化为稳定可读的案例标签，便于报告摘要引用
    - 变量 / Variables:
      `item` 为单条对比结果，包含 `batch_size`、`dim`、`seq_len`、`slots`、
      `read_topk`、`chunk_size`
    - 接入 / Integration: 若后续增加 `dim` 等维度，可继续在此标签中追加字段
    - 错误处理 / Error handling: 依赖上游结果结构完整；缺字段时由调用链抛出 KeyError
    - 关键词 / Keywords:
      case|label|summary|report|grid|signature|format|compare|result|标签
    """
    return (
        f"batch={item['batch_size']}, dim={item['dim']}, seq={item['seq_len']}, slots={item['slots']}, "
        f"topk={item['read_topk']}, chunk={item['chunk_size']}"
    )


def _average(items: list[float]) -> float:
    """Return numeric mean for non-empty float lists.

    中文说明:
    - 调用方 / Called by: `build_summary`
    - 调用对象 / Calls: `statistics.fmean`
    - 作用 / Purpose: 统一计算报告摘要中的均值，避免重复书写统计逻辑
    - 变量 / Variables: `items` 为待统计数值列表
    - 接入 / Integration: 新增均值型指标时可直接复用
    - 错误处理 / Error handling: 空列表时返回 `0.0`，避免摘要生成阶段异常中断
    - 关键词 / Keywords:
      average|mean|statistics|summary|metric|helper|float|aggregate|report|均值
    """
    if not items:
        return 0.0
    return float(statistics.fmean(items))


def build_summary(results: list[dict]) -> dict:
    """Build aggregate summary and grouped conclusions for report generation.

    中文说明:
    - 调用方 / Called by: `run_comparison`
    - 调用对象 / Calls: `_average`, `_case_label`, `sorted`
    - 作用 / Purpose: 生成总体统计、最佳最差案例与按维度分组的均值结论
    - 变量 / Variables:
      `results` 为完整网格结果列表, `faster_cases` 为 MHDSRA2 更快的案例集合,
      `grouped_by_*` 为不同维度的分组统计结果
    - 接入 / Integration: 报告模板、外部可视化或 CI 可直接消费本函数返回摘要结构
    - 错误处理 / Error handling: 空结果时返回结构化空摘要，避免 markdown 生成失败
    - 关键词 / Keywords:
      summary|grouped|statistics|speedup|memory|aggregate|report|grid|benchmark|摘要
    """
    if not results:
        return {
            "overall": {
                "total_cases": 0,
                "mhdsra2_faster_cases": 0,
                "mhdsra2_faster_ratio": 0.0,
                "avg_speedup_ratio": 0.0,
                "median_speedup_ratio": 0.0,
                "avg_state_bytes_ratio": 0.0,
                "avg_dsra_to_mhdsra2_state_bytes_ratio": 0.0,
                "avg_mhdsra2_to_dsra_state_bytes_ratio": 0.0,
            },
            "best_speedup_case": None,
            "worst_speedup_case": None,
            "mhdsra2_min_state_overhead_case": None,
            "mhdsra2_max_state_overhead_case": None,
            "best_memory_case": None,
            "worst_memory_case": None,
            "grouped": {},
            "top_speedup_cases": [],
        }

    speedups = [float(item["speedup_ratio"]) for item in results]
    dsra_to_mhdsra2_state_ratios = [
        float(item["dsra_to_mhdsra2_state_bytes_ratio"]) for item in results
    ]
    mhdsra2_to_dsra_state_ratios = [
        float(item["mhdsra2_to_dsra_state_bytes_ratio"]) for item in results
    ]
    faster_cases = [item for item in results if item["speedup_ratio"] > 1.0]
    best_speedup_case = max(results, key=lambda item: item["speedup_ratio"])
    worst_speedup_case = min(results, key=lambda item: item["speedup_ratio"])
    mhdsra2_min_state_overhead_case = min(
        results, key=lambda item: item["mhdsra2_to_dsra_state_bytes_ratio"]
    )
    mhdsra2_max_state_overhead_case = max(
        results, key=lambda item: item["mhdsra2_to_dsra_state_bytes_ratio"]
    )

    grouped = {}
    for field in ("batch_size", "dim", "seq_len", "slots", "read_topk", "chunk_size"):
        values = []
        unique_values = sorted({item[field] for item in results})
        for value in unique_values:
            matched = [item for item in results if item[field] == value]
            values.append(
                {
                    "value": value,
                    "avg_speedup_ratio": _average(
                        [float(item["speedup_ratio"]) for item in matched]
                    ),
                    "avg_dsra_to_mhdsra2_state_bytes_ratio": _average(
                        [
                            float(item["dsra_to_mhdsra2_state_bytes_ratio"])
                            for item in matched
                        ]
                    ),
                    "avg_mhdsra2_to_dsra_state_bytes_ratio": _average(
                        [
                            float(item["mhdsra2_to_dsra_state_bytes_ratio"])
                            for item in matched
                        ]
                    ),
                    "avg_dsra_ms": _average(
                        [float(item["dsra"]["elapsed_ms"]) for item in matched]
                    ),
                    "avg_dsra_std_ms": _average(
                        [float(item["dsra"]["elapsed_ms_std"]) for item in matched]
                    ),
                    "avg_mhdsra2_ms": _average(
                        [float(item["mhdsra2"]["elapsed_ms"]) for item in matched]
                    ),
                    "avg_mhdsra2_std_ms": _average(
                        [float(item["mhdsra2"]["elapsed_ms_std"]) for item in matched]
                    ),
                    "cases": len(matched),
                }
            )
        grouped[field] = values

    top_speedup_cases = []
    for item in sorted(results, key=lambda case: case["speedup_ratio"], reverse=True)[:5]:
        top_speedup_cases.append(
            {
                "label": _case_label(item),
                "speedup_ratio": float(item["speedup_ratio"]),
                "dsra_to_mhdsra2_state_bytes_ratio": float(
                    item["dsra_to_mhdsra2_state_bytes_ratio"]
                ),
                "mhdsra2_to_dsra_state_bytes_ratio": float(
                    item["mhdsra2_to_dsra_state_bytes_ratio"]
                ),
                "state_bytes_ratio": float(item["state_bytes_ratio"]),
            }
        )

    return {
        "overall": {
            "total_cases": len(results),
            "mhdsra2_faster_cases": len(faster_cases),
            "mhdsra2_faster_ratio": len(faster_cases) / max(len(results), 1),
            "avg_speedup_ratio": _average(speedups),
            "median_speedup_ratio": float(statistics.median(speedups)),
            "avg_dsra_to_mhdsra2_state_bytes_ratio": _average(
                dsra_to_mhdsra2_state_ratios
            ),
            "avg_mhdsra2_to_dsra_state_bytes_ratio": _average(
                mhdsra2_to_dsra_state_ratios
            ),
            "avg_state_bytes_ratio": _average(dsra_to_mhdsra2_state_ratios),
        },
        "best_speedup_case": {
            "label": _case_label(best_speedup_case),
            "speedup_ratio": float(best_speedup_case["speedup_ratio"]),
            "dsra_to_mhdsra2_state_bytes_ratio": float(
                best_speedup_case["dsra_to_mhdsra2_state_bytes_ratio"]
            ),
            "mhdsra2_to_dsra_state_bytes_ratio": float(
                best_speedup_case["mhdsra2_to_dsra_state_bytes_ratio"]
            ),
            "state_bytes_ratio": float(best_speedup_case["state_bytes_ratio"]),
        },
        "worst_speedup_case": {
            "label": _case_label(worst_speedup_case),
            "speedup_ratio": float(worst_speedup_case["speedup_ratio"]),
            "dsra_to_mhdsra2_state_bytes_ratio": float(
                worst_speedup_case["dsra_to_mhdsra2_state_bytes_ratio"]
            ),
            "mhdsra2_to_dsra_state_bytes_ratio": float(
                worst_speedup_case["mhdsra2_to_dsra_state_bytes_ratio"]
            ),
            "state_bytes_ratio": float(worst_speedup_case["state_bytes_ratio"]),
        },
        "mhdsra2_min_state_overhead_case": {
            "label": _case_label(mhdsra2_min_state_overhead_case),
            "mhdsra2_to_dsra_state_bytes_ratio": float(
                mhdsra2_min_state_overhead_case["mhdsra2_to_dsra_state_bytes_ratio"]
            ),
            "dsra_to_mhdsra2_state_bytes_ratio": float(
                mhdsra2_min_state_overhead_case["dsra_to_mhdsra2_state_bytes_ratio"]
            ),
            "state_bytes_ratio": float(mhdsra2_min_state_overhead_case["state_bytes_ratio"]),
            "speedup_ratio": float(mhdsra2_min_state_overhead_case["speedup_ratio"]),
        },
        "mhdsra2_max_state_overhead_case": {
            "label": _case_label(mhdsra2_max_state_overhead_case),
            "mhdsra2_to_dsra_state_bytes_ratio": float(
                mhdsra2_max_state_overhead_case["mhdsra2_to_dsra_state_bytes_ratio"]
            ),
            "dsra_to_mhdsra2_state_bytes_ratio": float(
                mhdsra2_max_state_overhead_case["dsra_to_mhdsra2_state_bytes_ratio"]
            ),
            "state_bytes_ratio": float(mhdsra2_max_state_overhead_case["state_bytes_ratio"]),
            "speedup_ratio": float(mhdsra2_max_state_overhead_case["speedup_ratio"]),
        },
        "best_memory_case": {
            "label": _case_label(mhdsra2_max_state_overhead_case),
            "state_bytes_ratio": float(mhdsra2_max_state_overhead_case["state_bytes_ratio"]),
            "speedup_ratio": float(mhdsra2_max_state_overhead_case["speedup_ratio"]),
        },
        "worst_memory_case": {
            "label": _case_label(mhdsra2_min_state_overhead_case),
            "state_bytes_ratio": float(mhdsra2_min_state_overhead_case["state_bytes_ratio"]),
            "speedup_ratio": float(mhdsra2_min_state_overhead_case["speedup_ratio"]),
        },
        "grouped": grouped,
        "top_speedup_cases": top_speedup_cases,
    }


def save_reports(payload: dict, reports_dir: Path) -> tuple[Path, Path]:
    """Persist comparison payload to `reports/` as JSON and Markdown files.

    中文说明:
    - 调用方 / Called by: `main`
    - 调用对象 / Calls: `ensure_reports_dir`, `write_json`, `write_markdown`
    - 作用 / Purpose: 将对比结果写入项目标准报告目录，满足目录结构约束
    - 变量 / Variables: `payload` 结果载荷, `reports_dir` 报告目录
    - 接入 / Integration: CI 或其他脚本可直接消费生成的 json/md 文件
    - 错误处理 / Error handling: 若目录不存在会自动创建；文件写入异常直接向上抛出
    - 关键词 / Keywords:
      reports|json|markdown|persist|artifact|directory|compare|payload|output|报告
    """
    reports_dir = ensure_reports_dir(reports_dir)
    json_path = reports_dir / "mhdsra2_vs_dsra_compare.json"
    md_path = reports_dir / "mhdsra2_vs_dsra_compare.md"
    write_json(json_path, payload)

    lines = [
        "# MHDSRA2 vs DSRA Comparison",
        "",
        "## Config",
        f"- device: `{payload['config']['device']}`",
        f"- batch_size: `{payload['config']['batch_size']}`",
        f"- dim: `{payload['config']['dim']}`",
        f"- warmup_runs: `{payload['config']['warmup_runs']}`",
        f"- repeat_runs: `{payload['config']['repeat_runs']}`",
        f"- use_retrieval: `{payload['config']['use_retrieval']}`",
        f"- retrieval_tokens: `{payload['config']['retrieval_tokens']}`",
        f"- slots grid: `{payload['config']['slots']}`",
        f"- read_topk grid: `{payload['config']['read_topk']}`",
        f"- chunk_size grid: `{payload['config']['chunk_sizes']}`",
        "",
        "## Automatic Summary",
        f"- total cases: `{payload['summary']['overall']['total_cases']}`",
        f"- MHDSRA2 faster cases: `{payload['summary']['overall']['mhdsra2_faster_cases']}` / `{payload['summary']['overall']['total_cases']}`",
        f"- MHDSRA2 faster ratio: `{payload['summary']['overall']['mhdsra2_faster_ratio']:.3f}`",
        f"- average speedup: `{payload['summary']['overall']['avg_speedup_ratio']:.3f}`",
        f"- median speedup: `{payload['summary']['overall']['median_speedup_ratio']:.3f}`",
        f"- average state-bytes ratio (MHDSRA2/DSRA): `{payload['summary']['overall']['avg_mhdsra2_to_dsra_state_bytes_ratio']:.3f}`",
        f"- best speedup case: `{payload['summary']['best_speedup_case']['label']}` => `{payload['summary']['best_speedup_case']['speedup_ratio']:.3f}x`",
        f"- weakest speedup case: `{payload['summary']['worst_speedup_case']['label']}` => `{payload['summary']['worst_speedup_case']['speedup_ratio']:.3f}x`",
        f"- MHDSRA2 min state-overhead case: `{payload['summary']['mhdsra2_min_state_overhead_case']['label']}` => state ratio `{payload['summary']['mhdsra2_min_state_overhead_case']['mhdsra2_to_dsra_state_bytes_ratio']:.3f}`",
        f"- MHDSRA2 max state-overhead case: `{payload['summary']['mhdsra2_max_state_overhead_case']['label']}` => state ratio `{payload['summary']['mhdsra2_max_state_overhead_case']['mhdsra2_to_dsra_state_bytes_ratio']:.3f}`",
        "",
        "## Grouped Conclusions",
    ]
    for field, display_name in (
        ("batch_size", "Batch Size"),
        ("dim", "Dim"),
        ("seq_len", "Seq Len"),
        ("slots", "Slots"),
        ("read_topk", "Read TopK"),
        ("chunk_size", "Chunk Size"),
    ):
        lines.extend(
            [
                "",
                f"### By {display_name}",
                "",
                "| Value | Cases | Avg DSRA ms | Avg DSRA std ms | Avg MHDSRA2 ms | Avg MHDSRA2 std ms | Avg Speedup | Avg State Ratio (MHDSRA2/DSRA) |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in payload["summary"]["grouped"][field]:
            lines.append(
                "| {value} | {cases} | {avg_dsra_ms:.3f} | {avg_dsra_std_ms:.3f} | {avg_mh_ms:.3f} | {avg_mh_std_ms:.3f} | {avg_speedup:.3f} | {avg_state_ratio:.3f} |".format(
                    value=item["value"],
                    cases=item["cases"],
                    avg_dsra_ms=item["avg_dsra_ms"],
                    avg_dsra_std_ms=item["avg_dsra_std_ms"],
                    avg_mh_ms=item["avg_mhdsra2_ms"],
                    avg_mh_std_ms=item["avg_mhdsra2_std_ms"],
                    avg_speedup=item["avg_speedup_ratio"],
                    avg_state_ratio=item["avg_mhdsra2_to_dsra_state_bytes_ratio"],
                )
            )

    lines.extend(
        [
            "",
            "## Top Speedup Cases",
            "",
            "| Rank | Case | Speedup | State Ratio (MHDSRA2/DSRA) |",
            "|---:|---|---:|---:|",
        ]
    )
    for rank, item in enumerate(payload["summary"]["top_speedup_cases"], start=1):
        lines.append(
            "| {rank} | {label} | {speedup:.3f} | {state_ratio:.3f} |".format(
                rank=rank,
                label=item["label"],
                speedup=item["speedup_ratio"],
                state_ratio=item["mhdsra2_to_dsra_state_bytes_ratio"],
            )
        )

    lines.extend(
        [
            "",
            "## Raw Cases",
            "",
        "| Batch | Dim | Seq Len | Slots | Read TopK | Chunk | DSRA ms | DSRA std ms | MHDSRA2 ms | MHDSRA2 std ms | Speedup (DSRA/MHDSRA2) | State Ratio (MHDSRA2/DSRA) | DSRA State Bytes | MHDSRA2 State Bytes |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["results"]:
        lines.append(
            "| {batch_size} | {dim} | {seq_len} | {slots} | {read_topk} | {chunk_size} | {dsra_ms:.3f} | {dsra_std_ms:.3f} | {mh_ms:.3f} | {mh_std_ms:.3f} | {speedup:.3f} | {state_ratio:.3f} | {dsra_state} | {mh_state} |".format(
                batch_size=item["batch_size"],
                dim=item["dim"],
                seq_len=item["seq_len"],
                slots=item["slots"],
                read_topk=item["read_topk"],
                chunk_size=item["chunk_size"],
                dsra_ms=item["dsra"]["elapsed_ms"],
                dsra_std_ms=item["dsra"]["elapsed_ms_std"],
                mh_ms=item["mhdsra2"]["elapsed_ms"],
                mh_std_ms=item["mhdsra2"]["elapsed_ms_std"],
                speedup=item["speedup_ratio"],
                state_ratio=item["mhdsra2_to_dsra_state_bytes_ratio"],
                dsra_state=item["dsra"]["max_state_bytes"],
                mh_state=item["mhdsra2"]["max_state_bytes"],
            )
        )
    write_markdown(md_path, lines)
    return json_path, md_path


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the comparison runner.

    中文说明:
    - 调用方 / Called by: `main`
    - 调用对象 / Calls: `argparse.ArgumentParser`, `add_argument`
    - 作用 / Purpose: 暴露多参数网格对比测试配置，供脚本和主入口复用
    - 变量 / Variables: 解析序列长度、维度、槽位/top-k/chunk 候选列表、设备与输出目录参数
    - 接入 / Integration: 可通过 `python scripts/compare_mhdsra2_vs_dsra.py ...` 调用
    - 错误处理 / Error handling: 参数类型与必填校验由 argparse 自动处理
    - 关键词 / Keywords:
      parser|cli|args|benchmark|grid|sequence|device|reports|config|脚本
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq-lengths",
        nargs="+",
        type=int,
        default=[131072, 262144, 524288, 786432, 1048576],
    )
    parser.add_argument("--batch-size", nargs="+", type=int, default=[1])
    parser.add_argument("--dim", nargs="+", type=int, default=[64])
    parser.add_argument("--slots", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--read-topk", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--repeat-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use-retrieval", action="store_true")
    parser.add_argument("--retrieval-tokens", type=int, default=32)
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    return parser


def main(argv: list[str] | None = None) -> dict:
    """Run the comparison workflow and emit machine-readable report paths.

    中文说明:
    - 调用方 / Called by: `scripts.main.run_mhdsra2_compare`, 直接 CLI 执行
    - 调用对象 / Calls: `build_parser`, `run_comparison`, `save_reports`
    - 作用 / Purpose: 作为主入口执行 mhdsra2 与 DSRA 的对比测试并打印产物位置
    - 变量 / Variables: `argv` 命令行参数，`payload` 对比结果，`json_path/md_path` 报告路径
    - 接入 / Integration: 可被 `python main.py mhdsra2_compare` 或直接脚本调用
    - 错误处理 / Error handling: 下层异常直接抛出，使测试和命令行都能得到非零退出
    - 关键词 / Keywords:
      main|entrypoint|reports|cli|compare|json|markdown|artifact|runner|主入口
    """
    args = build_parser().parse_args(argv)
    payload = run_comparison(args)
    json_path, md_path = save_reports(payload, args.reports_dir)
    print("MHDSRA2_COMPARE_STATUS=ok")
    print(f"MHDSRA2_COMPARE_JSON={json_path.as_posix()}")
    print(f"MHDSRA2_COMPARE_MD={md_path.as_posix()}")
    return payload


if __name__ == "__main__":
    main()
