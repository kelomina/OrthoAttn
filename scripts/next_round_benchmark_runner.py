"""Dedicated next-round benchmark runner for MHDSRA2 vs DSRA.

This runner keeps the existing lightweight compare script focused on microbench
latency/state overhead, while reusing its normalized report builders to land a
heavier cross-task benchmark based on task-level accuracy metrics.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from scripts.compare_mhdsra2_vs_dsra import (
    build_benchmark_comparison_row,
    build_benchmark_payload,
    save_benchmark_reports,
)
from scripts.diagnostic_memory_benchmark import (
    add_diagnostic_cli_arguments,
    run_diagnostic_benchmarks,
)
from scripts.needle_in_haystack_test import is_oom_error, run_single_niah_test
from scripts.json_retrieval_test import (
    DEFAULT_LOCAL_CONTEXT_MODE,
    DEFAULT_LOCAL_CONTEXT_SIZE,
    run_json_retrieval_generalization_test,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int) -> None:
    """Seed Python and Torch RNGs for fair DSRA/MHDSRA2 benchmark reruns.

    中文说明:
    - 调用方 / Called by: `run_niah_section`, `run_json_generalization_section`
    - 调用对象 / Calls: `random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`
    - 作用 / Purpose: 保证 DSRA 与 MHDSRA2 在相同 benchmark case 下共享一致随机性来源
    - 变量 / Variables: `seed` 为当前任务实例使用的整型随机种子
    - 接入 / Integration: 新增 benchmark section 时也应在每个模型运行前调用本函数
    - 错误处理 / Error handling: 在无 CUDA 环境下自动跳过 GPU seed 设置
    - 关键词 / Keywords:
      seed|deterministic|fairness|benchmark|torch|python|random|reproducible|runner|种子
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_niah_section(args: argparse.Namespace) -> dict:
    """Run Needle-In-A-Haystack accuracy rows for DSRA and MHDSRA2.

    中文说明:
    - 调用方 / Called by: `run_next_round_benchmark`
    - 调用对象 / Calls: `seed_everything`, `run_single_niah_test`, `build_benchmark_comparison_row`
    - 作用 / Purpose: 接入 `needle_in_haystack_test.py` 的最佳准确率口径，输出统一对比行
    - 变量 / Variables:
      `args.niah_seq_lengths` 为待测上下文长度, `rows` 为标准化输出表行
    - 接入 / Integration: 若后续加入更多 NIAH 指标，可继续向本 section 追加 row
    - 错误处理 / Error handling: OOM 时记录缺失行并附带 `notes=OOM`
    - 关键词 / Keywords:
      niah|needle|accuracy|best_acc|section|benchmark|dsra|mhdsra2|compare|统一口径
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows = []
    for seq_len in args.niah_seq_lengths:
        model_scores = {}
        notes = ""
        for model_idx, model_type in enumerate(("dsra", "mhdsra2")):
            seed_everything(args.seed + seq_len * 10 + model_idx)
            try:
                model_scores[model_type] = run_single_niah_test(
                    seq_len=seq_len,
                    device=device,
                    vocab_size=args.niah_vocab_size,
                    dim=args.niah_dim,
                    num_layers=args.niah_num_layers,
                    K=args.niah_slots,
                    kr=args.niah_read_topk,
                    model_type=model_type,
                )
            except torch.cuda.OutOfMemoryError:
                model_scores[model_type] = None
                notes = "OOM"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except RuntimeError as exc:
                if not is_oom_error(exc):
                    raise
                model_scores[model_type] = None
                notes = "OOM"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except torch.AcceleratorError as exc:
                if not is_oom_error(exc):
                    raise
                model_scores[model_type] = None
                notes = "OOM"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        rows.append(
            build_benchmark_comparison_row(
                suite="needle_in_haystack",
                task=f"seq_len={seq_len}",
                split="overall",
                metric="best_accuracy",
                dsra_value=model_scores.get("dsra"),
                mhdsra2_value=model_scores.get("mhdsra2"),
                higher_is_better=True,
                notes=notes,
                metadata={
                    "seq_len": seq_len,
                    "device": str(device),
                },
            )
        )

    return {
        "title": "Needle In Haystack",
        "description": "Uses the `run_single_niah_test()` best accuracy criterion from `needle_in_haystack_test.py`.",
        "rows": rows,
    }


def _json_accuracy_rows(dsra_result: dict, mhdsra2_result: dict) -> list[dict]:
    """Convert JSON retrieval pooled evaluations into normalized comparison rows.

    中文说明:
    - 调用方 / Called by: `run_json_generalization_section`
    - 调用对象 / Calls: `build_benchmark_comparison_row`
    - 作用 / Purpose: 接入 `json_retrieval_test.py` 的 teacher_forced/generation 准确率口径
    - 变量 / Variables:
      `dsra_result/mhdsra2_result` 为 generalization 测试返回结果,
      `split_eval_pairs` 定义 validation/test 两个池子
    - 接入 / Integration: 若未来补充更多 pooled accuracy 指标，可继续在本函数扩展
    - 错误处理 / Error handling: 缺失字段时按 `None` 写入，winner 自动降为 `missing`
    - 关键词 / Keywords:
      json_retrieval|teacher_forced|generation|accuracy|exact_match|sequence_accuracy|rows|compare|benchmark|统一口径
    """
    rows = []
    split_eval_pairs = (
        ("validation", dsra_result["validation_pool_evaluation"], mhdsra2_result["validation_pool_evaluation"]),
        ("test", dsra_result["test_pool_evaluation"], mhdsra2_result["test_pool_evaluation"]),
    )
    metric_specs = (
        ("teacher_forced_exact_match_rate", "teacher_forced_exact_match_rate"),
        ("teacher_forced_mean_sequence_accuracy", "teacher_forced_mean_sequence_accuracy"),
        ("generation_exact_match_rate", "generation_exact_match_rate"),
        ("generation_mean_sequence_accuracy", "generation_mean_sequence_accuracy"),
    )
    for split, dsra_eval, mh_eval in split_eval_pairs:
        for metric_key, metric_name in metric_specs:
            rows.append(
                build_benchmark_comparison_row(
                    suite="json_retrieval_generalization",
                    task="museum_artifact_generalization",
                    split=split,
                    metric=metric_name,
                    dsra_value=dsra_eval.get(metric_key),
                    mhdsra2_value=mh_eval.get(metric_key),
                    higher_is_better=True,
                    metadata={"metric_key": metric_key},
                )
            )
    return rows


def run_json_generalization_section(args: argparse.Namespace) -> dict:
    """Run JSON retrieval generalization benchmark for DSRA and MHDSRA2.

    中文说明:
    - 调用方 / Called by: `run_next_round_benchmark`
    - 调用对象 / Calls:
      `seed_everything`, `run_json_retrieval_generalization_test`, `_json_accuracy_rows`
    - 作用 / Purpose: 对齐 `json_retrieval_test.py` 的 teacher_forced/generation pooled accuracy 指标
    - 变量 / Variables:
      `json_kwargs` 为共享任务配置, `results_by_model` 保存两种模型的完整结果
    - 接入 / Integration: runner 层只设置网格为单点，避免再次把 compare 脚本做重
    - 错误处理 / Error handling: 下层训练/评测异常直接上抛，保留真实 benchmark 失败信号
    - 关键词 / Keywords:
      json|generalization|teacher_forced|generation|benchmark|runner|dsra|mhdsra2|accuracy|汇总
    """
    json_kwargs = {
        "reports_dir": None,
        "epochs": args.json_epochs,
        "epochs_grid": [args.json_epochs],
        "eval_interval": args.json_eval_interval,
        "dim": args.json_dim,
        "K": args.json_slots,
        "kr_grid": [args.json_read_topk],
        "chunk_size_grid": [args.json_chunk_size],
        "lr_grid": [args.json_lr],
        "warmup_ratio_grid": [args.json_warmup_ratio],
        "scheduled_sampling_max_ratio_grid": [args.json_scheduled_sampling_max_ratio],
        "train_dataset_size": args.json_train_dataset_size,
        "validation_dataset_size": args.json_validation_dataset_size,
        "test_dataset_size": args.json_test_dataset_size,
        "train_dataset_seed": args.seed,
        "validation_dataset_seed": args.seed + 101,
        "test_dataset_seed": args.seed + 202,
        "pair_split_seed": args.seed + 303,
        "generalization_score_mode": args.json_generalization_score_mode,
        "local_context_size": args.json_local_context_size,
        "local_context_mode": args.json_local_context_mode,
        "final_polish_epochs": args.json_final_polish_epochs,
        "final_generation_polish_epochs": args.json_final_generation_polish_epochs,
    }
    results_by_model = {}
    for model_idx, model_type in enumerate(("dsra", "mhdsra2")):
        seed_everything(args.seed + 10000 + model_idx)
        results_by_model[model_type] = run_json_retrieval_generalization_test(
            model_type=model_type,
            **json_kwargs,
        )

    return {
        "title": "JSON Retrieval Generalization",
        "description": (
            "Uses pooled validation/test accuracy metrics from "
            "`run_json_retrieval_generalization_test()` covering teacher-forced "
            "and generation evaluation."
        ),
        "rows": _json_accuracy_rows(
            dsra_result=results_by_model["dsra"],
            mhdsra2_result=results_by_model["mhdsra2"],
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the dedicated next-round benchmark runner.

    中文说明:
    - 调用方 / Called by: `main`
    - 调用对象 / Calls: `argparse.ArgumentParser`, `add_argument`
    - 作用 / Purpose: 暴露 next-round benchmark 的 Needle/JSON 两类配置与报告目录参数
    - 变量 / Variables: 命令行参数覆盖随机种子、序列长度与 JSON generalization 单点超参
    - 接入 / Integration: 可通过 `python scripts/next_round_benchmark_runner.py` 直接运行
    - 错误处理 / Error handling: 交由 argparse 做类型和必填校验
    - 关键词 / Keywords:
      parser|cli|runner|benchmark|needle|json|reports|config|args|命令行
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--niah-seq-lengths", nargs="+", type=int, default=[8192, 16384])
    parser.add_argument("--niah-vocab-size", type=int, default=100)
    parser.add_argument("--niah-dim", type=int, default=64)
    parser.add_argument("--niah-num-layers", type=int, default=2)
    parser.add_argument("--niah-slots", type=int, default=64)
    parser.add_argument("--niah-read-topk", type=int, default=8)

    parser.add_argument("--json-epochs", type=int, default=80)
    parser.add_argument("--json-eval-interval", type=int, default=10)
    parser.add_argument("--json-dim", type=int, default=128)
    parser.add_argument("--json-slots", type=int, default=128)
    parser.add_argument("--json-read-topk", type=int, default=32)
    parser.add_argument("--json-chunk-size", type=int, default=256)
    parser.add_argument("--json-lr", type=float, default=5e-4)
    parser.add_argument("--json-warmup-ratio", type=float, default=0.2)
    parser.add_argument("--json-scheduled-sampling-max-ratio", type=float, default=0.0)
    parser.add_argument("--json-train-dataset-size", type=int, default=12)
    parser.add_argument("--json-validation-dataset-size", type=int, default=4)
    parser.add_argument("--json-test-dataset-size", type=int, default=4)
    parser.add_argument("--json-generalization-score-mode", type=str, default="teacher_forced")
    parser.add_argument("--json-local-context-size", type=int, default=DEFAULT_LOCAL_CONTEXT_SIZE)
    parser.add_argument("--json-local-context-mode", type=str, default=DEFAULT_LOCAL_CONTEXT_MODE)
    parser.add_argument("--json-final-polish-epochs", type=int, default=0)
    parser.add_argument("--json-final-generation-polish-epochs", type=int, default=0)
    return add_diagnostic_cli_arguments(parser)


def run_next_round_benchmark(args: argparse.Namespace) -> dict:
    """Execute the dedicated next-round benchmark workflow.

    中文说明:
    - 调用方 / Called by: `main`
    - 调用对象 / Calls: `run_niah_section`, `run_json_generalization_section`, `build_benchmark_payload`
    - 作用 / Purpose: 汇总 Needle 与 JSON Retrieval 两类任务准确率，对外输出统一 benchmark payload
    - 变量 / Variables: `sections` 为各任务分节结果, `config` 为最终写入报告的运行配置
    - 接入 / Integration: 主入口或 CI 可直接消费本函数返回结构并调用统一报告函数
    - 错误处理 / Error handling: 下层 benchmark 异常直接向上抛出，避免静默吞错
    - 关键词 / Keywords:
      next_round|benchmark|runner|payload|sections|needle|json|compare|report|统一流程
    """
    sections = [
        run_niah_section(args),
        run_json_generalization_section(args),
    ]
    sections.extend(run_diagnostic_benchmarks(args))
    config = {
        "seed": args.seed,
        "niah_seq_lengths": list(args.niah_seq_lengths),
        "niah_dim": args.niah_dim,
        "niah_num_layers": args.niah_num_layers,
        "niah_slots": args.niah_slots,
        "niah_read_topk": args.niah_read_topk,
        "json_epochs": args.json_epochs,
        "json_dim": args.json_dim,
        "json_slots": args.json_slots,
        "json_read_topk": args.json_read_topk,
        "json_chunk_size": args.json_chunk_size,
        "json_lr": args.json_lr,
        "json_generalization_score_mode": args.json_generalization_score_mode,
        "json_local_context_size": args.json_local_context_size,
        "json_local_context_mode": args.json_local_context_mode,
        "diagnostic_device": args.diagnostic_device,
        "diagnostic_slots": args.diagnostic_slots,
        "diagnostic_key_count": args.diagnostic_key_count,
        "diagnostic_value_count": args.diagnostic_value_count,
        "diagnostic_chunk_size": args.diagnostic_chunk_size,
        "diagnostic_page_size": args.diagnostic_page_size,
        "diagnostic_retrieval_tau": args.diagnostic_retrieval_tau,
        "diagnostic_exact_seq_len": args.diagnostic_exact_seq_len,
        "diagnostic_exact_fact_spacing": args.diagnostic_exact_fact_spacing,
        "diagnostic_override_seq_len": args.diagnostic_override_seq_len,
        "diagnostic_override_gap_grid": list(args.diagnostic_override_gap_grid),
        "diagnostic_fixation_seq_len": args.diagnostic_fixation_seq_len,
        "diagnostic_fixation_distractor_grid": list(args.diagnostic_fixation_distractor_grid),
    }
    return build_benchmark_payload(config=config, sections=sections)


def main(argv: list[str] | None = None) -> dict:
    """Run the dedicated next-round benchmark and emit report artifact paths.

    中文说明:
    - 调用方 / Called by: `scripts.main.run_next_round_benchmark`
    - 调用对象 / Calls: `build_parser`, `run_next_round_benchmark`, `save_benchmark_reports`
    - 作用 / Purpose: 作为专门的 benchmark runner 入口，输出统一对比表与汇总报告
    - 变量 / Variables:
      `payload` 为最终 benchmark 载荷, `json_path/md_path` 为生成报告路径
    - 接入 / Integration: 支持直接 CLI 调用与 `main.py next_round_benchmark` 调用
    - 错误处理 / Error handling: benchmark 执行失败时保持非零退出语义
    - 关键词 / Keywords:
      main|runner|benchmark|reports|json|markdown|artifact|entrypoint|mhdsra2|下一轮
    """
    args = build_parser().parse_args(argv)
    payload = run_next_round_benchmark(args)
    json_path, md_path = save_benchmark_reports(payload, args.reports_dir)
    print("NEXT_ROUND_BENCHMARK_STATUS=ok")
    print(f"NEXT_ROUND_BENCHMARK_JSON={json_path.as_posix()}")
    print(f"NEXT_ROUND_BENCHMARK_MD={md_path.as_posix()}")
    payload["report_paths"] = {
        "json": json_path.as_posix(),
        "markdown": md_path.as_posix(),
    }
    return payload


if __name__ == "__main__":
    main()
