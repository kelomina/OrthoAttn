import random
import statistics
import time
from pathlib import Path

import torch

from scripts.json_retrieval_test import (
    DEFAULT_LOCAL_CONTEXT_MODE,
    DEFAULT_LOCAL_CONTEXT_SIZE,
    VOCAB_SIZE,
    build_retrieval_model,
    run_json_retrieval_generalization_test,
)
from src.dsra.report_utils import ensure_reports_dir, write_json, write_markdown


ATTENTION_FAMILY_MODEL_TYPES = (
    "dsra",
    "sparse_attention",
    "sliding_window_attention",
    "linear_attention",
    "standard_attention",
)
MODEL_LABELS = {
    "dsra": "DSRA",
    "sparse_attention": "Sparse Attention",
    "sliding_window_attention": "Sliding Window Attention",
    "linear_attention": "Linear Attention",
    "standard_attention": "Standard Attention",
}
TASK_VARIANTS = {
    "baseline": {
        "evidence_loss_weight": 0.0,
        "evidence_hint_weight": 0.0,
    },
    "evidence_extract": {
        "evidence_window_count": 16,
        "evidence_loss_weight": 0.5,
        "evidence_hint_weight": 0.0,
        "evidence_min_context_bytes": 16384,
    },
}
DEFAULT_GENERALIZATION_BENCHMARK_KWARGS = {
    "epochs": 80,
    "epochs_grid": [80],
    "eval_interval": 10,
    "dim": 128,
    "K": 128,
    "kr_grid": [32],
    "chunk_size_grid": [256],
    "lr_grid": [5e-4],
    "warmup_ratio_grid": [0.2],
    "scheduled_sampling_max_ratio_grid": [0.0],
    "train_dataset_size": 12,
    "validation_dataset_size": 4,
    "test_dataset_size": 4,
    "generalization_score_mode": "teacher_forced",
    "local_context_mode": DEFAULT_LOCAL_CONTEXT_MODE,
    "local_context_size": DEFAULT_LOCAL_CONTEXT_SIZE,
    "final_polish_epochs": 0,
    "final_generation_polish_epochs": 0,
}
DEFAULT_TASK_SEED_ROOTS = (7, 11, 19)


def build_task_seed_bundle(seed_root):
    return {
        "seed_root": int(seed_root),
        "train_dataset_seed": int(seed_root),
        "validation_dataset_seed": int(seed_root) + 101,
        "test_dataset_seed": int(seed_root) + 202,
        "pair_split_seed": int(seed_root) + 303,
        "torch_seed": int(seed_root) + 404,
        "python_seed": int(seed_root) + 505,
    }


def set_task_seed_bundle(seed_bundle):
    random.seed(seed_bundle["python_seed"])
    torch.manual_seed(seed_bundle["torch_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_bundle["torch_seed"])


def summarize_metric_values(values):
    filtered_values = [
        value
        for value in values
        if value is not None and isinstance(value, (int, float, bool))
    ]
    if not filtered_values:
        return None
    if len(filtered_values) == 1:
        return {"mean": filtered_values[0], "std": 0.0, "n": 1}
    return {
        "mean": statistics.mean(filtered_values),
        "std": statistics.stdev(filtered_values),
        "n": len(filtered_values),
    }


def aggregate_seeded_task_runs(seed_runs):
    metric_names = sorted(
        {
            metric_name
            for seed_run in seed_runs
            for metric_name in seed_run["metrics"].keys()
        }
    )
    return {
        metric_name: summarize_metric_values(
            [seed_run["metrics"].get(metric_name) for seed_run in seed_runs]
        )
        for metric_name in metric_names
    }


def get_model_family_config(model):
    family_config = {}
    for attr_name in (
        "window_size",
        "sparse_local_window",
        "sparse_global_stride",
    ):
        if hasattr(model, attr_name):
            family_config[attr_name] = int(getattr(model, attr_name))
    return family_config


def benchmark_attention_family_complexity(
    model_types=None,
    seq_lengths=None,
    dim=128,
    K=128,
    kr=32,
    chunk_size=256,
    batch_size=1,
    warmup_runs=1,
    measured_runs=2,
    local_context_size=DEFAULT_LOCAL_CONTEXT_SIZE,
    local_context_mode=DEFAULT_LOCAL_CONTEXT_MODE,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_types = tuple(model_types or ATTENTION_FAMILY_MODEL_TYPES)
    seq_lengths = list(seq_lengths or [1024, 4096, 16384, 32768])

    results = {
        "device": str(device),
        "seq_lengths": seq_lengths,
        "batch_size": batch_size,
        "dim": dim,
        "K": K,
        "kr": kr,
        "chunk_size": chunk_size,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "models": {},
    }

    for model_type in model_types:
        model = build_retrieval_model(
            model_type=model_type,
            vocab_size=VOCAB_SIZE,
            dim=dim,
            K=K,
            kr=kr,
            chunk_size=chunk_size,
            local_context_size=local_context_size,
            local_context_mode=local_context_mode,
        ).to(device)
        model.eval()

        model_results = {
            "label": MODEL_LABELS.get(model_type, model_type),
            "family_config": get_model_family_config(model),
            "length_results": [],
        }

        for seq_len in seq_lengths:
            try:
                timings_ms = []
                peak_mem_mb = 0.0
                for run_idx in range(warmup_runs + measured_runs):
                    x = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), dtype=torch.long, device=device)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()

                    start_time = time.perf_counter()
                    with torch.no_grad():
                        _ = model(x)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        peak_mem_mb = max(
                            peak_mem_mb,
                            torch.cuda.max_memory_allocated() / (1024 ** 2),
                        )
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    if run_idx >= warmup_runs:
                        timings_ms.append(elapsed_ms)

                model_results["length_results"].append(
                    {
                        "seq_len": seq_len,
                        "status": "ok",
                        "mean_time_ms": statistics.mean(timings_ms),
                        "std_time_ms": statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0,
                        "peak_mem_mb": peak_mem_mb,
                    }
                )
            except torch.cuda.OutOfMemoryError:
                model_results["length_results"].append(
                    {
                        "seq_len": seq_len,
                        "status": "oom",
                    }
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results["models"][model_type] = model_results

    return results


def summarize_generalization_result(result):
    validation_eval = result["validation_pool_evaluation"]
    test_eval = result["test_pool_evaluation"]
    return {
        "model_type": result["config"]["model_type"],
        "validation_generation_exact": validation_eval["generation_exact_match_rate"],
        "validation_generation_seq_acc": validation_eval["generation_mean_sequence_accuracy"],
        "validation_teacher_forced_exact": validation_eval["teacher_forced_exact_match_rate"],
        "validation_teacher_forced_seq_acc": validation_eval["teacher_forced_mean_sequence_accuracy"],
        "test_generation_exact": test_eval["generation_exact_match_rate"],
        "test_generation_seq_acc": test_eval["generation_mean_sequence_accuracy"],
        "test_teacher_forced_exact": test_eval["teacher_forced_exact_match_rate"],
        "test_teacher_forced_seq_acc": test_eval["teacher_forced_mean_sequence_accuracy"],
        "validation_evidence_window_acc": validation_eval.get("evidence_window_accuracy"),
        "test_evidence_window_acc": test_eval.get("evidence_window_accuracy"),
        "validation_extract_exact": validation_eval.get("extract_then_compose_exact_match_rate"),
        "test_extract_exact": test_eval.get("extract_then_compose_exact_match_rate"),
        "validation_extract_seq_acc": validation_eval.get("extract_then_compose_mean_sequence_accuracy"),
        "test_extract_seq_acc": test_eval.get("extract_then_compose_mean_sequence_accuracy"),
    }


def run_attention_family_json_retrieval_benchmark(
    reports_dir,
    model_types=None,
    task_variants=None,
    generalization_kwargs=None,
    task_seed_roots=None,
):
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_types = tuple(model_types or ATTENTION_FAMILY_MODEL_TYPES)
    task_variants = tuple(task_variants or TASK_VARIANTS.keys())
    task_seed_roots = tuple(task_seed_roots or DEFAULT_TASK_SEED_ROOTS)
    generalization_kwargs = {
        **DEFAULT_GENERALIZATION_BENCHMARK_KWARGS,
        **dict(generalization_kwargs or {}),
    }

    results = {
        "task_variants": {},
        "shared_generalization_kwargs": generalization_kwargs,
        "task_seed_roots": list(task_seed_roots),
    }

    for variant_name in task_variants:
        variant_kwargs = dict(TASK_VARIANTS[variant_name])
        variant_results = []
        for model_type in model_types:
            model_dir = reports_dir / variant_name / model_type
            seed_runs = []
            for seed_root in task_seed_roots:
                seed_bundle = build_task_seed_bundle(seed_root)
                set_task_seed_bundle(seed_bundle)
                seed_dir = model_dir / f"seed_{seed_root}"
                result = run_json_retrieval_generalization_test(
                    reports_dir=seed_dir / "reports",
                    model_type=model_type,
                    train_dataset_seed=seed_bundle["train_dataset_seed"],
                    validation_dataset_seed=seed_bundle["validation_dataset_seed"],
                    test_dataset_seed=seed_bundle["test_dataset_seed"],
                    pair_split_seed=seed_bundle["pair_split_seed"],
                    **generalization_kwargs,
                    **variant_kwargs,
                )
                seed_runs.append(
                    {
                        "seed_root": seed_root,
                        "seed_bundle": seed_bundle,
                        "report_dir": str(seed_dir / "reports"),
                        "metrics": summarize_generalization_result(result),
                    }
                )

            family_model = build_retrieval_model(
                model_type=model_type,
                vocab_size=VOCAB_SIZE,
                dim=generalization_kwargs.get("dim", 128),
                K=generalization_kwargs.get("K", 128),
                kr=(generalization_kwargs.get("kr_grid") or [32])[0],
                chunk_size=(generalization_kwargs.get("chunk_size_grid") or [256])[0],
                local_context_size=generalization_kwargs.get("local_context_size", DEFAULT_LOCAL_CONTEXT_SIZE),
                local_context_mode=generalization_kwargs.get("local_context_mode", DEFAULT_LOCAL_CONTEXT_MODE),
            )
            aggregate_metrics = aggregate_seeded_task_runs(seed_runs)
            summary = {
                "model_type": model_type,
                "label": MODEL_LABELS.get(model_type, model_type),
                "family_config": get_model_family_config(family_model),
                "aggregate_metrics": aggregate_metrics,
                "seed_runs": seed_runs,
                "model_report_dir": str(model_dir),
            }
            del family_model
            variant_results.append(summary)
        results["task_variants"][variant_name] = {
            "variant_kwargs": variant_kwargs,
            "model_results": variant_results,
        }

    return results


def save_attention_family_benchmark_reports(reports_dir, complexity_results, task_results):
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "complexity": complexity_results,
        "json_retrieval": task_results,
    }
    write_json(reports_dir / "attention_family_benchmark_summary.json", payload)

    lines = [
        "# Attention Family Benchmark Summary",
        "",
        "## Test Design",
        "- Complexity benchmark: random token forward pass with shared `dim`, `chunk_size`, and local-context setup.",
        "- Task benchmark: held-out `museum/artifact` JSON retrieval generalization under identical training settings.",
        f"- Task seed roots: `{task_results['task_seed_roots']}`",
        "- Task variants:",
        "  baseline: plain end-to-end token decoding",
        "  evidence_extract: add evidence window supervision and report extract-then-compose recovery",
        "",
        "## Complexity Benchmark",
        f"- Device: `{complexity_results['device']}`",
        f"- Sequence Lengths: `{complexity_results['seq_lengths']}`",
        f"- Batch Size: `{complexity_results['batch_size']}`",
        "",
    ]

    for model_type, model_results in complexity_results["models"].items():
        lines.append(f"### {model_results['label']}")
        if model_results["family_config"]:
            lines.append(f"- Family Config: `{model_results['family_config']}`")
        for length_result in model_results["length_results"]:
            if length_result["status"] != "ok":
                lines.append(f"- seq_len={length_result['seq_len']}: `OOM`")
            else:
                lines.append(
                    f"- seq_len={length_result['seq_len']}: "
                    f"time=`{length_result['mean_time_ms']:.2f} ms +/- {length_result['std_time_ms']:.2f}` "
                    f"| peak_mem=`{length_result['peak_mem_mb']:.2f} MB`"
                )
        lines.append("")

    lines.extend(["## JSON Retrieval Comparison", ""])
    for variant_name, variant_payload in task_results["task_variants"].items():
        variant_results = variant_payload["model_results"]
        lines.append(f"### {variant_name}")
        if variant_payload["variant_kwargs"]:
            lines.append(f"- Variant Config: `{variant_payload['variant_kwargs']}`")
        for result in variant_results:
            aggregate_metrics = result["aggregate_metrics"]
            val_tf_seq = aggregate_metrics["validation_teacher_forced_seq_acc"]
            test_tf_seq = aggregate_metrics["test_teacher_forced_seq_acc"]
            val_gen_seq = aggregate_metrics["validation_generation_seq_acc"]
            test_gen_seq = aggregate_metrics["test_generation_seq_acc"]
            lines.append(
                f"- {result['label']}: "
                f"val_tf_seq=`{val_tf_seq['mean']*100:.2f}% +/- {val_tf_seq['std']*100:.2f}` "
                f"| test_tf_seq=`{test_tf_seq['mean']*100:.2f}% +/- {test_tf_seq['std']*100:.2f}` "
                f"| val_gen_seq=`{val_gen_seq['mean']*100:.2f}% +/- {val_gen_seq['std']*100:.2f}` "
                f"| test_gen_seq=`{test_gen_seq['mean']*100:.2f}% +/- {test_gen_seq['std']*100:.2f}`"
            )
            if aggregate_metrics["validation_evidence_window_acc"] is not None:
                val_window = aggregate_metrics["validation_evidence_window_acc"]
                test_window = aggregate_metrics["test_evidence_window_acc"]
                val_extract_exact = aggregate_metrics["validation_extract_exact"]
                test_extract_exact = aggregate_metrics["test_extract_exact"]
                lines.append(
                    f"  evidence: val_window=`{val_window['mean']*100:.2f}% +/- {val_window['std']*100:.2f}` "
                    f"| test_window=`{test_window['mean']*100:.2f}% +/- {test_window['std']*100:.2f}` "
                    f"| val_extract_exact=`{val_extract_exact['mean']*100:.2f}% +/- {val_extract_exact['std']*100:.2f}` "
                    f"| test_extract_exact=`{test_extract_exact['mean']*100:.2f}% +/- {test_extract_exact['std']*100:.2f}`"
                )
            lines.append(f"  report root: `{result['model_report_dir']}`")
            for seed_run in result["seed_runs"]:
                lines.append(
                    f"  seed={seed_run['seed_root']}: "
                    f"test_tf_seq=`{seed_run['metrics']['test_teacher_forced_seq_acc']*100:.2f}%` "
                    f"| report=`{seed_run['report_dir']}`"
                )
        lines.append("")

    write_markdown(reports_dir / "attention_family_benchmark_summary.md", lines)


def run_attention_family_benchmark_suite(
    reports_dir=None,
    model_types=None,
    complexity_seq_lengths=None,
    complexity_kwargs=None,
    task_variants=None,
    generalization_kwargs=None,
    task_seed_roots=None,
):
    reports_dir = ensure_reports_dir(
        Path(reports_dir) if reports_dir is not None else Path(__file__).resolve().parents[1] / "reports"
    )
    task_reports_dir = reports_dir / "attention_family_json_retrieval"
    complexity_results = benchmark_attention_family_complexity(
        model_types=model_types,
        seq_lengths=complexity_seq_lengths,
        **(complexity_kwargs or {}),
    )
    task_results = run_attention_family_json_retrieval_benchmark(
        reports_dir=task_reports_dir,
        model_types=model_types,
        task_variants=task_variants,
        generalization_kwargs=generalization_kwargs,
        task_seed_roots=task_seed_roots,
    )
    save_attention_family_benchmark_reports(
        reports_dir=reports_dir,
        complexity_results=complexity_results,
        task_results=task_results,
    )
    return {
        "reports_dir": str(reports_dir),
        "complexity": complexity_results,
        "json_retrieval": task_results,
    }


if __name__ == "__main__":
    run_attention_family_benchmark_suite()
