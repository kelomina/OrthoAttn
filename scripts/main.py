import argparse
import contextlib
import os
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dsra.report_utils import ensure_reports_dir, write_markdown


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def get_reports_dir():
    return ensure_reports_dir(Path(__file__).resolve().parents[1])


def write_run_report(reports_dir, executed_tests):
    lines = [
        "# DSRA Unified Test Report",
        "",
        "## Executed Suites",
    ]
    for test_name in executed_tests:
        lines.append(f"- {test_name}")
    lines.extend(
        [
            "",
            "## Generated Files",
            "- `reports/all_output.txt`",
            "- `reports/ablation_summary.md`",
            "- `reports/ablation_summary.json`",
            "- `reports/needle_capacity_results.md`",
            "- `reports/needle_capacity_results.json`",
            "- `reports/json_retrieval_report.md`",
            "- `reports/json_retrieval_report.json`",
            "- `reports/json_retrieval_generalization_report.md`",
            "- `reports/json_retrieval_generalization_report.json`",
            "- `reports/mhdsra2_vs_dsra_compare.md`",
            "- `reports/mhdsra2_vs_dsra_compare.json`",
            "- `reports/mhdsra2_vs_dsra_next_round_benchmark.md`",
            "- `reports/mhdsra2_vs_dsra_next_round_benchmark.json`",
        ]
    )
    write_markdown(reports_dir / "run_summary.md", lines)

def run_unittests():
    print("\n" + "="*50)
    print("Running Unit Tests (Math, Gradients, LLM Compatibility)")
    print("="*50)
    tests_dir = Path(__file__).resolve().parents[1] / "tests"
    if not tests_dir.exists():
        print("\nNo tests directory found. Skipping unit tests.")
        return
    loader = unittest.TestLoader()
    suite = loader.discover(str(tests_dir), pattern='test_*.py', top_level_dir=str(Path(__file__).resolve().parents[1]))
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        print("\nUnit tests failed! Aborting further tests.")
        sys.exit(1)

def run_benchmark():
    print("\n" + "="*50)
    print("Running Complexity & Performance Benchmark")
    print("="*50)
    from scripts.benchmark_complexity import run_benchmark as bench
    bench()

def run_saturation():
    print("\n" + "="*50)
    print("Running State Saturation & Decay Test")
    print("="*50)
    from tests.test_state_saturation import run_saturation_test as sat
    sat()

def run_associative_recall():
    print("\n" + "="*50)
    print("Running Associative Recall Toy Task")
    print("="*50)
    from scripts.toy_task_associative_recall import train as ar_train
    ar_train()

def run_needle_in_haystack():
    print("\n" + "="*50)
    print("Running Needle-In-A-Haystack Long-Context Sweep")
    print("="*50)
    from scripts.needle_in_haystack_test import run_niah_test as niah
    return niah()


def run_needle_capacity_reports():
    print("\n" + "="*50)
    print("Running Needle-In-A-Haystack Capacity Reports")
    print("="*50)
    from scripts.needle_in_haystack_test import run_niah_capacity_test, save_niah_capacity_reports

    reports_dir = get_reports_dir()
    forward_results = run_niah_capacity_test(mode='forward_only')
    train_results = run_niah_capacity_test(mode='train_step')
    save_niah_capacity_reports(forward_results, train_results, reports_dir)
    return {"forward_only": forward_results, "train_step": train_results}


def run_json_retrieval():
    print("\n" + "="*50)
    print("Running JSON File Retrieval Test")
    print("="*50)
    from scripts.json_retrieval_test import run_json_retrieval_test

    return run_json_retrieval_test(reports_dir=get_reports_dir())


def run_json_retrieval_generalization():
    print("\n" + "="*50)
    print("Running JSON Retrieval Generalization Test")
    print("="*50)
    from scripts.json_retrieval_test import run_json_retrieval_generalization_test

    return run_json_retrieval_generalization_test(reports_dir=get_reports_dir())


def run_attention_family_benchmark():
    print("\n" + "="*50)
    print("Running Attention Family Benchmark")
    print("="*50)
    from scripts.attention_family_benchmark import run_attention_family_benchmark_suite

    return run_attention_family_benchmark_suite(reports_dir=get_reports_dir())


def run_ablation():
    print("\n" + "="*50)
    print("Running Ablation Study (Core Mechanisms)")
    print("="*50)
    from scripts.ablation_study import main as ablation
    return ablation(reports_dir=get_reports_dir())


def run_mhdsra2_verify():
    print("\n" + "=" * 50)
    print("Running MHDSRA2 Verification")
    print("=" * 50)
    from scripts.verify_mhdsra2 import main as verify_mhdsra2_main

    return verify_mhdsra2_main([])


def run_mhdsra2_compare():
    print("\n" + "=" * 50)
    print("Running MHDSRA2 vs DSRA Comparison")
    print("=" * 50)
    from scripts.compare_mhdsra2_vs_dsra import main as compare_main

    fast_compare = os.environ.get("DSRA_FAST_COMPARE", "").strip().lower() in {"1", "true", "yes"}
    if fast_compare:
        return compare_main(
            [
                "--seq-lengths",
                "256",
                "512",
                "1024",
                "--batch-size",
                "1",
                "--slots",
                "32",
                "64",
                "--read-topk",
                "4",
                "8",
                "--chunk-sizes",
                "16",
                "32",
                "64",
                "128",
                "--warmup-runs",
                "1",
                "--repeat-runs",
                "2",
                "--device",
                "cpu",
                "--reports-dir",
                str(get_reports_dir()),
            ]
        )
    return compare_main(["--reports-dir", str(get_reports_dir())])


def run_next_round_benchmark():
    print("\n" + "=" * 50)
    print("Running MHDSRA2 Next-Round Benchmark")
    print("=" * 50)
    from scripts.next_round_benchmark_runner import main as next_round_benchmark_main

    return next_round_benchmark_main(["--reports-dir", str(get_reports_dir())])


def main():
    parser = argparse.ArgumentParser(
        description="DSRA (Decoupled Sparse Routing Attention) Unified Test Runner",
        epilog=(
            "Environment variables:\n"
            "  DSRA_FAST_ALL=1\n"
            "    - Effect: when running `python main.py all`, execute a minimal suite for fast CI/regression.\n"
            "    - Reports: still generates `reports/all_output.txt` and `reports/run_summary.md`.\n"
            "  DSRA_FAST_COMPARE=1\n"
            "    - Effect: when running `python main.py mhdsra2_compare`, execute a smaller CPU comparison workload.\n"
            "    - Reports: generates `reports/mhdsra2_vs_dsra_compare.json` and `.md`.\n"
            "\n"
            "Machine-readable output (stdout):\n"
            "  When running `python main.py report` (or `python scripts/main.py report`):\n"
            "    DSRA_REPORT_STATUS=ok\n"
            "    DSRA_REPORT_RUN_SUMMARY=reports/run_summary.md\n"
            "    DSRA_REPORT_EXECUTED_SUITES=0\n"
            "\n"
            "  When running `python main.py all` (or `python scripts/main.py all`):\n"
            "    DSRA_ALL_STATUS=ok\n"
            "    DSRA_ALL_LOG=reports/all_output.txt\n"
            "    DSRA_ALL_RUN_SUMMARY=reports/run_summary.md\n"
            "    DSRA_ALL_EXECUTED_SUITES=<N>\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Define available test suites
    test_choices = [
        'unit',         # test_dsra_math.py, test_llm_compatibility.py
        'benchmark',    # benchmark_complexity.py
        'saturation',   # tests/test_state_saturation.py
        'recall',       # toy_task_associative_recall.py
        'needle',       # needle_in_haystack_test.py
        'needle_capacity',
        'json_retrieval',
        'json_retrieval_generalization',
        'attention_family_benchmark',
        'mhdsra2',
        'mhdsra2_compare',
        'next_round_benchmark',
        'ablation',     # ablation_study.py
        'report',
        'all'           # Run everything in sequence
    ]
    
    parser.add_argument(
        'test_name', 
        type=str, 
        choices=test_choices,
        help="Specify which test to run, or 'all' to run everything."
    )

    args = parser.parse_args()

    # Mapping choices to functions
    tests_to_run = []
    reports_dir = get_reports_dir()
    
    if args.test_name == 'all':
        fast_all = os.environ.get("DSRA_FAST_ALL", "").strip().lower() in {"1", "true", "yes"}
        tests_to_run = [
            ("mhdsra2", run_mhdsra2_verify),
        ]
        if not fast_all:
            tests_to_run = [
                ("unit", run_unittests),
                ("benchmark", run_benchmark),
                ("saturation", run_saturation),
                ("recall", run_associative_recall),
                ("needle", run_needle_in_haystack),
                ("needle_capacity", run_needle_capacity_reports),
                ("json_retrieval", run_json_retrieval),
                ("mhdsra2", run_mhdsra2_verify),
                ("mhdsra2_compare", run_mhdsra2_compare),
                ("ablation", run_ablation),
            ]
    else:
        mapping = {
            'unit': ("unit", run_unittests),
            'benchmark': ("benchmark", run_benchmark),
            'saturation': ("saturation", run_saturation),
            'recall': ("recall", run_associative_recall),
            'needle': ("needle", run_needle_in_haystack),
            'needle_capacity': ("needle_capacity", run_needle_capacity_reports),
            'json_retrieval': ("json_retrieval", run_json_retrieval),
            'json_retrieval_generalization': ("json_retrieval_generalization", run_json_retrieval_generalization),
            'attention_family_benchmark': ("attention_family_benchmark", run_attention_family_benchmark),
            'mhdsra2': ("mhdsra2", run_mhdsra2_verify),
            'mhdsra2_compare': ("mhdsra2_compare", run_mhdsra2_compare),
            'next_round_benchmark': ("next_round_benchmark", run_next_round_benchmark),
            'ablation': ("ablation", run_ablation),
            'report': ("report", lambda: write_run_report(reports_dir, [])),
        }
        tests_to_run.append(mapping[args.test_name])

    # Execute selected tests
    executed_tests = []
    if args.test_name == 'all':
        log_path = reports_dir / "all_output.txt"
        with log_path.open("w", encoding="utf-8") as log_file:
            tee = TeeStream(sys.stdout, log_file)
            with contextlib.redirect_stdout(tee):
                for test_name, test_func in tests_to_run:
                    executed_tests.append(test_name)
                    test_func()
                print("\n" + "="*50)
                print("All requested tests completed successfully!")
                print("="*50)
        write_run_report(reports_dir, executed_tests)
        run_summary = reports_dir / "run_summary.md"
        try:
            run_summary_display = run_summary.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            run_summary_display = run_summary.as_posix()
        try:
            log_path_display = log_path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            log_path_display = log_path.as_posix()
        print("DSRA_ALL_STATUS=ok")
        print(f"DSRA_ALL_LOG={log_path_display}")
        print(f"DSRA_ALL_RUN_SUMMARY={run_summary_display}")
        print(f"DSRA_ALL_EXECUTED_SUITES={len(executed_tests)}")
    else:
        test_name, test_func = tests_to_run[0]
        executed_tests.append(test_name)
        test_func()
        if test_name == "report":
            run_summary = reports_dir / "run_summary.md"
            try:
                run_summary_display = run_summary.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                run_summary_display = run_summary.as_posix()
            print("DSRA_REPORT_STATUS=ok")
            print(f"DSRA_REPORT_RUN_SUMMARY={run_summary_display}")
            print("DSRA_REPORT_EXECUTED_SUITES=0")
            return

        print("\n" + "=" * 50)
        print("All requested tests completed successfully!")
        print("=" * 50)

if __name__ == '__main__':
    main()
