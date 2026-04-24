import argparse
import contextlib
from pathlib import Path
import unittest
import sys

from report_utils import ensure_reports_dir, write_markdown


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
    return ensure_reports_dir(Path(__file__).resolve().parent)


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
        ]
    )
    write_markdown(reports_dir / "run_summary.md", lines)

def run_unittests():
    print("\n" + "="*50)
    print("Running Unit Tests (Math, Gradients, LLM Compatibility)")
    print("="*50)
    tests_dir = Path(__file__).resolve().parent / "tests"
    if not tests_dir.exists():
        print("\nNo tests directory found. Skipping unit tests.")
        return
    loader = unittest.TestLoader()
    suite = loader.discover(str(tests_dir), pattern='test_*.py', top_level_dir=str(Path(__file__).resolve().parent))
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        print("\nUnit tests failed! Aborting further tests.")
        sys.exit(1)

def run_benchmark():
    print("\n" + "="*50)
    print("Running Complexity & Performance Benchmark")
    print("="*50)
    from benchmark_complexity import run_benchmark as bench
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
    from toy_task_associative_recall import train as ar_train
    ar_train()

def run_needle_in_haystack():
    print("\n" + "="*50)
    print("Running Needle-In-A-Haystack Long-Context Sweep")
    print("="*50)
    from needle_in_haystack_test import run_niah_test as niah
    return niah()


def run_needle_capacity_reports():
    print("\n" + "="*50)
    print("Running Needle-In-A-Haystack Capacity Reports")
    print("="*50)
    from needle_in_haystack_test import run_niah_capacity_test, save_niah_capacity_reports

    reports_dir = get_reports_dir()
    forward_results = run_niah_capacity_test(mode='forward_only')
    train_results = run_niah_capacity_test(mode='train_step')
    save_niah_capacity_reports(forward_results, train_results, reports_dir)
    return {"forward_only": forward_results, "train_step": train_results}


def run_json_retrieval():
    print("\n" + "="*50)
    print("Running JSON File Retrieval Test")
    print("="*50)
    from json_retrieval_test import run_json_retrieval_test

    return run_json_retrieval_test(reports_dir=get_reports_dir())


def run_json_retrieval_generalization():
    print("\n" + "="*50)
    print("Running JSON Retrieval Generalization Test")
    print("="*50)
    from json_retrieval_test import run_json_retrieval_generalization_test

    return run_json_retrieval_generalization_test(reports_dir=get_reports_dir())


def run_attention_family_benchmark():
    print("\n" + "="*50)
    print("Running Attention Family Benchmark")
    print("="*50)
    from attention_family_benchmark import run_attention_family_benchmark_suite

    return run_attention_family_benchmark_suite(reports_dir=get_reports_dir())


def run_ablation():
    print("\n" + "="*50)
    print("Running Ablation Study (Core Mechanisms)")
    print("="*50)
    from ablation_study import main as ablation
    return ablation(reports_dir=get_reports_dir())

def main():
    parser = argparse.ArgumentParser(description="DSRA (Decoupled Sparse Routing Attention) Unified Test Runner")
    
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
        tests_to_run = [
            ("unit", run_unittests),
            ("benchmark", run_benchmark),
            ("saturation", run_saturation),
            ("recall", run_associative_recall),
            ("needle", run_needle_in_haystack),
            ("needle_capacity", run_needle_capacity_reports),
            ("json_retrieval", run_json_retrieval),
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
    else:
        test_name, test_func = tests_to_run[0]
        executed_tests.append(test_name)
        test_func()
        if test_name == "report":
            print("\nReport files generated successfully.")
        
        print("\n" + "="*50)
        print("All requested tests completed successfully!")
        print("="*50)

if __name__ == '__main__':
    main()
