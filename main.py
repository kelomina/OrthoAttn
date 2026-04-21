import argparse
import unittest
import sys

def run_unittests():
    print("\n" + "="*50)
    print("Running Unit Tests (Math, Gradients, LLM Compatibility)")
    print("="*50)
    # Discover and run all test_*.py files
    loader = unittest.TestLoader()
    suite = loader.discover('.', pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2)
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
    from test_state_saturation import run_saturation_test as sat
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
    niah()

def run_ablation():
    print("\n" + "="*50)
    print("Running Ablation Study (Core Mechanisms)")
    print("="*50)
    from ablation_study import main as ablation
    ablation()

def main():
    parser = argparse.ArgumentParser(description="DSRA (Decoupled Sparse Routing Attention) Unified Test Runner")
    
    # Define available test suites
    test_choices = [
        'unit',         # test_dsra_math.py, test_llm_compatibility.py
        'benchmark',    # benchmark_complexity.py
        'saturation',   # test_state_saturation.py
        'recall',       # toy_task_associative_recall.py
        'needle',       # needle_in_haystack_test.py
        'ablation',     # ablation_study.py
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
    
    if args.test_name == 'all':
        tests_to_run = [
            run_unittests,
            run_benchmark,
            run_saturation,
            run_associative_recall,
            run_needle_in_haystack,
            run_ablation
        ]
    else:
        mapping = {
            'unit': run_unittests,
            'benchmark': run_benchmark,
            'saturation': run_saturation,
            'recall': run_associative_recall,
            'needle': run_needle_in_haystack,
            'ablation': run_ablation
        }
        tests_to_run.append(mapping[args.test_name])

    # Execute selected tests
    for test_func in tests_to_run:
        test_func()
        
    print("\n" + "="*50)
    print("All requested tests completed successfully!")
    print("="*50)

if __name__ == '__main__':
    main()
