"""
Test Runner Script
=================
Runs all unit tests for the COVID-19 Warning System.

Usage:
    python tests/run_tests.py              # Run all tests
    python tests/run_tests.py -v           # Verbose output
    python tests/run_tests.py data         # Run data tests only
    python tests/run_tests.py model        # Run model tests only
    python tests/run_tests.py pipeline     # Run pipeline tests only
    python tests/run_tests.py app          # Run app tests only
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_all_tests(verbosity=2):
    """Run all unit tests"""
    loader = unittest.TestLoader()
    suite = loader.discover(str(Path(__file__).parent), pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return result.wasSuccessful()


def run_specific_tests(test_module, verbosity=2):
    """Run tests from a specific module"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(f'test_{test_module}')
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return result.wasSuccessful()


def main():
    """Main test runner"""
    print("=" * 80)
    print("COVID-19 WARNING SYSTEM - UNIT TESTS")
    print("=" * 80)
    print()
    
    # Parse arguments
    verbosity = 2 if '-v' in sys.argv else 1
    
    # Determine which tests to run
    if len(sys.argv) > 1 and sys.argv[1] not in ['-v']:
        test_type = sys.argv[1]
        
        test_map = {
            'data': 'data_preparation',
            'model': 'model_training',
            'pipeline': 'pipeline',
            'app': 'app'
        }
        
        if test_type in test_map:
            print(f"Running {test_type} tests...\n")
            success = run_specific_tests(test_map[test_type], verbosity)
        else:
            print(f"Unknown test type: {test_type}")
            print("Valid options: data, model, pipeline, app")
            return 1
    else:
        print("Running all tests...\n")
        success = run_all_tests(verbosity)
    
    print("\n" + "=" * 80)
    if success:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 80)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
