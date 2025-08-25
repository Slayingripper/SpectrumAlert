"""
Test runner for SpectrumAlert web interface tests
"""

import sys
import pytest
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_all_tests():
    """Run all web interface tests"""
    test_args = [
        "tests/web/",
        "-v",
        "--tb=short",
        "--color=yes",
        "-x",  # Stop on first failure
        "--disable-warnings"
    ]
    
    return pytest.main(test_args)


def run_frontend_tests():
    """Run only frontend tests"""
    test_args = [
        "tests/web/test_frontend.py",
        "-v",
        "--tb=short",
        "--color=yes"
    ]
    
    return pytest.main(test_args)


def run_backend_tests():
    """Run only backend tests"""
    test_args = [
        "tests/web/test_backend.py",
        "-v",
        "--tb=short",
        "--color=yes"
    ]
    
    return pytest.main(test_args)


def run_api_tests():
    """Run only API tests"""
    test_args = [
        "tests/web/test_app.py",
        "-v",
        "--tb=short",
        "--color=yes"
    ]
    
    return pytest.main(test_args)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SpectrumAlert web interface tests")
    parser.add_argument(
        "--suite",
        choices=["all", "frontend", "backend", "api"],
        default="all",
        help="Test suite to run"
    )
    
    args = parser.parse_args()
    
    if args.suite == "all":
        exit_code = run_all_tests()
    elif args.suite == "frontend":
        exit_code = run_frontend_tests()
    elif args.suite == "backend":
        exit_code = run_backend_tests()
    elif args.suite == "api":
        exit_code = run_api_tests()
    
    sys.exit(exit_code)
