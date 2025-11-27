#!/usr/bin/env python3
"""
Local Testing Script for Students

This script runs PUBLIC tests on your code and provides a summary.
Use this to check your progress before submission.

⚠️ WARNING: Passing all public tests does NOT guarantee full credit!
    - Public tests check BASIC functionality only
    - Hidden tests (used for grading) are MORE COMPREHENSIVE and STRICTER
    - Always test edge cases and read the assignment requirements carefully

Usage:
    python local_test.py              # Run all parts
    python local_test.py --part 1     # Run Part 1 only
    python local_test.py --part 2     # Run Part 2 only
    python local_test.py --part 3     # Run Part 3 only
    python local_test.py --part 4     # Run Part 4 only
    python local_test.py -v           # Verbose output
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_tests(part: int | None = None, verbose: bool = False) -> dict:
    """Run public tests and collect results."""
    
    # Map parts to test files
    test_files = {
        1: "public_tests/test_metrics.py",
        2: "public_tests/test_regression.py",
        3: "public_tests/test_logistic_softmax.py",
        4: "public_tests/test_decision_tree.py",
    }
    
    # Determine which tests to run
    if part is not None:
        if part not in test_files:
            print(f"❌ Error: Invalid part {part}. Must be 1, 2, 3, or 4.")
            sys.exit(1)
        tests_to_run = {part: test_files[part]}
    else:
        tests_to_run = test_files
    
    results = {}
    
    for part_num, test_file in tests_to_run.items():
        test_path = Path(__file__).parent / test_file
        
        if not test_path.exists():
            print(f"⚠️  Warning: Test file not found: {test_file}")
            results[part_num] = {"error": "Test file not found"}
            continue
        
        # Run pytest with PYTHONPATH set to include assignment directory and code subdirectory
        cmd = [sys.executable, "-m", "pytest", str(test_path), "-v" if verbose else "-q"]
        env = os.environ.copy()
        assignment_dir = Path(__file__).parent
        code_dir = assignment_dir / "code"
        # Add both assignment and code directories to PYTHONPATH
        pythonpath = os.pathsep.join([str(assignment_dir), str(code_dir)])
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = os.pathsep.join([env["PYTHONPATH"], pythonpath])
        else:
            env["PYTHONPATH"] = pythonpath
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent,
                env=env,
            )
            
            # Parse output
            output = result.stdout + result.stderr
            
            # Count passed/failed from pytest output
            # Look for summary line like "5 passed in 0.43s" or "2 failed, 3 passed"
            import re
            
            passed = 0
            failed = 0
            errors = 0
            
            # Try to find summary line
            summary_match = re.search(r'(\d+) passed', output)
            if summary_match:
                passed = int(summary_match.group(1))
            
            failed_match = re.search(r'(\d+) failed', output)
            if failed_match:
                failed = int(failed_match.group(1))
            
            error_match = re.search(r'(\d+) error', output)
            if error_match:
                errors = int(error_match.group(1))
            
            total = passed + failed + errors
            
            results[part_num] = {
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "total": total,
                "output": output,
                "returncode": result.returncode,
            }
            
        except Exception as e:
            results[part_num] = {"error": str(e)}
    
    return results


def print_summary(results: dict) -> None:
    """Print a summary of test results."""
    
    part_names = {
        1: "Evaluation Metrics",
        2: "Non-linear Regression",
        3: "Logistic & Softmax Regression",
        4: "Decision Tree Metrics",
    }
    
    print()
    print("=" * 70)
    print("PUBLIC TEST RESULTS")
    print("=" * 70)
    print()
    
    all_passed = True
    total_passed = 0
    total_tests = 0
    
    for part_num in sorted(results.keys()):
        result = results[part_num]
        part_name = part_names.get(part_num, f"Part {part_num}")
        
        print(f"Part {part_num}: {part_name}")
        print("-" * 70)
        
        if "error" in result:
            print(f"  ❌ Error: {result['error']}")
            all_passed = False
        else:
            passed = result["passed"]
            failed = result["failed"]
            errors = result["errors"]
            total = result["total"]
            
            total_passed += passed
            total_tests += total
            
            if failed > 0 or errors > 0:
                all_passed = False
                status = "❌ FAILED"
            else:
                status = "✅ PASSED"
            
            print(f"  {status}")
            print(f"  Tests: {passed}/{total} passed")
            
            if failed > 0:
                print(f"  Failed: {failed}")
            if errors > 0:
                print(f"  Errors: {errors}")
        
        print()
    
    print("=" * 70)
    print(f"OVERALL: {total_passed}/{total_tests} public tests passed")
    print("=" * 70)
    print()
    
    if all_passed and total_tests > 0:
        print("🎉 Congratulations! All public tests passed!")
        print()
        print("⚠️  IMPORTANT REMINDERS:")
        print("  • Public tests check BASIC functionality only")
        print("  • Hidden tests (for grading) are MORE COMPREHENSIVE")
        print("  • Make sure to:")
        print("    - Test edge cases yourself")
        print("    - Read all assignment requirements")
        print("    - Keep your code clean and well documented")
        print("    - Review your code for quality and efficiency")
        print()
    else:
        print("📝 Some tests failed. Review the output above and:")
        print("  • Read error messages carefully")
        print("  • Check function signatures match requirements")
        print("  • Test with the provided datasets")
        print("  • Review the assignment instructions")
        print()
    
    print("💡 TIP: Run with -v flag for detailed output:")
    print("   python local_test.py -v")
    print()


def print_verbose_output(results: dict) -> None:
    """Print detailed test output."""
    
    part_names = {
        1: "Evaluation Metrics",
        2: "Non-linear Regression",
        3: "Logistic & Softmax Regression",
        4: "Decision Tree Metrics",
    }
    
    print()
    print("=" * 70)
    print("DETAILED TEST OUTPUT")
    print("=" * 70)
    
    for part_num in sorted(results.keys()):
        result = results[part_num]
        part_name = part_names.get(part_num, f"Part {part_num}")
        
        print()
        print(f"Part {part_num}: {part_name}")
        print("-" * 70)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(result["output"])
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run public tests on your assignment code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--part",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run tests for a specific part (1=Metrics, 2=Regression, 3=Logistic/Softmax, 4=Decision Tree)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed test output",
    )
    
    args = parser.parse_args()
    
    print()
    print("=" * 70)
    print("MACHINE LEARNING FOUNDATIONS ASSIGNMENT - LOCAL TEST RUNNER")
    print("=" * 70)
    print()
    print("Running public tests on your code...")
    
    if args.part:
        part_names = {
            1: "Evaluation Metrics",
            2: "Non-linear Regression",
            3: "Logistic & Softmax Regression",
            4: "Decision Tree Metrics",
        }
        print(f"Testing Part {args.part}: {part_names[args.part]} only")
    else:
        print("Testing all parts...")
    
    print()
    
    # Run tests
    results = run_tests(part=args.part, verbose=args.verbose)
    
    # Print summary
    print_summary(results)
    
    # Print verbose output if requested
    if args.verbose:
        print_verbose_output(results)
    
    # Exit with appropriate code
    all_passed = all(
        r.get("failed", 0) == 0 and r.get("errors", 0) == 0 and "error" not in r
        for r in results.values()
    )
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

