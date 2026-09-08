"""Run the whole test suite.

    python tests/run_all.py

No test framework is needed - the project deliberately keeps its dependency
list to what the system itself uses.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import test_activity          # noqa: E402
import test_dataset           # noqa: E402
import test_demo              # noqa: E402
import test_logbook           # noqa: E402
import test_normalisation     # noqa: E402
import test_tracking          # noqa: E402

MODULES = [
    test_normalisation,
    test_activity,
    test_logbook,
    test_tracking,
    test_dataset,
    test_demo,
]


def main() -> int:
    print("=" * 70)
    print("  BAS Crew Activity Recognition - test suite")
    print("=" * 70)

    started = time.time()
    passed = failed = 0
    broken = []

    for module in MODULES:
        try:
            result = module.run()
        except Exception as exc:                      # a crashed suite is a failure
            failed += 1
            broken.append(module.__name__)
            print(f"  {module.__name__}\n    ERROR  {type(exc).__name__}: {exc}")
            continue
        result.report()
        passed += result.passed
        failed += result.failed
        print()

    elapsed = time.time() - started
    print("=" * 70)
    total = passed + failed
    if failed:
        print(f"  FAILED  {failed} of {total} checks  ({elapsed:.1f}s)")
        if broken:
            print(f"  suites that could not run: {', '.join(broken)}")
    else:
        print(f"  ALL PASS  {passed} checks  ({elapsed:.1f}s)")
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
