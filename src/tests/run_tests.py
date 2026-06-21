"""Legacy test runner. Prefer: pytest src/tests"""

import pytest


def run_tests() -> int:
    return pytest.main(["src/tests", "-v", "--tb=short"])


if __name__ == "__main__":
    raise SystemExit(run_tests())
