"""
Run all test modules: config, spend generation, impressions, revenue, pipeline.
Run from project root: python test.py
"""
import sys
from pathlib import Path

# Ensure project root is on path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def main():
    from tests import test_config
    from tests import test_spend_generation
    from tests import test_impressions_simulation
    from tests import test_revenue_simulation
    from tests import test_pipeline

    test_config.main()
    test_spend_generation.main()
    test_impressions_simulation.main()
    test_revenue_simulation.main()
    test_pipeline.main()

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
