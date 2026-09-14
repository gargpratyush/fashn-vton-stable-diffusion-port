"""Explicit no-model test tier; integration scripts have separate fixture-driven entry points."""
from pathlib import Path
import argparse
import sys
import unittest


MODULES = (
    "test_format_code", "test_fashn_comparison_report", "test_fashn_q45_study",
    "test_fashn_runtime_weights", "test_fashn_trajectories",
    "test_fashn_portability", "test_fashn_recovery", "test_fashn_quality_workflow",
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", action="store_true", help="Also run torch-based oracle component tests")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    modules = (*MODULES, "test_fashn_trajectory_probe") if args.oracle else MODULES
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
