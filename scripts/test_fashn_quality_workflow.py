from pathlib import Path
import unittest

import yaml


class WorkflowTests(unittest.TestCase):
    def test_main_and_all_pull_requests_run_real_gates(self):
        path = Path(__file__).resolve().parent.parent / ".github/workflows/fashn-quality.yml"
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        events = workflow.get("on", workflow.get(True))
        self.assertEqual(events["push"], {"branches": ["main"]})
        self.assertIn("pull_request", events)
        self.assertIsNone(events["pull_request"])
        self.assertEqual(workflow["permissions"], {"contents": "read"})
        cpu = workflow["jobs"]["cpu"]
        self.assertEqual(set(cpu["strategy"]["matrix"]["os"]), {"ubuntu-24.04", "windows-2022"})
        commands = "\n".join(step.get("run", "") for step in cpu["steps"])
        for required in ("-DSD_BUILD_TESTS=ON", "-DSD_FASHN_PREPROCESS=OFF", "--no-tests=error",
                         "run_fashn_fast_tests.py", "node --test", "clang-tidy-19", "-fsanitize=address,undefined"):
            self.assertIn(required, commands)


if __name__ == "__main__":
    unittest.main()
