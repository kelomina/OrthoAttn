import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestMHDSRA2Smoke(unittest.TestCase):
    def run_command(self, *args):
        completed = subprocess.run(
            [sys.executable, *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return completed

    def test_verify_script_runs_successfully(self):
        result = self.run_command("scripts/verify_mhdsra2.py")
        output = result.stdout + result.stderr

        self.assertEqual(
            result.returncode,
            0,
            msg=f"verify_mhdsra2.py exited with {result.returncode}\n{output}",
        )
        self.assertIn("[OK] smoke test passed", output)
        self.assertIn("Estimated GPU attention working-set memory", output)

    def test_main_entry_runs_mhdsra2_suite(self):
        result = self.run_command("main.py", "mhdsra2")
        output = result.stdout + result.stderr

        self.assertEqual(
            result.returncode,
            0,
            msg=f"main.py mhdsra2 exited with {result.returncode}\n{output}",
        )
        self.assertIn("Running MHDSRA2 Verification", output)
        self.assertIn("All requested tests completed successfully!", output)


if __name__ == "__main__":
    unittest.main()
