import subprocess
import sys
import unittest
from pathlib import Path


class TestScriptsMainEntry(unittest.TestCase):
    """Regression test for running `scripts/main.py` directly.

    中文说明:
    - 调用方 / Called by: `python -m unittest`、`python main.py unit`
    - 调用对象 / Calls: 子进程执行 `python scripts/main.py -h`
    - 作用 / Purpose: 防止 `scripts/main.py` 直跑时出现 `ModuleNotFoundError: No module named 'src'`
    - 变量 / Variables:
      `project_root` 项目根目录, `result` 子进程结果
    - 接入 / Integration: 放置于 `tests/`，由 `unittest discover` 自动发现
    - 错误处理 / Error handling: 非零返回码或输出缺失关键字即失败
    - 关键词 / Keywords:
      scripts.main|entrypoint|help|ModuleNotFoundError|sys.path|src|regression|unittest|subprocess|cli
    """

    def test_scripts_main_help_runs_without_module_not_found(self):
        """Validate `python scripts/main.py -h` works and prints DSRA_FAST_ALL help.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `python scripts/main.py -h`
        - 作用 / Purpose: 确保脚本入口可用，并且 `--help` 文案包含 DSRA_FAST_ALL 说明
        - 变量 / Variables:
          `stdout`/`stderr` 子进程输出
        - 接入 / Integration: 作为最轻量的 CLI 回归测试
        - 错误处理 / Error handling: 返回码非 0 或 stderr 含 ModuleNotFoundError 即失败
        - 关键词 / Keywords:
          help|epilog|DSRA_FAST_ALL|cli|entry|regression|subprocess|exit_code|stdout|stderr
        """
        project_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(project_root / "scripts" / "main.py"), "-h"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}",
        )
        self.assertNotIn("ModuleNotFoundError", result.stderr)
        self.assertIn("DSRA_FAST_ALL", result.stdout)
        self.assertIn("Machine-readable output (stdout):", result.stdout)
        self.assertIn("DSRA_REPORT_STATUS=ok", result.stdout)
        self.assertIn("DSRA_ALL_STATUS=ok", result.stdout)

    def test_scripts_main_report_generates_run_summary(self):
        """Validate `python scripts/main.py report` generates `reports/run_summary.md`.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `python scripts/main.py report`
        - 作用 / Purpose: 确保 scripts 入口能执行 report 分支并生成报告索引文件
        - 变量 / Variables:
          `project_root` 项目根目录
          `reports_dir` 报告目录
          `run_summary` 报告索引文件路径
          `result` 子进程执行结果
        - 接入 / Integration: 放置于 `tests/`，由 `unittest discover` 自动发现
        - 错误处理 / Error handling: 返回码非 0、stderr 含 ModuleNotFoundError、或文件未生成/为空均视为失败
        - 关键词 / Keywords:
          report|run_summary.md|reports|cli|entrypoint|regression|subprocess|ModuleNotFoundError|exit_code|file_generation
        """
        project_root = Path(__file__).resolve().parents[1]
        reports_dir = project_root / "reports"
        run_summary = reports_dir / "run_summary.md"
        if run_summary.exists():
            run_summary.unlink()

        result = subprocess.run(
            [sys.executable, str(project_root / "scripts" / "main.py"), "report"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}",
        )
        self.assertNotIn("ModuleNotFoundError", result.stderr)
        self.assertIn("DSRA_REPORT_STATUS=ok", result.stdout)
        self.assertIn("DSRA_REPORT_RUN_SUMMARY=reports/run_summary.md", result.stdout)
        self.assertIn("DSRA_REPORT_EXECUTED_SUITES=0", result.stdout)
        self.assertTrue(run_summary.exists())
        self.assertGreater(run_summary.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
