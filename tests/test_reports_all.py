import os
from pathlib import Path
import subprocess
import sys
import unittest


class TestAllReportsRegression(unittest.TestCase):
    """Regression test for `python main.py all` report outputs.

    中文说明:
    - 调用方 / Called by: `python -m unittest`、`python main.py unit`
    - 调用对象 / Calls: 子进程执行 `python main.py all`（带环境变量 DSRA_FAST_ALL）
    - 作用 / Purpose: 直接校验 `reports/run_summary.md` 与 `reports/all_output.txt` 会被生成
    - 变量 / Variables:
      `env` 子进程环境, `reports_dir` 报告目录, `run_summary`/`all_output` 期望产物
    - 接入 / Integration: 放置于 `tests/`，由 `unittest discover` 自动发现
    - 错误处理 / Error handling: 子进程非零返回码或文件缺失即失败
    - 关键词 / Keywords:
      reports|run_summary|all_output|main.py all|regression|unittest|subprocess|fast_all|artifact|pipeline
    """

    def test_main_all_generates_run_summary_and_all_output(self):
        """Validate that `python main.py all` generates expected report files.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `python main.py all`
        - 作用 / Purpose: 覆盖 all 分支的日志 tee 与 run_summary 写入路径
        - 变量 / Variables:
          `DSRA_FAST_ALL` 用于让 all 分支只跑最小套件以加快回归测试
        - 接入 / Integration: 可作为 CI 的基础报告产物检查
        - 错误处理 / Error handling: 断言返回码与文件存在性/非空性
        - 关键词 / Keywords:
          all|tee|reports_dir|file_exists|non_empty|fast|env|summary|log|artifact
        """
        project_root = Path(__file__).resolve().parents[1]
        reports_dir = project_root / "reports"
        run_summary = reports_dir / "run_summary.md"
        all_output = reports_dir / "all_output.txt"

        if run_summary.exists():
            run_summary.unlink()
        if all_output.exists():
            all_output.unlink()

        env = os.environ.copy()
        env["DSRA_FAST_ALL"] = "1"
        result = subprocess.run(
            [sys.executable, "main.py", "all"],
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}",
        )
        self.assertTrue(run_summary.exists())
        self.assertTrue(all_output.exists())
        self.assertGreater(run_summary.stat().st_size, 0)
        self.assertGreater(all_output.stat().st_size, 0)

    def test_main_report_generates_run_summary(self):
        """Validate that `python main.py report` generates run_summary.md.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `python main.py report`
        - 作用 / Purpose: 覆盖 report 分支的 run_summary 写入路径（不跑其他 suite）
        - 变量 / Variables:
          `run_summary` 期望产物
        - 接入 / Integration: 可作为 CI 的轻量报告产物检查
        - 错误处理 / Error handling: 断言返回码与文件存在性/非空性/关键标题
        - 关键词 / Keywords:
          report|run_summary|lightweight|artifact|unittest|subprocess|reports_dir|markdown|pipeline|regression
        """
        project_root = Path(__file__).resolve().parents[1]
        reports_dir = project_root / "reports"
        run_summary = reports_dir / "run_summary.md"

        if run_summary.exists():
            run_summary.unlink()

        result = subprocess.run(
            [sys.executable, "main.py", "report"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}",
        )
        self.assertTrue(run_summary.exists())
        self.assertGreater(run_summary.stat().st_size, 0)
        content = run_summary.read_text(encoding="utf-8")
        self.assertIn("# DSRA Unified Test Report", content)


if __name__ == "__main__":
    unittest.main()
