import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import scripts.compare_mhdsra2_vs_dsra as compare_module


class TestMHDSRA2VsDSRACompare(unittest.TestCase):
    """Regression tests for the MHDSRA2 vs DSRA comparison runner.

    中文说明:
    - 调用方 / Called by: `python -m unittest`、`python main.py unit`
    - 调用对象 / Calls: `scripts.compare_mhdsra2_vs_dsra.main`, `python main.py mhdsra2_compare`
    - 作用 / Purpose: 校验对比脚本能生成标准报告产物，并能挂接到统一测试入口
    - 变量 / Variables:
      `reports_dir` 临时报告目录, `json_path/md_path` 产物路径, `result` 子进程执行结果
    - 接入 / Integration: 放置于 `tests/`，由 `unittest discover` 自动发现
    - 错误处理 / Error handling: 返回码非 0、报告缺失、JSON 结构异常均视为失败
    - 关键词 / Keywords:
      mhdsra2|dsra|compare|reports|json|markdown|main.py|subprocess|regression|benchmark
    """

    def test_compare_script_generates_reports_in_reports_directory(self):
        """Validate direct script entry writes comparison artifacts under `reports/`.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `scripts.compare_mhdsra2_vs_dsra.main`
        - 作用 / Purpose: 直接验证脚本主入口会在 `reports/` 目录下生成 json/md 双报告
        - 变量 / Variables:
          `tmp_dir` 临时目录, `reports_dir` 临时报告目录, `payload` 返回结果
        - 接入 / Integration: 作为最小冒烟测试，保证目录结构与结果载荷稳定
        - 错误处理 / Error handling: 通过文件存在、关键键名和结果数量断言捕获回归
        - 关键词 / Keywords:
          script|reports_dir|artifact|payload|json|markdown|smoke|compare|structure|回归
        """
        project_root = Path(__file__).resolve().parents[1]
        reports_dir = project_root / "reports" / "test_mhdsra2_vs_dsra_compare" / "reports"
        payload = compare_module.main(
            [
                "--seq-lengths",
                "8",
                "16",
                "--batch-size",
                "1",
                "--dim",
                "32",
                "--slots",
                "8",
                "--read-topk",
                "4",
                "--chunk-sizes",
                "8",
                "--warmup-runs",
                "1",
                "--repeat-runs",
                "2",
                "--device",
                "cpu",
                "--reports-dir",
                str(reports_dir),
            ]
        )
        json_path = reports_dir / "mhdsra2_vs_dsra_compare.json"
        md_path = reports_dir / "mhdsra2_vs_dsra_compare.md"

        self.assertTrue(json_path.exists())
        self.assertTrue(md_path.exists())
        self.assertEqual(len(payload["results"]), 2)
        self.assertEqual(payload["results"][0]["chunk_size"], 8)
        self.assertEqual(payload["results"][0]["slots"], 8)
        self.assertEqual(payload["results"][0]["read_topk"], 4)
        self.assertEqual(payload["results"][0]["batch_size"], 1)
        self.assertEqual(payload["results"][0]["dim"], 32)
        self.assertIn("elapsed_ms_std", payload["results"][0]["dsra"])
        self.assertIn("elapsed_ms_samples", payload["results"][0]["mhdsra2"])
        self.assertEqual(payload["config"]["warmup_runs"], 1)
        self.assertEqual(payload["config"]["repeat_runs"], 2)
        self.assertEqual(payload["config"]["batch_size"], [1])
        self.assertEqual(payload["config"]["dim"], [32])
        self.assertIn("summary", payload)
        self.assertEqual(payload["summary"]["overall"]["total_cases"], 2)
        self.assertEqual(len(payload["summary"]["grouped"]["seq_len"]), 2)
        self.assertEqual(len(payload["summary"]["grouped"]["batch_size"]), 1)
        self.assertEqual(len(payload["summary"]["grouped"]["dim"]), 1)
        self.assertIn("mhdsra2_min_state_overhead_case", payload["summary"])
        self.assertIn("mhdsra2_max_state_overhead_case", payload["summary"])

        payload_from_disk = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload_from_disk["results"][0]["dsra"]["model"], "dsra")
        self.assertEqual(payload_from_disk["results"][0]["mhdsra2"]["model"], "mhdsra2")
        self.assertIn("MHDSRA2 vs DSRA Comparison", md_path.read_text(encoding="utf-8"))
        self.assertIn("Automatic Summary", md_path.read_text(encoding="utf-8"))
        self.assertIn("DSRA std ms", md_path.read_text(encoding="utf-8"))
        self.assertIn("MHDSRA2/DSRA", md_path.read_text(encoding="utf-8"))
        self.assertIn("Batch Size", md_path.read_text(encoding="utf-8"))

    def test_main_entry_runs_compare_suite(self):
        """Validate `python main.py mhdsra2_compare` runs and writes project reports.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: 子进程执行 `python main.py mhdsra2_compare`
        - 作用 / Purpose: 保证统一测试入口已接入对比测试，并生成正式 `reports/` 产物
        - 变量 / Variables:
          `project_root` 项目根目录, `env` 启用快速模式的环境变量, `result` 子进程结果
        - 接入 / Integration: 作为 CLI 回归测试，覆盖 `scripts/main.py` 的 dispatch 分支
        - 错误处理 / Error handling: 非零返回码、输出缺失关键字、报告缺失即失败
        - 关键词 / Keywords:
          main.py|cli|dispatch|fast_compare|reports|compare|subprocess|artifact|entrypoint|回归
        """
        project_root = Path(__file__).resolve().parents[1]
        reports_dir = project_root / "reports"
        json_path = reports_dir / "mhdsra2_vs_dsra_compare.json"
        md_path = reports_dir / "mhdsra2_vs_dsra_compare.md"

        env = os.environ.copy()
        env["DSRA_FAST_COMPARE"] = "1"
        result = subprocess.run(
            [sys.executable, "main.py", "mhdsra2_compare"],
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr

        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}",
        )
        self.assertIn("Running MHDSRA2 vs DSRA Comparison", output)
        self.assertIn("All requested tests completed successfully!", output)
        self.assertTrue(json_path.exists())
        self.assertTrue(md_path.exists())


if __name__ == "__main__":
    unittest.main()
