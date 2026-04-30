"""Infrastructure repository for JSON retrieval report artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ..report_utils import ensure_reports_dir, write_json, write_markdown


class JsonRetrievalReportRepository:
    """Persist JSON retrieval report artifacts under the reports directory.

    中文说明:
    - 调用方 / Called by: `scripts.json_retrieval_test.save_json_retrieval_reports`,
      `scripts.json_retrieval_test.save_json_retrieval_generalization_reports`
    - 调用对象 / Calls: `ensure_reports_dir`, `write_json`, `write_markdown`
    - 作用 / Purpose: 将报告落盘职责从实验脚本移入基础设施层
    - 变量 / Variables: `reports_dir` 是报告根目录
    - 接入 / Integration: 构造后调用 `write_report` 输出 json/md 双产物
    - 错误处理 / Error handling: 文件系统错误和序列化错误向上抛出，不忽略报告失败
    - 关键词 / Keywords:
      repository|report|json_retrieval|infrastructure|write_json|markdown|artifact|reports|persist|报告
    """

    def __init__(self, reports_dir: Path) -> None:
        """Create a report repository rooted at `reports_dir`.

        中文说明:
        - 调用方 / Called by: JSON retrieval report save functions
        - 调用对象 / Calls: `ensure_reports_dir`
        - 作用 / Purpose: 初始化并确保报告目录存在
        - 变量 / Variables: `reports_dir` 是用户传入或默认报告目录
        - 接入 / Integration: 保存报告前创建一次仓储实例
        - 错误处理 / Error handling: 目录创建失败会向上抛出
        - 关键词 / Keywords:
          init|repository|reports_dir|ensure|json|markdown|infrastructure|artifact|path|初始化
        """
        self.reports_dir = ensure_reports_dir(reports_dir)

    def write_report(
        self,
        *,
        json_filename: str,
        markdown_filename: str,
        payload: Mapping[str, object],
        markdown_lines: Sequence[str],
    ) -> tuple[Path, Path]:
        """Write paired JSON and Markdown report artifacts.

        中文说明:
        - 调用方 / Called by: JSON retrieval report save functions
        - 调用对象 / Calls: `write_json`, `write_markdown`
        - 作用 / Purpose: 统一输出报告 JSON 负载与 Markdown 摘要
        - 变量 / Variables:
          `json_filename/markdown_filename` 是产物文件名, `payload` 是结构化结果,
          `markdown_lines` 是 Markdown 行列表
        - 接入 / Integration: 新报告类型复用本函数即可进入基础设施层
        - 错误处理 / Error handling: 写入失败直接向上抛出，避免静默丢报告
        - 关键词 / Keywords:
          write_report|json|markdown|payload|lines|repository|artifact|reports|persist|写入
        """
        json_path = self.reports_dir / json_filename
        markdown_path = self.reports_dir / markdown_filename
        write_json(json_path, dict(payload))
        write_markdown(markdown_path, list(markdown_lines))
        return json_path, markdown_path
