from __future__ import annotations

import sys
import types
import zipfile
from pathlib import Path

import pytest
import torch

from scripts.audit_installed_packages_osv import (
    build_osv_query_batch,
    summarize_osv_response,
)
from scripts.json_retrieval_test import load_json_retrieval_case, resolve_project_json_file
from scripts.needle_in_haystack_test import (
    build_parser,
    resolve_niah_checkpoint_path,
    resolve_niah_reports_dir,
    sanitize_report_name,
    save_niah_verification_report,
)
from scripts.tiny_llama_shared import (
    WIKITEXT_HF_REVISION,
    WIKITEXT103_REVISION,
    safe_extract_zip,
    validate_downloaded_archive,
)
from src.dsra import swanlab_utils


def _minimal_niah_report_result() -> dict:
    """Build the smallest valid NIAH result payload for report-writer tests.

    中文说明:
    - 调用方 / Called by: tests in this module.
    - 调用对象 / Calls: none.
    - 作用 / Purpose: 避免安全测试启动真实训练，同时满足 Markdown 报告字段要求。
    - 返回 / Returns: 可被 `save_niah_verification_report` 写入的最小结果 dict。
    - 错误处理 / Error handling: 无；字段缺失会在调用方测试中暴露。
    - 副作用 / Side effects: 无。
    """
    return {
        "config": {},
        "status": "success",
        "seq_len": 8,
        "best_accuracy": 1.0,
        "best_accuracy_step": 1,
        "passed_accuracy": True,
        "final_loss": 0.0,
        "elapsed_sec": 0.0,
        "peak_memory_allocated_mb": 0.0,
        "peak_memory_reserved_mb": 0.0,
        "parameter_count": 1,
        "device": "cpu",
        "cuda_device_name": None,
        "torch_version": "test",
        "torch_cuda_version": None,
        "steps_observed": [],
        "robust_evals_observed": [],
    }


def test_niah_report_name_rejects_path_traversal(tmp_path) -> None:
    """Protect NIAH reports from escaping the selected reports directory.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `save_niah_verification_report`, `sanitize_report_name`.
    - 作用 / Purpose: 防止 `--report-name ../x` 把 JSON/Markdown 写到 reports/ 外。
    - 变量 / Variables: `reports_dir` 是允许写入目录；`escape_name` 模拟恶意 CLI 输入。
    - 错误处理 / Error handling: 断言 `ValueError`，并确认逃逸文件没有被创建。
    - 副作用 / Side effects: 只在 pytest 临时目录内尝试写文件。
    """
    reports_dir = tmp_path / "reports"
    escape_name = "../reports_path_escape_probe"

    with pytest.raises(ValueError):
        save_niah_verification_report(_minimal_niah_report_result(), reports_dir, escape_name, "Probe")

    assert not (tmp_path / "reports_path_escape_probe.json").exists()
    assert not (tmp_path / "reports_path_escape_probe.md").exists()


def test_niah_report_name_allows_plain_filename(tmp_path) -> None:
    """Validate normal report stems still write JSON and Markdown under reports/.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `sanitize_report_name`, `save_niah_verification_report`.
    - 作用 / Purpose: 确认路径护栏不会误伤普通报告名前缀。
    - 变量 / Variables: `report_name` 是合法文件名前缀。
    - 错误处理 / Error handling: 写入失败或路径错误会触发断言失败。
    - 副作用 / Side effects: 在 pytest 临时目录写入两个报告文件。
    """
    report_name = "mhdsra2_security_probe"

    assert sanitize_report_name(report_name) == report_name
    paths = save_niah_verification_report(
        _minimal_niah_report_result(),
        tmp_path / "reports",
        report_name,
        "Probe",
        project_root=tmp_path,
    )

    json_path = Path(paths["json"])
    markdown_path = Path(paths["markdown"])
    assert json_path.parent.name == "reports"
    assert markdown_path.parent.name == "reports"
    assert json_path.name == "mhdsra2_security_probe.json"
    assert markdown_path.name == "mhdsra2_security_probe.md"


def test_niah_reports_dir_rejects_project_escape(tmp_path) -> None:
    """Ensure NIAH report output cannot be redirected outside project reports/.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `save_niah_verification_report`, `resolve_niah_reports_dir`.
    - 作用 / Purpose: 防止 `--reports-dir ../outside` 把报告写到项目根目录外。
    - 变量 / Variables: `project_root` 是测试项目根；`outside_dir` 是恶意外部目录。
    - 错误处理 / Error handling: 断言 `ValueError`，并确认外部 reports 目录没有创建。
    - 副作用 / Side effects: 只在 pytest 临时目录内解析路径。
    """
    project_root = tmp_path / "project"
    outside_dir = tmp_path / "outside"

    with pytest.raises(ValueError):
        save_niah_verification_report(
            _minimal_niah_report_result(),
            outside_dir,
            "mhdsra2_security_probe",
            "Probe",
            project_root=project_root,
        )

    assert not (outside_dir / "reports").exists()


def test_niah_reports_dir_accepts_project_reports(tmp_path) -> None:
    """Validate project root and project reports both resolve to project reports/.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `resolve_niah_reports_dir`.
    - 作用 / Purpose: 保留正常 `reports` 和项目根输出用法。
    - 变量 / Variables: `project_root` 是测试项目根目录。
    - 错误处理 / Error handling: 路径解析错误会触发断言失败。
    - 副作用 / Side effects: 在 pytest 临时目录创建 project/reports。
    """
    project_root = tmp_path / "project"
    expected_reports = (project_root / "reports").resolve()

    assert resolve_niah_reports_dir("reports", project_root=project_root) == expected_reports
    assert resolve_niah_reports_dir(project_root, project_root=project_root) == expected_reports


def test_niah_checkpoint_path_stays_under_project_reports(tmp_path) -> None:
    """Ensure NIAH checkpoint paths cannot write outside reports/checkpoints/.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `resolve_niah_checkpoint_path`.
    - 作用 / Purpose: 防止 `--save-checkpoint ../x.pt` 写到项目目录外。
    - 变量 / Variables: `project_root` 是测试项目根；`checkpoint_dir` 是允许目录。
    - 错误处理 / Error handling: 路径逃逸或非法后缀触发 `ValueError`。
    - 副作用 / Side effects: 可创建 pytest 临时目录下的 checkpoints 目录。
    """
    project_root = tmp_path / "project"
    checkpoint_dir = project_root / "reports" / "checkpoints"

    resolved = resolve_niah_checkpoint_path(
        "best.pt",
        project_root=project_root,
        create_parent=True,
    )

    assert resolved == (checkpoint_dir / "best.pt").resolve()
    assert checkpoint_dir.exists()
    with pytest.raises(ValueError):
        resolve_niah_checkpoint_path("../best.pt", project_root=project_root)
    with pytest.raises(ValueError):
        resolve_niah_checkpoint_path("best.txt", project_root=project_root)


def test_json_retrieval_case_paths_stay_inside_project_root(tmp_path) -> None:
    """Ensure JSON retrieval inputs cannot read files outside the project root.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `resolve_project_json_file`, `load_json_retrieval_case`.
    - 作用 / Purpose: 防止 `input_path` 或 `metadata_path` 通过路径穿越读取项目外 JSON。
    - 变量 / Variables: `project_root` 是临时项目根；`fixtures` 是合法测试夹具目录。
    - 错误处理 / Error handling: 项目外路径和非 JSON 后缀触发 `ValueError`。
    - 副作用 / Side effects: 在 pytest 临时目录写入小型 JSON 夹具。
    """
    project_root = tmp_path / "project"
    fixtures = project_root / "tests" / "fixtures"
    fixtures.mkdir(parents=True)
    input_file = fixtures / "input.json"
    metadata_file = fixtures / "metadata.json"
    input_file.write_text("[65, 66]", encoding="utf-8")
    metadata_file.write_text(
        '{"expected_answer_bytes": [66], "question": "Q"}',
        encoding="utf-8",
    )

    case = load_json_retrieval_case(
        "tests/fixtures/input.json",
        "tests/fixtures/metadata.json",
        project_root=project_root,
    )

    assert case["sample_bytes"] == b"AB"
    assert case["expected_answer_bytes"] == b"B"
    assert resolve_project_json_file(input_file, project_root=project_root) == input_file.resolve()
    with pytest.raises(ValueError):
        resolve_project_json_file(tmp_path / "outside.json", project_root=project_root)
    with pytest.raises(ValueError):
        resolve_project_json_file("tests/fixtures/input.txt", project_root=project_root)


def test_niah_parser_defaults_swanlab_to_disabled() -> None:
    """Ensure the NIAH CLI does not upload metrics unless explicitly requested.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `build_parser`.
    - 作用 / Purpose: 保护 `verify-2m` 和 `benchmark-scale` 默认不向 SwanLab cloud 上传。
    - 变量 / Variables: `args` 是解析出的 CLI 参数。
    - 错误处理 / Error handling: 默认值变回 cloud 会触发断言失败。
    - 副作用 / Side effects: 无。
    """
    args = build_parser().parse_args(["verify-2m"])

    assert args.swanlab_mode == "disabled"


def test_init_swanlab_default_does_not_call_swanlab_init(monkeypatch) -> None:
    """Ensure `init_swanlab()` has a privacy-safe default mode.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `src.dsra.swanlab_utils.init_swanlab`.
    - 作用 / Purpose: 防止省略 mode 时默认创建 SwanLab cloud run。
    - 变量 / Variables: `called` 记录底层 `swanlab.init` 是否被触发。
    - 错误处理 / Error handling: 如果默认模式触发外部初始化，断言会失败。
    - 副作用 / Side effects: monkeypatch 只作用于本测试进程。
    """
    called = {"init": False}

    def fake_init(**_kwargs):
        called["init"] = True
        raise AssertionError("swanlab.init should not be called by default")

    monkeypatch.setattr(swanlab_utils.swanlab, "init", fake_init)

    run = swanlab_utils.init_swanlab()

    assert not run.enabled
    assert not called["init"]


def test_init_swanlab_rejects_unknown_mode() -> None:
    """Validate invalid SwanLab modes fail before any upload attempt.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `src.dsra.swanlab_utils.init_swanlab`.
    - 作用 / Purpose: 防止拼写错误或未知 mode 被直接传给 SwanLab SDK。
    - 变量 / Variables: `mode` 是非法模式名。
    - 错误处理 / Error handling: 断言抛出 `ValueError`。
    - 副作用 / Side effects: 无。
    """
    with pytest.raises(ValueError):
        swanlab_utils.init_swanlab(mode="public")


def test_device_resolvers_use_cuda_zero_policy() -> None:
    """Validate auto/cuda aliases resolve to cuda:0 and reject other CUDA indexes.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: tiny LLaMA, comparison, and arithmetic device resolvers.
    - 作用 / Purpose: 防止项目入口返回裸 `cuda` 或接受 `cuda:1`。
    - 变量 / Variables: `resolver` 是一个设备解析函数。
    - 错误处理 / Error handling: `cuda:1` 必须抛出 `ValueError`。
    - 副作用 / Side effects: 不分配 CUDA 张量，只构造 `torch.device`。
    """
    from scripts.compare_mhdsra2_vs_dsra import _resolve_device as compare_resolve_device
    from scripts.tiny_llama_shared import resolve_device as tiny_resolve_device
    from src.dsra.application.arithmetic_emergence_service import resolve_torch_device

    for resolver in (tiny_resolve_device, compare_resolve_device, resolve_torch_device):
        assert resolver("cpu") == torch.device("cpu")
        assert resolver("cuda") == torch.device("cuda:0")
        with pytest.raises(ValueError):
            resolver("cuda:1")


def test_scripts_do_not_default_swanlab_to_cloud() -> None:
    """Statically guard scripts from reintroducing implicit SwanLab cloud logging.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `Path.read_text`.
    - 作用 / Purpose: 防止 `scripts/` 或 SwanLab 工具默认值重新写回 cloud。
    - 变量 / Variables: `forbidden_patterns` 是会导致默认上传的源码片段。
    - 错误处理 / Error handling: 命中任何片段时断言失败并列出文件。
    - 副作用 / Side effects: 只读源码文件。
    """
    project_root = Path(__file__).resolve().parents[1]
    checked_paths = list((project_root / "scripts").glob("*.py")) + [
        project_root / "src" / "dsra" / "swanlab_utils.py",
    ]
    forbidden_patterns = (
        'mode="cloud"',
        "mode='cloud'",
        'default="cloud"',
        "default='cloud'",
        'swanlab_mode: str = "cloud"',
        "swanlab_mode: str = 'cloud'",
    )

    matches = []
    for path in checked_paths:
        source = path.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            if pattern in source:
                matches.append(f"{path.relative_to(project_root)} contains {pattern}")

    assert matches == []


def test_wikitext_zip_rejects_path_traversal(tmp_path) -> None:
    """Ensure WikiText ZIP extraction cannot write outside the data directory.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `scripts.tiny_llama_shared.safe_extract_zip`.
    - 作用 / Purpose: 防止 WikiText-2 下载包里的 `../x` 条目触发 Zip Slip。
    - 变量 / Variables: `archive_path` 是恶意 ZIP；`extract_dir` 是允许目录。
    - 错误处理 / Error handling: 路径逃逸必须抛 `ValueError`。
    - 副作用 / Side effects: 只在 pytest 临时目录创建 ZIP。
    """
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as zip_file:
        zip_file.writestr("../escape.txt", "bad")

    with pytest.raises(ValueError):
        safe_extract_zip(
            archive_path,
            tmp_path / "data",
            max_total_bytes=1024,
            max_member_bytes=1024,
        )
    assert not (tmp_path / "escape.txt").exists()


def test_wikitext_zip_enforces_extracted_size_limit(tmp_path) -> None:
    """Ensure WikiText ZIP extraction rejects oversized payload metadata.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `scripts.tiny_llama_shared.safe_extract_zip`.
    - 作用 / Purpose: 防止小压缩包解出超预算文本，降低压缩炸弹风险。
    - 变量 / Variables: `max_total_bytes` 故意小于 ZIP 条目解压大小。
    - 错误处理 / Error handling: 超出总大小上限必须抛 `ValueError`。
    - 副作用 / Side effects: 只在 pytest 临时目录创建 ZIP。
    """
    archive_path = tmp_path / "large.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zip_file:
        zip_file.writestr("wikitext-2/wiki.train.tokens", "a" * 32)

    with pytest.raises(ValueError):
        safe_extract_zip(
            archive_path,
            tmp_path / "data",
            max_total_bytes=16,
            max_member_bytes=1024,
        )


def test_wikitext_zip_enforces_member_count_limit(tmp_path) -> None:
    """Ensure WikiText ZIP extraction rejects too many archive entries.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `scripts.tiny_llama_shared.safe_extract_zip`.
    - 作用 / Purpose: 防止 ZIP 里塞入海量小文件造成目录和 inode 资源消耗。
    - 变量 / Variables: `max_members` 故意小于 ZIP 条目数量。
    - 错误处理 / Error handling: 超出条目数量上限必须抛 `ValueError`。
    - 副作用 / Side effects: 只在 pytest 临时目录创建 ZIP。
    """
    archive_path = tmp_path / "many.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zip_file:
        zip_file.writestr("wikitext-2/a.tokens", "a")
        zip_file.writestr("wikitext-2/b.tokens", "b")

    with pytest.raises(ValueError):
        safe_extract_zip(
            archive_path,
            tmp_path / "data",
            max_total_bytes=1024,
            max_member_bytes=1024,
            max_members=1,
        )


def test_wikitext_archive_hash_mismatch_is_rejected(tmp_path) -> None:
    """Ensure WikiText downloads must match the pinned archive hash.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `scripts.tiny_llama_shared.validate_downloaded_archive`.
    - 作用 / Purpose: 防止下载到错误 XML、损坏 ZIP 或被篡改 ZIP 后继续解压。
    - 变量 / Variables: `expected_sha256` 故意设置为错误值。
    - 错误处理 / Error handling: 摘要不符必须抛 `RuntimeError`。
    - 副作用 / Side effects: 只在 pytest 临时目录创建 ZIP。
    """
    archive_path = tmp_path / "wikitext-2-v1.zip"
    with zipfile.ZipFile(archive_path, "w") as zip_file:
        zip_file.writestr("wikitext-2/wiki.train.tokens", "sample")

    with pytest.raises(RuntimeError):
        validate_downloaded_archive(
            archive_path,
            expected_sha256="0" * 64,
            max_archive_bytes=1024,
            label="WikiText-2",
        )


def test_wikitext2_download_uses_pinned_revision(monkeypatch, tmp_path) -> None:
    """Ensure WikiText-2 HuggingFace downloads pin a concrete dataset revision.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `scripts.tiny_llama_shared.download_wikitext2`.
    - 作用 / Purpose: 防止 WikiText-2 默认分支变动改变 tiny LLaMA PPL 实验数据。
    - 变量 / Variables: `fake_load_dataset` 记录传给 datasets 的 revision。
    - 错误处理 / Error handling: 缺少固定 revision 会触发断言失败。
    - 副作用 / Side effects: 在 pytest 临时目录写出三个小型 `.tokens` 文件。
    """
    from scripts.tiny_llama_shared import download_wikitext2

    calls = []

    class FakeSplit:
        def __init__(self, rows):
            self.rows = rows

        def __getitem__(self, key):
            if key != "text":
                raise KeyError(key)
            return self.rows

    def fake_load_dataset(name, config_name, *, revision):
        calls.append((name, config_name, revision))
        return {
            "train": FakeSplit(["train"]),
            "validation": FakeSplit(["valid"]),
            "test": FakeSplit(["test"]),
        }

    fake_datasets = types.SimpleNamespace(load_dataset=fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    train_path = download_wikitext2(str(tmp_path / "wikitext-2-cache"))

    assert calls == [("Salesforce/wikitext", "wikitext-2-raw-v1", WIKITEXT_HF_REVISION)]
    assert train_path.read_text(encoding="utf-8") == "train"
    assert (train_path.parent / "wiki.valid.tokens").read_text(encoding="utf-8") == "valid"
    assert (train_path.parent / "wiki.test.tokens").read_text(encoding="utf-8") == "test"


def test_wikitext103_download_uses_pinned_revision(monkeypatch, tmp_path) -> None:
    """Ensure WikiText-103 HuggingFace downloads pin a concrete dataset revision.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `scripts.tiny_llama_shared.download_wikitext103`.
    - 作用 / Purpose: 防止 HuggingFace 数据集默认分支变动改变 PPL 实验数据。
    - 变量 / Variables: `fake_load_dataset` 记录传给 datasets 的 revision。
    - 错误处理 / Error handling: 缺少固定 revision 会触发断言失败。
    - 副作用 / Side effects: 在 pytest 临时目录写出三个小型 `.tokens` 文件。
    """
    from scripts.tiny_llama_shared import download_wikitext103

    calls = []

    class FakeSplit:
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, key):
            if key != "text":
                raise KeyError(key)
            return self.rows

    def fake_load_dataset(name, config_name, *, revision):
        calls.append((name, config_name, revision))
        return {
            "train": FakeSplit(["train"]),
            "validation": FakeSplit(["valid"]),
            "test": FakeSplit(["test"]),
        }

    fake_datasets = types.SimpleNamespace(load_dataset=fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    result = download_wikitext103(str(tmp_path / "wikitext-103-cache"))

    assert calls == [("Salesforce/wikitext", "wikitext-103-raw-v1", WIKITEXT103_REVISION)]
    assert result["train"].read_text(encoding="utf-8") == "train"
    assert result["valid"].read_text(encoding="utf-8") == "valid"
    assert result["test"].read_text(encoding="utf-8") == "test"


def test_osv_audit_builds_pypi_query_and_summary() -> None:
    """Validate the dependency OSV audit uses PyPI querybatch semantics.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `build_osv_query_batch`, `summarize_osv_response`.
    - 作用 / Purpose: 防止依赖审计入口漏掉 ecosystem/version 或打乱漏洞归属。
    - 变量 / Variables: `packages` 是两个模拟已安装包。
    - 错误处理 / Error handling: 查询格式或漏洞摘要不匹配会触发断言失败。
    - 副作用 / Side effects: 无网络调用。
    """
    packages = [{"name": "idna", "version": "3.14"}, {"name": "safe", "version": "1.0"}]

    payload = build_osv_query_batch(packages)
    vulnerable = summarize_osv_response(
        packages,
        {
            "results": [
                {"vulns": [{"id": "GHSA-65pc-fj4g-8rjx", "summary": "test"}]},
                {},
            ]
        },
    )

    assert payload == {
        "queries": [
            {"package": {"name": "idna", "ecosystem": "PyPI"}, "version": "3.14"},
            {"package": {"name": "safe", "ecosystem": "PyPI"}, "version": "1.0"},
        ]
    }
    assert vulnerable == [
        {
            "name": "idna",
            "version": "3.14",
            "vulnerabilities": [
                {"id": "GHSA-65pc-fj4g-8rjx", "summary": "test", "aliases": []}
            ],
        }
    ]


def test_osv_audit_rejects_truncated_results() -> None:
    """Ensure OSV querybatch response truncation cannot be reported as clean.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `scripts.audit_installed_packages_osv.summarize_osv_response`.
    - 作用 / Purpose: 防止 OSV 返回缺失或短 `results` 时误写 0 漏洞报告。
    - 变量 / Variables: `packages` 有两个查询项，但 OSV 响应只有一个结果。
    - 错误处理 / Error handling: 长度不匹配必须抛 `RuntimeError`。
    - 副作用 / Side effects: 无网络调用。
    """
    packages = [{"name": "idna", "version": "3.14"}, {"name": "pillow", "version": "12.1.1"}]

    with pytest.raises(RuntimeError):
        summarize_osv_response(packages, {"results": [{}]})


def test_osv_audit_rejects_missing_results() -> None:
    """Ensure malformed OSV querybatch responses fail closed.

    中文说明:
    - 调用方 / Called by: pytest.
    - 调用对象 / Calls: `scripts.audit_installed_packages_osv.summarize_osv_response`.
    - 作用 / Purpose: 防止缺少 `results` 字段的响应被当作无漏洞。
    - 变量 / Variables: `packages` 是一个模拟查询项。
    - 错误处理 / Error handling: 缺少 `results` 必须抛 `RuntimeError`。
    - 副作用 / Side effects: 无网络调用。
    """
    packages = [{"name": "idna", "version": "3.14"}]

    with pytest.raises(RuntimeError):
        summarize_osv_response(packages, {})
