"""Audit installed Python packages against OSV."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


OSV_QUERYBATCH_URL = "https://api.osv.dev/v1/querybatch"


def collect_installed_packages() -> list[dict[str, str]]:
    """Collect installed distribution names and versions from the current Python.

    中文说明:
    - 调用方 / Called by: CLI `main`.
    - 调用对象 / Calls: `importlib.metadata.distributions`.
    - 作用 / Purpose: 读取当前 `.env` 里实际安装的 PyPI 包，作为 OSV 查询输入。
    - 返回 / Returns: `{"name": ..., "version": ...}` 字典列表。
    - 错误处理 / Error handling: 缺少 name/version 的发行包会跳过。
    - 副作用 / Side effects: 只读当前 Python 环境的包元数据。

    English documentation:
    Function name:
        collect_installed_packages
    Purpose:
        Collect installed PyPI distribution names and versions.
    Called by:
        CLI `main`.
    Calls:
        `importlib.metadata.distributions`.
    Parameters:
        None.
    Returns:
        Sorted package dictionaries with `name` and `version`.
    Error handling:
        Skips incomplete distribution metadata.
    Side effects:
        Reads local Python package metadata only.
    """
    packages = []
    seen = set()
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if not name or not version:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        packages.append({"name": name, "version": version})
    return sorted(packages, key=lambda item: item["name"].lower())


def build_osv_query_batch(packages: list[dict[str, str]]) -> dict[str, Any]:
    """Build an OSV querybatch payload for PyPI packages.

    中文说明:
    - 调用方 / Called by: `query_osv`.
    - 调用对象 / Calls: none.
    - 作用 / Purpose: 把本地包列表转换成 OSV API 需要的 PyPI 查询格式。
    - 参数 / Parameters: `packages` 是 name/version 字典列表。
    - 返回 / Returns: 可 JSON 序列化的 OSV querybatch payload。
    - 错误处理 / Error handling: 空 name/version 会被跳过。
    - 副作用 / Side effects: 无。

    English documentation:
    Function name:
        build_osv_query_batch
    Purpose:
        Convert package dictionaries into an OSV querybatch payload.
    Called by:
        `query_osv`.
    Calls:
        None.
    Parameters:
        - packages: dictionaries with package name and version.
    Returns:
        JSON-serializable OSV payload.
    Error handling:
        Skips entries without name or version.
    Side effects:
        None.
    """
    queries = []
    for package in packages:
        name = package.get("name")
        version = package.get("version")
        if not name or not version:
            continue
        queries.append({
            "package": {"name": name, "ecosystem": "PyPI"},
            "version": version,
        })
    return {"queries": queries}


def query_osv(packages: list[dict[str, str]]) -> dict[str, Any]:
    """Submit installed package versions to the OSV querybatch API.

    中文说明:
    - 调用方 / Called by: CLI `main`.
    - 调用对象 / Calls: `build_osv_query_batch`, `urllib.request.urlopen`.
    - 作用 / Purpose: 查询当前环境是否命中公开漏洞记录。
    - 参数 / Parameters: `packages` 是当前 Python 环境的包列表。
    - 返回 / Returns: OSV API JSON 响应。
    - 错误处理 / Error handling: 网络或 API 失败会抛 `RuntimeError`。
    - 副作用 / Side effects: 向 `api.osv.dev` 发送包名和版本。

    English documentation:
    Function name:
        query_osv
    Purpose:
        Query OSV for vulnerabilities in installed PyPI packages.
    Called by:
        CLI `main`.
    Calls:
        Payload builder and `urllib.request.urlopen`.
    Parameters:
        - packages: installed package dictionaries.
    Returns:
        Parsed OSV JSON response.
    Error handling:
        Wraps network/API errors in `RuntimeError`.
    Side effects:
        Sends package names and versions to OSV.
    """
    payload = json.dumps(build_osv_query_batch(packages)).encode("utf-8")
    request = urllib.request.Request(
        OSV_QUERYBATCH_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"OSV query failed: {exc}") from exc


def summarize_osv_response(
    packages: list[dict[str, str]],
    osv_response: dict[str, Any],
) -> list[dict[str, Any]]:
    """Pair OSV querybatch results with package names and versions.

    中文说明:
    - 调用方 / Called by: CLI `main` and tests.
    - 调用对象 / Calls: none.
    - 作用 / Purpose: 把 OSV 的批量响应还原成易读的 per-package 漏洞列表。
    - 参数 / Parameters: `packages` 是查询顺序；`osv_response` 是 OSV 返回 JSON。
    - 返回 / Returns: 命中漏洞的包列表，每项包含 name/version/vulnerabilities。
    - 错误处理 / Error handling: 缺失、类型错误或长度不匹配的 `results` 会抛 `RuntimeError`。
    - 副作用 / Side effects: 无。

    English documentation:
    Function name:
        summarize_osv_response
    Purpose:
        Pair OSV querybatch results with package metadata.
    Called by:
        CLI `main` and tests.
    Calls:
        None.
    Parameters:
        - packages: query package order.
        - osv_response: OSV JSON response.
    Returns:
        Vulnerable package summaries.
    Error handling:
        Raises `RuntimeError` when OSV omits, truncates, or malforms `results`.
    Side effects:
        None.
    """
    results = osv_response.get("results")
    if not isinstance(results, list):
        raise RuntimeError("OSV response missing list field: results")
    if len(results) != len(packages):
        raise RuntimeError(
            "OSV response result count mismatch: "
            f"expected {len(packages)}, got {len(results)}"
        )
    vulnerable = []
    for package, result in zip(packages, results):
        vulns = result.get("vulns", []) if isinstance(result, dict) else []
        if not vulns:
            continue
        vulnerable.append({
            "name": package["name"],
            "version": package["version"],
            "vulnerabilities": [
                {
                    "id": vuln.get("id"),
                    "summary": vuln.get("summary"),
                    "aliases": vuln.get("aliases", []),
                }
                for vuln in vulns
            ],
        })
    return vulnerable


def build_report(packages: list[dict[str, str]], vulnerable: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a stable JSON report for dependency audit results.

    中文说明:
    - 调用方 / Called by: CLI `main`.
    - 调用对象 / Calls: none.
    - 作用 / Purpose: 生成报告文件，记录查询包数、漏洞包数和具体命中项。
    - 参数 / Parameters: `packages` 是全部查询包；`vulnerable` 是漏洞命中列表。
    - 返回 / Returns: 可写入 JSON 的报告字典。
    - 错误处理 / Error handling: 无。
    - 副作用 / Side effects: 无。

    English documentation:
    Function name:
        build_report
    Purpose:
        Build a stable JSON report for the audit result.
    Called by:
        CLI `main`.
    Calls:
        None.
    Parameters:
        Queried package list and vulnerable package summaries.
    Returns:
        JSON-serializable report dictionary.
    Error handling:
        None.
    Side effects:
        None.
    """
    return {
        "ecosystem": "PyPI",
        "osv_endpoint": OSV_QUERYBATCH_URL,
        "package_count": len(packages),
        "vulnerable_package_count": len(vulnerable),
        "vulnerable_packages": vulnerable,
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the OSV audit CLI parser.

    中文说明:
    - 调用方 / Called by: CLI `main`.
    - 调用对象 / Calls: `argparse.ArgumentParser`.
    - 作用 / Purpose: 暴露报告路径和 fail-on-vulnerability 开关。
    - 返回 / Returns: 参数解析器。
    - 错误处理 / Error handling: 由 argparse 处理非法参数。
    - 副作用 / Side effects: 无。

    English documentation:
    Function name:
        build_parser
    Purpose:
        Build the dependency audit CLI parser.
    Called by:
        CLI `main`.
    Calls:
        `argparse.ArgumentParser`.
    Parameters:
        None.
    Returns:
        Parser instance.
    Error handling:
        Delegated to argparse.
    Side effects:
        None.
    """
    parser = argparse.ArgumentParser(
        description="Audit installed PyPI packages against OSV querybatch.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports") / "dependency_osv_audit.json",
        help="JSON report path.",
    )
    parser.add_argument(
        "--fail-on-vuln",
        action="store_true",
        help="Exit with status 1 when any installed package has OSV vulnerabilities.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the installed-package OSV audit.

    中文说明:
    - 调用方 / Called by: `python scripts/audit_installed_packages_osv.py`.
    - 调用对象 / Calls: parser, package collector, OSV query and report writer.
    - 作用 / Purpose: 给当前 `.env` 提供可重复的依赖漏洞审计入口。
    - 参数 / Parameters: `argv` 是测试可覆盖的参数列表。
    - 返回 / Returns: 进程退出码。
    - 错误处理 / Error handling: OSV 查询失败直接抛出；有漏洞且开启 fail 开关返回 1。
    - 副作用 / Side effects: 写入 JSON 报告；向 OSV 发送包名和版本。

    English documentation:
    Function name:
        main
    Purpose:
        Run the installed-package OSV audit and write a JSON report.
    Called by:
        Script entrypoint.
    Calls:
        Parser, package collector, OSV query and report writer.
    Parameters:
        - argv: optional CLI argument override.
    Returns:
        Process exit code.
    Error handling:
        Raises query failures; returns 1 for vulnerabilities when requested.
    Side effects:
        Writes a report file and sends package names/versions to OSV.
    """
    args = build_parser().parse_args(argv)
    packages = collect_installed_packages()
    osv_response = query_osv(packages)
    vulnerable = summarize_osv_response(packages, osv_response)
    report = build_report(packages, vulnerable)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Audited {len(packages)} packages against OSV.")
    print(f"Vulnerable packages: {len(vulnerable)}")
    print(f"Report written to {args.output}")
    if args.fail_on_vuln and vulnerable:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
