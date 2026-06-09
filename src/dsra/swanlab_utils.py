from __future__ import annotations

from typing import Any

import swanlab

DEFAULT_SWANLAB_MODE = "disabled"
ALLOWED_SWANLAB_MODES = {"cloud", "local", "offline", "disabled"}


def init_swanlab(
    project: str = "MHDSRA2",
    experiment_name: str | None = None,
    config: dict[str, Any] | None = None,
    mode: str = DEFAULT_SWANLAB_MODE,
    description: str | None = None,
    tags: list[str] | None = None,
) -> SwanLabRunProxy:
    """Create a SwanLab run only when the caller explicitly enables logging.

    中文说明:
    - 调用方 / Called by: training, benchmark, and diagnostic scripts that record metrics.
    - 调用对象 / Calls: `swanlab.init` only when `mode` is not `disabled`.
    - 作用 / Purpose: 统一 SwanLab 隐私默认值；默认不向外部服务上传实验配置或指标。
    - 参数 / Parameters: `mode` 必须是 cloud/local/offline/disabled 之一。
    - 返回 / Returns: `SwanLabRunProxy`; disabled 模式返回不可写代理。
    - 错误处理 / Error handling: 非法 mode 抛 `ValueError`; 初始化失败降级为 disabled。
    - 副作用 / Side effects: 只有显式 cloud/local/offline 时才创建 SwanLab run。

    English documentation:
    Function name:
        init_swanlab
    Purpose:
        Centralize SwanLab initialization with a privacy-safe disabled default.
    Called by:
        Experiment and benchmark scripts.
    Calls:
        `swanlab.init` when logging is explicitly enabled.
    Parameters:
        - mode: one of cloud/local/offline/disabled.
    Returns:
        `SwanLabRunProxy`.
    Error handling:
        Invalid modes raise `ValueError`; initialization failures fall back to disabled.
    Side effects:
        May create an external/local SwanLab run when explicitly enabled.
    """
    if mode not in ALLOWED_SWANLAB_MODES:
        raise ValueError(
            f"Invalid SwanLab mode: {mode!r}. "
            f"Expected one of {sorted(ALLOWED_SWANLAB_MODES)}."
        )
    if mode == "disabled":
        return SwanLabRunProxy(enabled=False)
    try:
        run = swanlab.init(
            project=project,
            experiment_name=experiment_name,
            config=config,
            mode=mode,
            description=description,
            tags=tags,
        )
        return SwanLabRunProxy(enabled=True, _run=run)
    except Exception as e:
        print(f"[SwanLab] 初始化失败: {e}")
        print("[SwanLab] 使用禁用模式继续训练（可以通过设置mode='disabled'跳过SwanLab）")
        return SwanLabRunProxy(enabled=False)


class SwanLabRunProxy:
    def __init__(self, enabled: bool = False, _run: Any = None) -> None:
        self.enabled = enabled
        self._run = _run

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled:
            return
        self._run.log(data, step=step)

    def finish(self) -> None:
        if not self.enabled:
            return
        self._run.finish()

    @property
    def id(self) -> str | None:
        if not self.enabled:
            return None
        return self._run.id
