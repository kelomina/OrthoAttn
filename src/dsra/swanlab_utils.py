from __future__ import annotations

from typing import Any

import swanlab


def init_swanlab(
    project: str = "MHDSRA2",
    experiment_name: str | None = None,
    config: dict[str, Any] | None = None,
    mode: str = "cloud",
    description: str | None = None,
    tags: list[str] | None = None,
) -> SwanLabRunProxy:
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
        print(f"[SwanLab] 使用禁用模式继续训练（可以通过设置mode='disabled'跳过SwanLab）")
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
