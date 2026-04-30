"""Application model factory for retrieval experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch.nn as nn

from ..domain import RetrievalModelSpec

RetrievalModelBuilder = Callable[[RetrievalModelSpec], nn.Module]


class RetrievalModelFactory:
    """Build retrieval models from a domain specification and registered builders.

    中文说明:
    - 调用方 / Called by: `scripts.json_retrieval_test.build_retrieval_model`
    - 调用对象 / Calls: registered builder callables
    - 作用 / Purpose: 将模型构建分派从脚本层移入应用层，统一处理归档别名和错误
    - 变量 / Variables:
      `builders` 是模型类型到构造函数的注册表, `spec` 是领域层模型规格
    - 接入 / Integration: 脚本层提供本地模型类注册表，应用层负责选择实际构造器
    - 错误处理 / Error handling: 未注册模型类型抛出 `ValueError`
    - 关键词 / Keywords:
      factory|retrieval|model|application|builders|mhdsra2|json|spec|registry|模型工厂
    """

    def __init__(self, builders: Mapping[str, RetrievalModelBuilder]) -> None:
        """Create a retrieval model factory.

        中文说明:
        - 调用方 / Called by: `scripts.json_retrieval_test.build_retrieval_model`
        - 调用对象 / Calls: `dict`
        - 作用 / Purpose: 固化当前可构建模型族注册表
        - 变量 / Variables: `builders` 是标准模型类型到构造器的映射
        - 接入 / Integration: 新模型族只需在脚本或组合根注册 builder
        - 错误处理 / Error handling: 空注册表允许创建，但实际构建时会抛错
        - 关键词 / Keywords:
          init|factory|builders|registry|application|retrieval|model|json|mhdsra2|初始化
        """
        self.builders = dict(builders)

    def build(self, spec: RetrievalModelSpec) -> nn.Module:
        """Build a model from the canonical model type in `spec`.

        中文说明:
        - 调用方 / Called by: `scripts.json_retrieval_test.build_retrieval_model`
        - 调用对象 / Calls: registered `RetrievalModelBuilder`
        - 作用 / Purpose: 根据领域规格创建实际模型实例
        - 变量 / Variables: `spec` 是包含归档别名归一化逻辑的模型规格
        - 接入 / Integration: 返回对象保持现有 PyTorch `nn.Module` 接口
        - 错误处理 / Error handling: 未注册类型抛出 `ValueError`
        - 关键词 / Keywords:
          build|factory|retrieval|model|spec|mhdsra2|registry|application|nn.Module|构建
        """
        builder = self.builders.get(spec.model_type)
        if builder is None:
            raise ValueError(f"Unsupported model_type: {spec.requested_model_type}")
        return builder(spec)
