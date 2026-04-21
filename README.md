# DSRA (Decoupled Sparse Routing Attention)

这是一个基于“固定容量可微状态库 + 语义路由 + 正交增量更新”的流式长序列注意力机制的可行性验证与测试套件。

## 核心模块
- `dsra_layer.py`: DSRA 核心网络层（含因果掩码、衰减机制、正交投影更新）。
- `dsra_model.py`: 支持多层堆叠的完整 LLM 架构测试包装。

## 测试套件
- `test_dsra_math.py`: 基础张量维度、正交性及极端边界条件单元测试。
- `test_llm_compatibility.py`: 自回归生成与增量解码（KV-Cache）对齐测试。
- `benchmark_complexity.py`: O(N) 性能与显存复杂度基准测试。
- `test_state_saturation.py`: 状态库饱和度及 $\lambda$ 衰减机制验证。
- `toy_task_associative_recall.py`: 关联召回基础玩具任务。
- `needle_in_haystack_test.py`: 极长上下文的大海捞针召回能力测试。
- `ablation_study.py`: 核心机制（正交更新、指令旁路）消融实验。