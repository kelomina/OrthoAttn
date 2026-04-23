# DSRA (Decoupled Sparse Routing Attention)

这是一个基于"固定容量可微状态库 + 语义路由 + 正交增量更新"的流式长序列注意力机制的可行性验证与测试套件。

## 核心模块
- `dsra_layer.py`: DSRA 核心网络层（含因果掩码、衰减机制、正交投影更新、分块因果掩码）。
- `dsra_model.py`: 支持多层堆叠的完整 LLM 架构包装（MultiLayerDSRAModel）。

## 核心机制
- **正交增量更新（Orthogonal Update）**: 写入状态时将新信息投影到与已有状态正交的子空间，防止容量饱和。
- **指令旁路（Instruction Bypass）**: 由 Sigmoid 门控决定当前位置使用状态检索还是局部注意力，动态选择信息源。
- **位置编码模式（pe_mode）**:
  - `none`: 无位置编码
  - `rope`: Rotary Position Embedding
  - `alibi`: Attention with Linear Biases
  - `timestamps`: 基于时间戳的软衰减路由

## 测试套件
- `test_dsra_math.py`: 基础张量维度、正交性及极端边界条件单元测试。
- `test_llm_compatibility.py`: 自回归生成与增量解码（KV-Cache）对齐测试。
- `benchmark_complexity.py`: O(N) 性能与显存复杂度基准测试。
- `test_state_saturation.py`: 状态库饱和度及 $\lambda$ 衰减机制验证。
- `toy_task_associative_recall.py`: 关联召回玩具任务，使用结构化 key/value/noise 分离词表。
- `needle_in_haystack_test.py`: 极长上下文的大海捞针召回能力测试。
- `ablation_study.py`: 核心机制（正交更新、指令旁路、位置编码）消融实验。
- `main.py`: 统一测试入口，支持 `unit`、`benchmark`、`saturation`、`recall`、`needle`、`ablation` 及 `all`。

## 运行方式
```bash
# 单元测试
python main.py unit

# 关联召回玩具任务
python main.py recall

# 大海捞针测试
python main.py needle

# 消融实验
python main.py ablation

# 全部测试
python main.py all
```
