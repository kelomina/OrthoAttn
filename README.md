# DSRA (Decoupled Sparse Routing Attention)

这是一个基于"固定容量可微状态库 + 语义路由 + 正交增量更新"的流式长序列注意力机制的可行性验证与测试套件。

## 目录结构
- `src/dsra/`: 正式源码目录，包含 DSRA 核心层、模型和报告工具。
- `scripts/`: 正式脚本目录，包含统一入口、基准、消融和检索实验。
- `tests/`: 单元测试目录，`tests/fixtures/` 存放测试夹具数据。
- `reports/`: 正式报告输出目录。
- `archive/`: 历史副本与旧报告归档目录。
- 根目录同名 `.py` 文件为兼容薄封装，继续支持历史导入方式与 `python main.py ...` 调用。

## 核心模块
- `src/dsra/dsra_layer.py`: DSRA 核心网络层（含因果掩码、衰减机制、正交投影更新、分块因果掩码）。
- `src/dsra/dsra_model.py`: 支持多层堆叠的完整 LLM 架构包装（`MultiLayerDSRAModel`）。

## 核心机制
- **正交增量更新（Orthogonal Update）**: 写入状态时将新信息投影到与已有状态正交的子空间，防止容量饱和。
- **指令旁路（Instruction Bypass）**: 由 Sigmoid 门控决定当前位置使用状态检索还是局部注意力，动态选择信息源。
- **位置编码模式（pe_mode）**:
  - `none`: 无位置编码
  - `rope`: Rotary Position Embedding
  - `alibi`: Attention with Linear Biases
  - `timestamps`: 基于时间戳的软衰减路由

## 测试套件
- `tests/test_dsra_math.py`: 基础张量维度、正交性及极端边界条件单元测试。
- `tests/test_llm_compatibility.py`: 自回归生成与增量解码（KV-Cache）对齐测试。
- `scripts/benchmark_complexity.py`: O(N) 性能与显存复杂度基准测试。
- `tests/test_state_saturation.py`: 状态库饱和度及 `lambda` 衰减机制验证。
- `scripts/toy_task_associative_recall.py`: 关联召回玩具任务，使用结构化 key/value/noise 分离词表。
- `scripts/needle_in_haystack_test.py`: 极长上下文的大海捞针召回能力测试。
- `scripts/ablation_study.py`: 核心机制（正交更新、指令旁路、位置编码）消融实验。
- `main.py`: 兼容统一入口，继续支持 `unit`、`benchmark`、`saturation`、`recall`、`needle`、`ablation` 及 `all`。

## 运行方式
```bash
# 单元测试
python main.py unit

# 直接运行 pytest
pytest

# 关联召回玩具任务
python main.py recall

# 大海捞针测试
python main.py needle

# 消融实验
python main.py ablation

# 全部测试
python main.py all
```

## 兼容说明
- 历史命令仍可继续使用，例如 `python main.py unit`。
- 历史导入仍可继续使用，例如 `from json_retrieval_test import run_json_retrieval_test`。
- 新开发建议优先从 `src/dsra/` 与 `scripts/` 引用正式实现。
