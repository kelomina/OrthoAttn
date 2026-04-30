# DSRA Legacy Architecture Archive

## 中文说明

- 调用方 / Called by: 维护人员、迁移脚本和需要查看历史实现的开发者。
- 调用对象 / Calls: `archive/root_copies/dsra_layer copy.py`, `archive/root_copies/dsra_model copy.py`。
- 作用 / Purpose: 标记原始 DSRA 架构已经归档，当前运行时代码全面切换为 MHDSRA2。
- 变量 / Variables:
  `dsra` 是历史模型名，`mhdsra2` 是当前活动架构，`archive/root_copies/` 保存历史源码副本。
- 接入 / Integration:
  新开发应接入 `src/dsra/mhdsra2/`、`src/dsra/application/`、`src/dsra/domain/` 和 `src/dsra/infrastructure/`。
- 错误处理 / Error handling:
  归档文件不参与运行、测试或发布；如需恢复历史行为，应先复制到新分支并补齐测试。
- 关键词 / Keywords:
  dsra|legacy|archive|mhdsra2|migration|compatibility|domain|application|infrastructure|归档

## Status

Archived DSRA alias / MHDSRA2 is the active migration contract. Runtime code now
treats `dsra` as a compatibility alias for `mhdsra2`.

Historical copies:

- `archive/root_copies/dsra_layer copy.py`
- `archive/root_copies/dsra_model copy.py`

Active implementation:

- `src/dsra/mhdsra2/improved_dsra_mha.py`
- `src/dsra/mhdsra2/paged_exact_memory.py`
- `src/dsra/dsra_layer.py` as a compatibility adapter backed by MHDSRA2
- `src/dsra/dsra_model.py` as a multi-layer MHDSRA2 model with archived DSRA aliases
