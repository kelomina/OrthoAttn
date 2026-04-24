# 任务清单: 目录分层重构与兼容适配

目录: `helloagents/plan/202604250411_dir_structure_refactor/`

---

## 1. 目录重构
- [ ] 1.1 创建 `src/`、`scripts/`、`archive/` 基础目录，并建立正式源码与脚本的新位置，验证 why.md#需求-分层目录重构-场景-新开发者浏览仓库
- [ ] 1.2 迁移核心模块和实验脚本到新目录，并保持 `tests/` 目录继续可发现，依赖任务1.1

## 2. 兼容适配
- [ ] 2.1 在根目录保留 `main.py` 和历史模块名兼容入口，内部转发到新位置，验证 why.md#需求-兼容旧调用方式-场景-继续执行历史命令
- [ ] 2.2 更新 `pyproject.toml` 与 README 说明，使 `pytest`、`python main.py unit` 和旧导入方式继续可用，依赖任务2.1

## 3. 报告与归档
- [ ] 3.1 归档 `* copy.py`、`README copy.md`、`pyproject copy.toml`、`reports copy/` 和副本测试数据，验证 why.md#需求-规范报告与归档-场景-查看本次实验产物
- [ ] 3.2 规范 `reports/` 目录，消除可简化的重复嵌套层级，依赖任务3.1

## 4. 安全检查
- [ ] 4.1 检查兼容层是否仅做显式转发，确认未引入不受控路径修改或危险删除操作

## 5. 测试
- [ ] 5.1 执行 `pytest`
- [ ] 5.2 执行 `python main.py unit`
