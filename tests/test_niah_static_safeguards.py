import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NIAH_SOURCE_PATH = PROJECT_ROOT / "scripts" / "needle_in_haystack_test.py"


def _load_niah_source():
    """Load the NIAH source file for static regression assertions.

    中文说明:
    - 调用方 / Called by: this static test module
    - 调用对象 / Calls: `Path.read_text`
    - 作用 / Purpose: 读取 `scripts/needle_in_haystack_test.py` 源码文本，供 AST 和文本级
      静态断言使用，避免运行 NIAH 动态训练
    - 参数 / Parameters: 无
    - 返回 / Returns: 源码字符串
    - 内部关键变量 / Internal variables: `NIAH_SOURCE_PATH` 指向被检查脚本
    - 接入 / Integration: 仅用于静态测试，不应作为业务代码入口
    - 错误处理 / Error handling: 文件缺失或编码错误由 Python 直接抛出
    - 副作用 / Side effects: 只读文件，不写 reports/，不训练模型
    - 事务边界 / Transaction boundary: 不涉及 Unit of Work、数据库或文件写事务
    - 并发与幂等 / Concurrency and idempotency: 同一源码内容下可重复调用
    - 关键词 / Keywords:
      static|source|needle|haystack|ast|test|no_training|read|guard|静态

    English documentation:
    Function name:
        _load_niah_source
    Purpose:
        Load the NIAH source text for static regression checks.
    Called by:
        Static tests in this module.
    Calls:
        `Path.read_text`.
    Parameters:
        None.
    Returns:
        Source text as a string.
    Error handling:
        Propagates normal file read errors.
    Side effects:
        Read-only filesystem access.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Repeatable for the same source file.
    English keywords:
        static, source, needle, haystack, ast, test, no_training, read, guard, regression
    """
    return NIAH_SOURCE_PATH.read_text(encoding="utf-8")


def _load_function_source(function_name):
    """Extract one function's source segment without importing the NIAH module.

    中文说明:
    - 调用方 / Called by: static regression tests in this module
    - 调用对象 / Calls: `_load_niah_source`, `ast.parse`, `ast.get_source_segment`
    - 作用 / Purpose: 对目标函数做静态结构检查，避免触发 torch import、模型构造或训练
    - 参数 / Parameters: `function_name` 是要查找的函数名
    - 返回 / Returns: 函数源码字符串
    - 内部关键变量 / Internal variables: `module_ast` 是源码 AST；`node` 是候选函数节点
    - 接入 / Integration: 仅用于 tests/ 下的静态防回归检查
    - 错误处理 / Error handling: 找不到函数时抛出 `AssertionError`
    - 副作用 / Side effects: 只读源码，不执行被测模块
    - 事务边界 / Transaction boundary: 不涉及 Unit of Work、数据库或文件写事务
    - 并发与幂等 / Concurrency and idempotency: 对同一源码内容结果稳定
    - 关键词 / Keywords:
      static|function|source_segment|ast|needle|no_import|guard|test|源码|函数

    English documentation:
    Function name:
        _load_function_source
    Purpose:
        Extract a function source segment without importing the NIAH module.
    Called by:
        Static regression tests in this module.
    Calls:
        `_load_niah_source`, `ast.parse`, and `ast.get_source_segment`.
    Parameters:
        - function_name: target function name.
    Returns:
        Function source text.
    Error handling:
        Raises `AssertionError` when the function is not found.
    Side effects:
        Read-only source inspection.
    Transaction boundary:
        Not applicable.
    Concurrency and idempotency:
        Repeatable for the same source file.
    English keywords:
        static, function, source_segment, ast, needle, no_import, guard, test, source, function
    """
    source = _load_niah_source()
    module_ast = ast.parse(source)
    for node in module_ast.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"Function not found: {function_name}")


def test_haystack_generation_uses_variable_needle_values_static():
    """Statically guard against fixed-answer NIAH samples.

    中文说明:
    - 调用方 / Called by: pytest static suite
    - 调用对象 / Calls: `_load_function_source`
    - 作用 / Purpose: 防止 `generate_haystack_with_needle` 回退到固定 `needle_v = 3`，
      该回退会让模型通过输出常量伪造检索能力
    - 错误处理 / Error handling: 源码结构回退时断言失败
    - 关键词 / Keywords:
      static|variable_value|needle_v|constant_answer|niah|haystack|test|accuracy|guard|固定答案
    """
    function_source = _load_function_source("generate_haystack_with_needle")

    assert "needle_values = torch.randint" in function_source
    assert "range_sizes" in function_source
    assert "query_key_positions = min_query_key_pos_per_sample" in function_source
    assert ".clamp(max=range_sizes - 1)" in function_source
    assert "Y[batch_indices, query_positions] = needle_values" in function_source
    assert "X[:, -1] = QUERY_TOKEN_ID" not in function_source
    assert "needle_v = 3" not in function_source


def test_query_position_extraction_checks_per_batch_counts_static():
    """Statically guard against ambiguous query-position extraction.

    中文说明:
    - 调用方 / Called by: pytest static suite
    - 调用对象 / Calls: `_load_function_source`
    - 作用 / Purpose: 防止查询总数刚好等于 batch size 但分布到错误样本时，未初始化
      query position 污染 accuracy/loss
    - 错误处理 / Error handling: 源码缺少 per-row count 检查时断言失败
    - 关键词 / Keywords:
      static|query_counts|bincount|duplicate_query|missing_query|niah|target|guard|test|查询
    """
    function_source = _load_function_source("_find_query_positions_or_final")

    assert "torch.bincount" in function_source
    assert "query_counts == 1" in function_source
    assert "return final_positions" in function_source


def test_legacy_full_logit_extractor_rejects_selected_logits_static():
    """Statically guard full-logit extraction against selected-logit misuse.

    中文说明:
    - 调用方 / Called by: pytest static suite
    - 调用对象 / Calls: `_load_function_source`
    - 作用 / Purpose: 确认 `extract_query_targets` 显式要求 `[batch, seq_len, vocab]`
      全序列 logits，避免未来把 `[batch, vocab]` selected logits 传入后产生难定位错误
    - 错误处理 / Error handling: 缺少维度检查时断言失败
    - 关键词 / Keywords:
      static|full_logits|selected_logits|shape_check|valueerror|niah|legacy|guard|test|维度
    """
    function_source = _load_function_source("extract_query_targets")

    assert "if logits.dim() != 3" in function_source
    assert "extract_query_targets expects full-sequence logits" in function_source


def test_verification_success_uses_current_step_loss_static():
    """Statically guard against best-accuracy-only pass reporting.

    中文说明:
    - 调用方 / Called by: pytest static suite
    - 调用对象 / Calls: `_load_function_source`
    - 作用 / Purpose: 防止 `run_niah_verification_case` 仅凭历史 best accuracy 标记通过；
      成功条件必须来自独立 eval 的 min-depth accuracy 与 eval loss
    - 错误处理 / Error handling: 成功条件回退到 `best_accuracy` 时断言失败
    - 关键词 / Keywords:
      static|success_criteria|best_accuracy|final_accuracy|loss|report|niah|guard|test|通过
    """
    function_source = _load_function_source("run_niah_verification_case")

    assert "robust_eval_min_depth_accuracy >= target_accuracy" in function_source
    assert "and robust_eval_mean_loss < stop_loss" in function_source
    assert "should_stop = best_accuracy >= target_accuracy" not in function_source
    assert '"passed_success_criteria": status == "success"' in function_source
    assert '"final_accuracy": final_accuracy' in function_source
    assert '"passed_accuracy": final_min_depth_accuracy >= target_accuracy' in function_source


def test_training_uses_independent_depth_evaluation_static():
    """Statically guard `run_single_niah_test` against train-batch best accuracy.

    中文说明:
    - 调用方 / Called by: pytest static suite
    - 调用对象 / Calls: `_load_function_source`
    - 作用 / Purpose: 确认 legacy NIAH sweep 默认返回独立 depth eval 的最终均值，而不是训练
      batch 上的幸运 `best_overall_acc` 或历史最高 eval 值
    - 错误处理 / Error handling: 源码回退到训练准确率返回时断言失败
    - 关键词 / Keywords:
      static|independent_eval|final_eval|train_accuracy|niah|sweep|return|guard|test|训练准确率
    """
    function_source = _load_function_source("run_single_niah_test")

    assert "evaluate_niah_depths(" in function_source
    assert "best_eval_mean_acc" in function_source
    assert "final_eval_mean_acc" in function_source
    assert "return metrics if return_metrics else final_eval_mean_acc" in function_source
    assert "return best_eval_mean_acc" not in function_source
    assert "return best_overall_acc" not in function_source


def test_depth_schedule_is_round_robin_static():
    """Statically guard against random depth sampling in NIAH loops.

    中文说明:
    - 调用方 / Called by: pytest static suite
    - 调用对象 / Calls: `_load_function_source`
    - 作用 / Purpose: 确认训练入口使用 deterministic round-robin depth schedule，而不是
      `random.choice` 导致 depth 覆盖不均和复现性变差
    - 错误处理 / Error handling: 发现随机 depth 选择时断言失败
    - 关键词 / Keywords:
      static|round_robin|random_choice|depth|schedule|reproducible|niah|guard|test|轮询
    """
    source = _load_niah_source()
    train_source = _load_function_source("run_single_niah_test")
    verification_source = _load_function_source("run_niah_verification_case")

    assert "def get_niah_depth_for_optimizer_step" in source
    assert "def get_niah_depth_for_epoch" in source
    assert "random.choice" not in train_source
    assert "random.choice" not in verification_source
    assert "get_niah_depth_for_epoch(epoch, depths_to_test)" in train_source
    assert "for optimizer_step in range(epochs)" in verification_source
    assert "get_niah_depth_for_optimizer_step(optimizer_step, depths_to_test)" in verification_source


def test_verification_logs_light_eval_every_optimizer_step_static():
    """Statically guard SwanLab logging against sparse train-only uploads.

    中文说明:
    - 调用方 / Called by: pytest static suite
    - 调用对象 / Calls: `_load_function_source`
    - 作用 / Purpose: 确认 verify-2m 每个 optimizer step 都执行 light eval，并在一次
      SwanLab payload 中上传 train、light eval 和 sample-level 指标
    - 错误处理 / Error handling: 回退到 `epoch % log_interval` 稀疏上传时断言失败
    - 关键词 / Keywords:
      static|swanlab|optimizer_step|light_eval|sample_metrics|log_interval|niah|guard|上传|指标
    """
    function_source = _load_function_source("run_niah_verification_case")

    assert "for optimizer_step in range(epochs)" in function_source
    assert "light_eval_result = evaluate_niah_depths(" in function_source
    assert '"train/loss": loss_value' in function_source
    assert 'add_niah_eval_metrics_to_swanlab_payload(' in function_source
    assert '"eval/light"' in function_source
    assert "include_samples=True" in function_source
    assert "swanlab_run.log(swanlab_payload, step=optimizer_step)" in function_source
    assert "epoch % log_interval" not in function_source


def test_robust_eval_waits_for_completed_interval_static():
    """Statically guard robust eval against blocking the initial step-zero upload.

    中文说明:
    - 调用方 / Called by: pytest static suite
    - 调用对象 / Calls: `_load_function_source`
    - 作用 / Purpose: 确认 robust eval 基于已完成 step 数触发，首次默认在完成第 N step 后执行，
      避免 step 0 先跑 96 个 2M batch
    - 错误处理 / Error handling: 回退到 `optimizer_step % interval == 0` 会触发断言失败
    - 关键词 / Keywords:
      static|robust_eval|completed_step|step_zero|interval|niah|guard|performance|评估|阻塞
    """
    function_source = _load_function_source("run_niah_verification_case")

    assert "completed_step_count = optimizer_step + 1" in function_source
    assert "should_run_robust_eval = completed_step_count % resolved_robust_eval_interval == 0" in function_source
    assert "optimizer_step % resolved_robust_eval_interval == 0" not in function_source


def test_capacity_probe_aggregates_all_depths_static():
    """Statically guard capacity accuracy against a single random depth.

    中文说明:
    - 调用方 / Called by: pytest static suite
    - 调用对象 / Calls: `_load_function_source`
    - 作用 / Purpose: 确认容量测试遍历 `NIAH_DEPTHS` 并返回 min-depth accuracy，避免单个
      随机 depth 的单 batch 结果被误读为代表性准确率
    - 错误处理 / Error handling: 容量测试回退为随机单 depth 时断言失败
    - 关键词 / Keywords:
      static|capacity|all_depths|min_depth_accuracy|random_depth|niah|accuracy|guard|test|容量
    """
    function_source = _load_function_source("run_single_niah_capacity_test")

    assert "for depth in NIAH_DEPTHS" in function_source
    assert "for _ in range(batches_per_depth)" in function_source
    assert "min_depth_accuracy" in function_source
    assert "random.choice" not in function_source


def test_oom_detection_uses_explicit_cuda_memory_condition_static():
    """Statically guard OOM message classification readability.

    中文说明:
    - 调用方 / Called by: pytest static suite
    - 调用对象 / Calls: `_load_function_source`
    - 作用 / Purpose: 确认 OOM 判断使用明确 OOM/allocator pattern，避免把普通
      `cuda` + `memory` 诊断文本误判为可跳过 OOM
    - 错误处理 / Error handling: 判断逻辑回退时断言失败
    - 关键词 / Keywords:
      static|oom|allocator|cuda|memory|runtimeerror|guard|test|错误|显存
    """
    function_source = _load_function_source("is_oom_error")

    assert "torch.cuda.OutOfMemoryError" in function_source
    assert "oom_patterns" in function_source
    assert '"cublas_status_alloc_failed"' in function_source
    assert '("cuda" in message and "memory" in message)' not in function_source


def test_legacy_sweeps_seed_and_report_final_eval_static():
    """Statically guard legacy NIAH sweeps against untraceable RNG and best-only reporting.

    中文说明:
    - 调用方 / Called by: pytest static suite
    - 调用对象 / Calls: `_load_function_source`
    - 作用 / Purpose: 防止 `run_niah_test` / `run_niah_capacity_test` 回退到未固定 seed，
      或把 final eval sweep 主指标重新改成 best eval
    - 错误处理 / Error handling: 源码缺少 seed 或 final eval 指标时断言失败
    - 关键词 / Keywords:
      static|seed|final_eval|legacy|sweep|capacity|niah|guard|test|随机
    """
    train_source = _load_function_source("run_niah_test")
    capacity_source = _load_function_source("run_niah_capacity_test")

    assert "resolved_seed = resolve_niah_run_seed(seed)" in train_source
    assert "seed_all(resolved_seed" in train_source
    assert "Final Eval Accuracy" in train_source
    assert "Best Eval Accuracy" in train_source
    assert "resolved_seed = resolve_niah_run_seed(seed)" in capacity_source
    assert "seed_all(resolved_seed" in capacity_source
