"""Application service for decimal arithmetic emergence probes."""

from __future__ import annotations

import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Literal, cast

import torch
import torch.nn as nn
import torch.optim as optim

from scripts.toy_task_associative_recall import StandardAttentionModel
from src.dsra.domain import (
    ArithmeticCurriculumStage,
    ArithmeticEmergenceResult,
    ArithmeticExample,
    ArithmeticRuleDatasetSpec,
)
from src.dsra.dsra_model import MultiLayerMHDSRA2Model


MHDSRA2_MODEL = "mhdsra2"
STANDARD_ATTENTION_MODEL = "standard_attention"
MINIMAL_RULE_SET = "minimal_rule_set"
CURRICULUM_RULE_SET = "curriculum_rule_set"
SINGLE_FACT_ONLY = "single_fact_only"
UNIT_NO_CARRY_STAGE = "unit_no_carry"
UNIT_WITH_CARRY_STAGE = "unit_with_carry"
TWO_DIGIT_RULES_STAGE = "two_digit_rules"
UNIT_WITH_CARRY_ONLY = "unit_with_carry_only"
TWO_DIGIT_ONLY = "two_digit_only"
PREREQ_PLUS_TWO_DIGIT = "prereq_plus_two_digit"
BASELINE_TRAINING_STRATEGY = "baseline"
CARRY_REPLAY_TRAINING_STRATEGY = "carry_replay"
STAGE_WEIGHTED_LOSS_TRAINING_STRATEGY = "stage_weighted_loss"
TWO_DIGIT_REPLAY_TRAINING_STRATEGY = "two_digit_replay"
TWO_DIGIT_WEIGHTED_LOSS_TRAINING_STRATEGY = "two_digit_weighted_loss"
COMBINED_TRAINING_STRATEGY = "combined"
TRAINING_STRATEGIES = (
    BASELINE_TRAINING_STRATEGY,
    CARRY_REPLAY_TRAINING_STRATEGY,
    STAGE_WEIGHTED_LOSS_TRAINING_STRATEGY,
    COMBINED_TRAINING_STRATEGY,
)
TWO_DIGIT_TRAINING_STRATEGIES = (
    BASELINE_TRAINING_STRATEGY,
    TWO_DIGIT_REPLAY_TRAINING_STRATEGY,
    TWO_DIGIT_WEIGHTED_LOSS_TRAINING_STRATEGY,
    COMBINED_TRAINING_STRATEGY,
)
TWO_DIGIT_DIAGNOSTIC_DATASETS = (
    CURRICULUM_RULE_SET,
    TWO_DIGIT_ONLY,
    PREREQ_PLUS_TWO_DIGIT,
)
ALL_TRAINING_STRATEGIES = tuple(dict.fromkeys((*TRAINING_STRATEGIES, *TWO_DIGIT_TRAINING_STRATEGIES)))
TrainingStrategy = Literal[
    "baseline",
    "carry_replay",
    "stage_weighted_loss",
    "two_digit_replay",
    "two_digit_weighted_loss",
    "combined",
]
DEFAULT_LAYER_COUNTS = (1, 2, 4, 8, 16)
DEFAULT_STRATEGY_GRID_LAYER_COUNTS = (4,)
DEFAULT_SEEDS = (101, 202, 303)
DEFAULT_STRATEGY_GRID_REPLAY_RATIOS = (0.25, 0.5, 0.75)
DEFAULT_STRATEGY_GRID_STAGE_PATIENCES = (1, 2, 3)
DEFAULT_MAX_STEPS_PER_STAGE = 64
DEFAULT_STRATEGY_GRID_STEP_BUDGETS = (DEFAULT_MAX_STEPS_PER_STAGE,)
DEFAULT_TRAINING_STEPS = DEFAULT_MAX_STEPS_PER_STAGE
DEFAULT_DIM = 16
DEFAULT_SLOTS = 8
DEFAULT_TOPK = 2
DEFAULT_CHUNK_SIZE = 16
TRAIN_EXACT_MATCH_THRESHOLD = 0.95
CURRICULUM_STAGE_EXACT_MATCH_THRESHOLD = 0.95
DEFAULT_CURRICULUM_EVAL_INTERVAL = 8
DEFAULT_REPLAY_RATIO = 0.5
DEFAULT_STAGE_PATIENCE = 2
DEFAULT_LEARNING_RATE = 0.03
DEFAULT_ARITHMETIC_EMERGENCE_MAX_STEPS_PER_STAGE = 512
DEFAULT_ARITHMETIC_EMERGENCE_REPLAY_RATIO = 0.75
DEFAULT_ARITHMETIC_EMERGENCE_STAGE_PATIENCE = 3
DEFAULT_ARITHMETIC_EMERGENCE_LEARNING_RATE = 0.01
DEFAULT_ARITHMETIC_EMERGENCE_DEVICE = "auto"
DEFAULT_CARRY_REPLAY_RATIO = 0.75
DEFAULT_STAGE_LOSS_WEIGHTS: Mapping[str, float] = {UNIT_WITH_CARRY_STAGE: 2.0}
DEFAULT_CARRY_DIAGNOSTIC_LAYER_COUNTS = (4, 8, 16)
DEFAULT_CARRY_DIAGNOSTIC_STEP_BUDGETS = (128, 256)
DEFAULT_CARRY_DIAGNOSTIC_EVAL_INTERVALS = (4, 8, 16)
DEFAULT_CARRY_DIAGNOSTIC_LEARNING_RATES = (0.003, 0.01, 0.03)
DEFAULT_TWO_DIGIT_DIAGNOSTIC_LAYER_COUNTS = (4, 8, 16)
DEFAULT_TWO_DIGIT_DIAGNOSTIC_STEP_BUDGETS = (512, 1024)
DEFAULT_TWO_DIGIT_DIAGNOSTIC_LEARNING_RATES = (0.003, 0.01)
DEFAULT_TWO_DIGIT_REPLAY_RATIO = 0.75
DEFAULT_TWO_DIGIT_STAGE_LOSS_WEIGHTS: Mapping[str, float] = {TWO_DIGIT_RULES_STAGE: 2.0}
HEADLINE_EXACT_MATCH_THRESHOLD = 1.0
OOD_EXACT_MATCH_THRESHOLD = 0.80


@dataclass(frozen=True)
class GeneratedArithmeticAnswer:
    """Greedy generated answer for one decimal arithmetic prompt.

    中文说明:
    - 调用方 / Called by: `greedy_generate_answer`,
      `is_exact_generated_answer`, `evaluate_arithmetic_examples`.
    - 调用对象 / Calls: 无；该类型只保存生成文本和终止状态。
    - 作用 / Purpose: 区分完整 `<eos>` 终止生成与未闭合/截断生成。
    - 变量 / Variables:
      `text` 是生成答案, `stopped_on_eos` 表示是否遇到 `<eos>`,
      `token_ids` 是生成 token 序列。
    - 接入 / Integration: 精确匹配必须同时检查 `text` 和 `stopped_on_eos`。
    - 错误处理 / Error handling: 解码异常由 tokenizer 负责抛出。
    - 关键词 / Keywords:
      greedy|generation|answer|eos|decimal|arithmetic|mhdsra2|exact_match|application|生成
    """

    text: str
    stopped_on_eos: bool
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class ArithmeticStageMetric:
    """Exact-match metric for one curriculum stage.

    中文说明:
    - 调用方 / Called by: `evaluate_curriculum_stage_metrics`,
      `run_one_arithmetic_emergence_curve`, report serialization and tests.
    - 调用对象 / Calls: none; this type only stores immutable metric data.
    - 作用 / Purpose: 用强类型记录单个课程阶段的阶段内 EM。
    - 变量 / Variables: `stage_name` 是课程阶段名, `exact_match` 是该阶段生成精确率。
    - 接入 / Integration: 曲线快照和最终 run 结果都复用本类型。
    - 错误处理 / Error handling: 数值合法性由评估函数保证。
    - 关键词 / Keywords:
      stage|metric|exact_match|curriculum|decimal|addition|mhdsra2|curve|阶段|指标
    """

    stage_name: str
    exact_match: float


@dataclass(frozen=True)
class ArithmeticCurriculumSnapshot:
    """One adaptive curriculum progress checkpoint.

    中文说明:
    - 调用方 / Called by: `run_one_arithmetic_emergence_curve`,
      `build_layer_emergence_payload`, report rendering and tests.
    - 调用对象 / Calls: none; this type only stores checkpoint data.
    - 作用 / Purpose: 记录某一步的主动训练阶段、是否推进、以及各阶段 EM 曲线点。
    - 变量 / Variables: `step` 是训练步数, `active_stage_name` 是检查前训练阶段,
      `advanced_to_stage_name` 是达标后进入的下一阶段或 `None`。
    - 接入 / Integration: JSON/Markdown 报告直接序列化本类型。
    - 错误处理 / Error handling: 推进逻辑由训练服务保证一致。
    - 关键词 / Keywords:
      snapshot|curriculum|adaptive|advance|stage_em|curve|mhdsra2|report|快照|推进
    """

    dataset_name: str
    model_name: str
    seed: int
    num_layers: int
    step: int
    active_stage_name: str
    advanced_to_stage_name: str | None
    stage_exact_matches: tuple[ArithmeticStageMetric, ...]


@dataclass(frozen=True)
class ArithmeticStageAggregate:
    """Layer-level aggregate for one curriculum stage.

    中文说明:
    - 调用方 / Called by: `aggregate_curriculum_stage_progress`,
      `build_layer_emergence_payload`, `build_layer_emergence_markdown`.
    - 调用对象 / Calls: none; this type stores immutable aggregate data.
    - 作用 / Purpose: 用按层摘要替代逐 seed 曲线表, 降低报告噪音。
    - 变量 / Variables: `pass_rate` 是阶段通过率, `advance_step_mean` 是达标步数均值,
      `final_exact_match_mean` 是训练结束时该阶段 EM 均值。
    - 接入 / Integration: JSON 和 Markdown 报告直接序列化本类型。
    - 错误处理 / Error handling: 聚合输入为空时由应用服务跳过该组。
    - 关键词 / Keywords:
      aggregate|curriculum|stage|pass_rate|advance_step|layer|mhdsra2|report|聚合|阶段
    """

    dataset_name: str
    model_name: str
    num_layers: int
    stage_name: str
    num_runs: int
    pass_rate: float
    advance_step_mean: float | None
    advance_step_variance: float | None
    final_exact_match_mean: float
    final_exact_match_variance: float


@dataclass(frozen=True)
class ArithmeticStrategyGridRun:
    """One curriculum strategy grid run.

    中文说明:
    - 调用方 / Called by: `build_curriculum_strategy_grid_payload`.
    - 调用对象 / Calls: none; this type stores one strategy and run result.
    - 作用 / Purpose: 将 replay/patience 策略参数和单 seed 运行结果绑定。
    - 变量 / Variables: `replay_ratio` 是回放比例, `stage_patience` 是阶段耐心次数。
    - 接入 / Integration: 网格 JSON 报告直接序列化本类型。
    - 错误处理 / Error handling: 参数合法性由训练服务校验。
    - 关键词 / Keywords:
      strategy|grid|replay_ratio|stage_patience|run|curriculum|mhdsra2|report|策略|网格
    """

    replay_ratio: float
    stage_patience: int
    max_steps_per_stage: int
    run: ArithmeticEmergenceRun


@dataclass(frozen=True)
class ArithmeticStrategyGridResult:
    """Aggregated result for one curriculum strategy grid cell.

    中文说明:
    - 调用方 / Called by: `aggregate_curriculum_strategy_grid_runs`,
      `build_curriculum_strategy_grid_markdown`.
    - 调用对象 / Calls: none; this type stores immutable aggregate data.
    - 作用 / Purpose: 汇总某 replay/patience/layer 组合是否稳定保留目标课程阶段。
    - 变量 / Variables: `target_retention_rate` 是保留前 N 阶段的比例,
      `stable_target_retention` 表示所有 seed 都保留目标阶段。
    - 接入 / Integration: Markdown 网格摘要按本类型渲染。
    - 错误处理 / Error handling: 聚合输入为空时上游跳过。
    - 关键词 / Keywords:
      strategy|aggregate|retention|replay_ratio|patience|grid|mhdsra2|report|稳定|保留
    """

    replay_ratio: float
    stage_patience: int
    max_steps_per_stage: int
    num_layers: int
    num_runs: int
    target_stage_count: int
    target_retention_rate: float
    stable_target_retention: bool
    retained_stage_count_mean: float
    retained_stage_count_variance: float
    ever_passed_stage_count_mean: float
    ever_passed_stage_count_variance: float
    train_exact_match_mean: float
    train_exact_match_variance: float
    final_loss_mean: float
    final_loss_variance: float


@dataclass(frozen=True)
class ArithmeticCarryDiagnosticRun:
    """One serialized point in the carry diagnostic grid.

    中文说明:
    - 调用方 / Called by: carry diagnostic grid builders, CLI checkpoint writer and tests.
    - 调用对象 / Calls: none; this type stores immutable grid metadata and run result.
    - 作用 / Purpose: 绑定 dataset/strategy/lr/eval/step/layer/seed 与一次训练结果。
    - 变量 / Variables: `run` 是底层 arithmetic emergence 训练结果。
    - 接入 / Integration: `asdict` 后逐行写入 checkpoint JSONL。
    - 错误处理 / Error handling: 参数合法性由训练入口和 CLI 解析负责。
    - 关键词 / Keywords:
      carry_diagnostic|grid|run|checkpoint|learning_rate|strategy|mhdsra2|arithmetic|诊断|进位
    """

    dataset_name: str
    training_strategy: str
    learning_rate: float
    curriculum_eval_interval: int
    max_steps_per_stage: int
    num_layers: int
    seed: int
    replay_ratio: float
    stage_patience: int
    carry_replay_ratio: float
    run: ArithmeticEmergenceRun


@dataclass(frozen=True)
class ArithmeticCarryDiagnosticAggregate:
    """Aggregated result for one carry diagnostic grid cell.

    中文说明:
    - 调用方 / Called by: `aggregate_carry_diagnostic_run_rows` and report rendering.
    - 调用对象 / Calls: none; this type stores immutable aggregate metrics.
    - 作用 / Purpose: 汇总同一 strategy/lr/eval/step/layer 的多 seed carry 诊断结果。
    - 变量 / Variables: `carry_exact_match_mean` 是进位规则 EM 均值。
    - 接入 / Integration: JSON/Markdown 报告表直接来自本类型。
    - 错误处理 / Error handling: 空分组由聚合函数跳过。
    - 关键词 / Keywords:
      carry|aggregate|learning_rate|eval_interval|strategy|retention|mhdsra2|diagnostic|聚合|进位
    """

    dataset_name: str
    training_strategy: str
    learning_rate: float
    curriculum_eval_interval: int
    max_steps_per_stage: int
    num_layers: int
    num_runs: int
    target_stage_count: int
    target_retention_rate: float
    stable_target_retention: bool
    carry_exact_match_mean: float
    carry_exact_match_variance: float
    train_exact_match_mean: float
    train_exact_match_variance: float
    retained_stage_count_mean: float
    retained_stage_count_variance: float
    final_loss_mean: float
    final_loss_variance: float


@dataclass(frozen=True)
class ArithmeticTwoDigitDiagnosticRun:
    """One serialized point in the two-digit diagnostic grid.

    中文说明:
    - 调用方 / Called by: two-digit diagnostic payload builders, CLI checkpoint writer and tests.
    - 调用对象 / Calls: none; this type stores immutable grid metadata and run result.
    - 作用 / Purpose: 绑定 dataset/strategy/lr/step/layer/seed 与一次 two-digit 训练结果。
    - 变量 / Variables: `run` 是底层 arithmetic emergence 训练结果。
    - 接入 / Integration: `asdict` 后写入 two-digit checkpoint JSONL。
    - 错误处理 / Error handling: 参数合法性由训练入口和 CLI 解析负责。
    - 关键词 / Keywords:
      two_digit|diagnostic|grid|checkpoint|learning_rate|strategy|mhdsra2|arithmetic|诊断|两位数
    """

    dataset_name: str
    training_strategy: str
    learning_rate: float
    max_steps_per_stage: int
    num_layers: int
    seed: int
    replay_ratio: float
    stage_patience: int
    two_digit_replay_ratio: float
    run: ArithmeticEmergenceRun


@dataclass(frozen=True)
class ArithmeticTwoDigitDiagnosticAggregate:
    """Aggregated result for one two-digit diagnostic grid cell.

    中文说明:
    - 调用方 / Called by: `aggregate_two_digit_diagnostic_run_rows` and report rendering.
    - 调用对象 / Calls: none; this type stores immutable aggregate metrics.
    - 作用 / Purpose: 汇总同一 dataset/strategy/lr/step/layer 的多 seed two-digit 结果。
    - 变量 / Variables: `two_digit_exact_match_mean` 是 two_digit_rules EM 均值。
    - 接入 / Integration: JSON/Markdown 报告表直接来自本类型。
    - 错误处理 / Error handling: 空分组由聚合函数跳过。
    - 关键词 / Keywords:
      two_digit|aggregate|learning_rate|strategy|retention|mhdsra2|diagnostic|rules|聚合|两位数
    """

    dataset_name: str
    training_strategy: str
    learning_rate: float
    max_steps_per_stage: int
    num_layers: int
    num_runs: int
    target_stage_count: int
    target_retention_rate: float
    stable_target_retention: bool
    two_digit_exact_match_mean: float
    two_digit_exact_match_variance: float
    train_exact_match_mean: float
    train_exact_match_variance: float
    retained_stage_count_mean: float
    retained_stage_count_variance: float
    final_loss_mean: float
    final_loss_variance: float


@dataclass(frozen=True)
class ArithmeticEmergenceRun:
    """Single-seed arithmetic emergence run for one model/depth/dataset.

    中文说明:
    - 调用方 / Called by: `run_one_arithmetic_emergence_curve`,
      `aggregate_arithmetic_emergence_runs`.
    - 调用对象 / Calls: 无；该类型只保存不可变指标。
    - 作用 / Purpose: 记录单 seed 下训练内、headline 和 OOD 生成精确匹配结果。
    - 变量 / Variables:
      `dataset_name/model_name/seed/num_layers` 定义实验点,
      `headline_prediction` 是 `100+100=` 的 greedy 输出。
    - 接入 / Integration: JSON 报告直接序列化本类型。
    - 错误处理 / Error handling: 训练或生成异常在构造前向上抛出。
    - 关键词 / Keywords:
      run|seed|arithmetic|emergence|headline|ood|mhdsra2|decimal|report|单次
    """

    dataset_name: str
    model_name: str
    seed: int
    num_layers: int
    learning_rate: float
    training_strategy: str
    carry_replay_ratio: float
    train_exact_match: float
    headline_exact_match: float
    ood_exact_match: float
    final_loss: float
    headline_prediction: str
    headline_stopped_on_eos: bool
    completed_curriculum_stages: int
    ever_passed_stage_count: int
    retained_stage_count: int
    training_steps_executed: int
    stopped_reason: str
    stage_exact_matches: tuple[ArithmeticStageMetric, ...]
    curriculum_snapshots: tuple[ArithmeticCurriculumSnapshot, ...]


class DecimalArithmeticTokenizer:
    """Character-level tokenizer for decimal addition equations.

    中文说明:
    - 调用方 / Called by: `encode_training_example`,
      `greedy_generate_answer`, tests.
    - 调用对象 / Calls: built-in dict/list operations.
    - 作用 / Purpose: 固定 `0-9`, `+`, `=`, `;`, `<bos>`, `<eos>`, `<pad>` 字符级词表。
    - 变量 / Variables:
      `token_to_id` 是 token 到 id 映射, `id_to_token` 是反向映射。
    - 接入 / Integration: 所有算术训练和生成都通过本 tokenizer。
    - 错误处理 / Error handling: 未知字符抛出 `ValueError`。
    - 关键词 / Keywords:
      tokenizer|decimal|character|addition|bos|eos|pad|mhdsra2|application|分词
    """

    pad_token = "<pad>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    alphabet = "0123456789+=;"

    def __init__(self) -> None:
        """Create the fixed decimal arithmetic vocabulary.

        中文说明:
        - 调用方 / Called by: report service and tests.
        - 调用对象 / Calls: built-in `enumerate`.
        - 作用 / Purpose: 初始化特殊 token 与算术字符 token 映射。
        - 变量 / Variables: `tokens` 是有序 token 表。
        - 接入 / Integration: 直接实例化即可使用。
        - 错误处理 / Error handling: 构造过程无外部输入, 不吞异常。
        - 关键词 / Keywords:
          init|vocabulary|decimal|tokens|bos|eos|pad|arithmetic|mhdsra2|初始化
        """
        tokens = [self.pad_token, self.bos_token, self.eos_token, *list(self.alphabet)]
        self.token_to_id = {token: token_id for token_id, token in enumerate(tokens)}
        self.id_to_token = {token_id: token for token, token_id in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size.

        中文说明:
        - 调用方 / Called by: `build_arithmetic_model`, tests.
        - 调用对象 / Calls: built-in `len`.
        - 作用 / Purpose: 为模型 embedding/out_proj 提供词表大小。
        - 变量 / Variables: `token_to_id` 是当前词表映射。
        - 接入 / Integration: 构造模型时传入 `vocab_size`。
        - 错误处理 / Error handling: 纯长度查询, 不吞异常。
        - 关键词 / Keywords:
          vocab_size|tokenizer|decimal|model|embedding|arithmetic|mhdsra2|application|词表|大小
        """
        return len(self.token_to_id)

    @property
    def bos_id(self) -> int:
        """Return the beginning-of-sequence token id.

        中文说明:
        - 调用方 / Called by: `encode_text`.
        - 调用对象 / Calls: dict lookup.
        - 作用 / Purpose: 提供 `<bos>` id。
        - 变量 / Variables: `bos_token` 是特殊 token 字符串。
        - 接入 / Integration: 编码 prompt/training text 时使用。
        - 错误处理 / Error handling: 映射缺失会抛出 `KeyError`。
        - 关键词 / Keywords:
          bos|token_id|tokenizer|decimal|sequence|arithmetic|mhdsra2|application|开始|标记
        """
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        """Return the end-of-sequence token id.

        中文说明:
        - 调用方 / Called by: `encode_text`, `greedy_generate_answer`.
        - 调用对象 / Calls: dict lookup.
        - 作用 / Purpose: 提供 `<eos>` id 并支持完整生成判定。
        - 变量 / Variables: `eos_token` 是特殊 token 字符串。
        - 接入 / Integration: 训练目标和 greedy 终止均使用。
        - 错误处理 / Error handling: 映射缺失会抛出 `KeyError`。
        - 关键词 / Keywords:
          eos|token_id|tokenizer|decimal|generation|arithmetic|mhdsra2|application|结束|标记
        """
        return self.token_to_id[self.eos_token]

    def encode_text(self, text: str, *, add_bos: bool, add_eos: bool) -> list[int]:
        """Encode decimal arithmetic text into token ids.

        中文说明:
        - 调用方 / Called by: `encode_training_example`,
          `greedy_generate_answer`, tests.
        - 调用对象 / Calls: dict lookup, list append.
        - 作用 / Purpose: 将十进制算术字符串转换为模型 token ids。
        - 变量 / Variables: `text` 是输入字符串, `token_ids` 是输出 id 列表。
        - 接入 / Integration: prompt 编码用 `add_bos=True, add_eos=False`。
        - 错误处理 / Error handling: 遇到未知字符抛出 `ValueError`。
        - 关键词 / Keywords:
          encode|text|decimal|token_ids|bos|eos|arithmetic|mhdsra2|application|编码
        """
        token_ids = [self.bos_id] if add_bos else []
        for char in text:
            if char not in self.token_to_id:
                raise ValueError(f"Unsupported arithmetic character: {char!r}")
            token_ids.append(self.token_to_id[char])
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode_token_ids(self, token_ids: Sequence[int]) -> str:
        """Decode token ids into decimal arithmetic text.

        中文说明:
        - 调用方 / Called by: `greedy_generate_answer`, tests.
        - 调用对象 / Calls: dict lookup, string join.
        - 作用 / Purpose: 将生成 token ids 还原为答案文本, 并忽略特殊 token。
        - 变量 / Variables: `token_ids` 是待解码序列, `chars` 是输出字符列表。
        - 接入 / Integration: greedy 输出后调用。
        - 错误处理 / Error handling: 未知 id 抛出 `ValueError`。
        - 关键词 / Keywords:
          decode|token_ids|decimal|answer|generation|arithmetic|mhdsra2|application|解码|文本
        """
        chars = []
        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id))
            if token is None:
                raise ValueError(f"Unsupported arithmetic token id: {token_id}")
            if token in {self.pad_token, self.bos_token, self.eos_token}:
                continue
            chars.append(token)
        return "".join(chars)


def build_curriculum_arithmetic_spec() -> ArithmeticRuleDatasetSpec:
    """Build the staged low-value decimal addition curriculum.

    中文说明:
    - 调用方 / Called by: `build_default_arithmetic_spec`,
      `build_default_arithmetic_specs`, tests.
    - 调用对象 / Calls: `ArithmeticExample`, `ArithmeticCurriculumStage`,
      `ArithmeticRuleDatasetSpec`.
    - 作用 / Purpose: 构造“个位无进位 -> 进位 -> 两位数”的课程训练集,
      同时严格保留百位 headline/OOD 作为外推测试。
    - 变量 / Variables: `no_carry_units/carry_units/two_digit_rules` 分别是三段课程,
      `training_examples` 是按课程顺序展开后的完整训练集。
    - 接入 / Integration: 默认报告使用本规约判定 MHDSRA2 最小涌现层数。
    - 错误处理 / Error handling: 返回后由 `validate_training_scope` 校验泄漏和阶段一致性。
    - 关键词 / Keywords:
      curriculum|decimal|addition|no_carry|carry|two_digit|100+100|mhdsra2|课程|外推
    """
    no_carry_units = ArithmeticCurriculumStage(
        name="unit_no_carry",
        examples=(
            ArithmeticExample(0, 0, 0),
            ArithmeticExample(1, 1, 2),
            ArithmeticExample(2, 3, 5),
            ArithmeticExample(4, 5, 9),
        ),
    )
    carry_units = ArithmeticCurriculumStage(
        name="unit_with_carry",
        examples=(
            ArithmeticExample(5, 5, 10),
            ArithmeticExample(8, 2, 10),
            ArithmeticExample(9, 1, 10),
            ArithmeticExample(9, 9, 18),
        ),
    )
    two_digit_rules = ArithmeticCurriculumStage(
        name="two_digit_rules",
        examples=(
            ArithmeticExample(10, 10, 20),
            ArithmeticExample(11, 11, 22),
            ArithmeticExample(12, 12, 24),
            ArithmeticExample(20, 20, 40),
            ArithmeticExample(30, 40, 70),
            ArithmeticExample(55, 44, 99),
        ),
    )
    curriculum_stages = (no_carry_units, carry_units, two_digit_rules)
    training_examples = tuple(
        example
        for stage in curriculum_stages
        for example in stage.examples
    )
    return ArithmeticRuleDatasetSpec(
        name=CURRICULUM_RULE_SET,
        training_examples=training_examples,
        curriculum_stages=curriculum_stages,
        headline_example=ArithmeticExample(100, 100, 200),
        ood_examples=(
            ArithmeticExample(101, 101, 202),
            ArithmeticExample(110, 110, 220),
            ArithmeticExample(99, 1, 100),
            ArithmeticExample(20, 80, 100),
        ),
    )


def build_default_arithmetic_spec() -> ArithmeticRuleDatasetSpec:
    """Build the default curriculum decimal addition rule set.

    中文说明:
    - 调用方 / Called by: `build_default_arithmetic_specs`, tests.
    - 调用对象 / Calls: `ArithmeticExample`, `ArithmeticRuleDatasetSpec`.
    - 作用 / Purpose: 构造不含百位样例的少量规则训练集和百位 OOD/headline 测试集。
    - 变量 / Variables: `training_examples` 是低位规则样例, `ood_examples` 是外推测试样例。
    - 接入 / Integration: 默认报告使用本规约判定最小涌现层数。
    - 错误处理 / Error handling: 返回后由 `validate_training_scope` 校验泄漏。
    - 关键词 / Keywords:
      minimal_rule_set|decimal|addition|training|ood|100+100|mhdsra2|application|规则|外推
    """
    return build_curriculum_arithmetic_spec()


def build_single_fact_control_spec() -> ArithmeticRuleDatasetSpec:
    """Build the `1+1=2` negative-control dataset.

    中文说明:
    - 调用方 / Called by: `build_default_arithmetic_specs`, tests.
    - 调用对象 / Calls: `ArithmeticExample`, `ArithmeticRuleDatasetSpec`.
    - 作用 / Purpose: 证明单事实不足以唯一推出十进制加法规律。
    - 变量 / Variables: `training_examples` 只包含 `1+1=2`。
    - 接入 / Integration: 报告中作为负控展示, 不参与最小成功层数判定。
    - 错误处理 / Error handling: 返回后由 `validate_training_scope` 校验泄漏。
    - 关键词 / Keywords:
      single_fact_only|negative_control|1+1=2|decimal|addition|mhdsra2|application|负控|单事实|规律
    """
    return ArithmeticRuleDatasetSpec(
        name=SINGLE_FACT_ONLY,
        training_examples=(ArithmeticExample(1, 1, 2),),
        headline_example=ArithmeticExample(100, 100, 200),
        ood_examples=(
            ArithmeticExample(101, 101, 202),
            ArithmeticExample(110, 110, 220),
            ArithmeticExample(99, 1, 100),
            ArithmeticExample(20, 80, 100),
        ),
    )


def build_unit_with_carry_only_spec() -> ArithmeticRuleDatasetSpec:
    """Build a diagnostic dataset that trains only unit carry examples.

    中文说明:
    - 调用方 / Called by: carry diagnostic grid builders and tests.
    - 调用对象 / Calls: `ArithmeticExample`, `ArithmeticRuleDatasetSpec`.
    - 作用 / Purpose: 隔离 `unit_with_carry`，验证模型是否能单独学会个位进位规则。
    - 变量 / Variables: `carry_examples` 是四条个位进位训练样例。
    - 接入 / Integration: 作为 `mhdsra2_carry_diagnostic_grid` 的诊断数据集。
    - 错误处理 / Error handling: 返回后继续使用 `validate_training_scope` 检查泄漏。
    - 关键词 / Keywords:
      unit_with_carry_only|carry|diagnostic|dataset|arithmetic|mhdsra2|training|rule|进位|诊断
    """
    carry_examples = (
        ArithmeticExample(5, 5, 10),
        ArithmeticExample(8, 2, 10),
        ArithmeticExample(9, 1, 10),
        ArithmeticExample(9, 9, 18),
    )
    return ArithmeticRuleDatasetSpec(
        name=UNIT_WITH_CARRY_ONLY,
        training_examples=carry_examples,
        diagnostic_stages=(
            ArithmeticCurriculumStage(name=UNIT_WITH_CARRY_STAGE, examples=carry_examples),
        ),
        headline_example=ArithmeticExample(100, 100, 200),
        ood_examples=(
            ArithmeticExample(101, 101, 202),
            ArithmeticExample(110, 110, 220),
            ArithmeticExample(99, 1, 100),
            ArithmeticExample(20, 80, 100),
        ),
    )


def build_two_digit_only_spec() -> ArithmeticRuleDatasetSpec:
    """Build a diagnostic dataset that trains only two-digit rule examples.

    中文说明:
    - 调用方 / Called by: two-digit diagnostic grid builders and tests.
    - 调用对象 / Calls: `ArithmeticExample`, `ArithmeticRuleDatasetSpec`.
    - 作用 / Purpose: 隔离 `two_digit_rules`，验证模型是否能单独学会两位数规则。
    - 变量 / Variables: `two_digit_examples` 是两位数规则训练样例。
    - 接入 / Integration: 作为 `mhdsra2_two_digit_diagnostic_grid` 的诊断数据集。
    - 错误处理 / Error handling: 返回后继续使用 `validate_training_scope` 检查泄漏。
    - 关键词 / Keywords:
      two_digit_only|diagnostic|dataset|arithmetic|mhdsra2|training|rule|addition|两位数|诊断
    """
    two_digit_examples = (
        ArithmeticExample(10, 10, 20),
        ArithmeticExample(11, 11, 22),
        ArithmeticExample(12, 12, 24),
    )
    return ArithmeticRuleDatasetSpec(
        name=TWO_DIGIT_ONLY,
        training_examples=two_digit_examples,
        diagnostic_stages=(
            ArithmeticCurriculumStage(name=TWO_DIGIT_RULES_STAGE, examples=two_digit_examples),
        ),
        headline_example=ArithmeticExample(100, 100, 200),
        ood_examples=(
            ArithmeticExample(101, 101, 202),
            ArithmeticExample(110, 110, 220),
            ArithmeticExample(99, 1, 100),
            ArithmeticExample(20, 80, 100),
        ),
    )


def build_prereq_plus_two_digit_spec() -> ArithmeticRuleDatasetSpec:
    """Build a non-adaptive mixed prerequisite plus two-digit diagnostic dataset.

    中文说明:
    - 调用方 / Called by: two-digit diagnostic grid builders and tests.
    - 调用对象 / Calls: `build_curriculum_arithmetic_spec`, `ArithmeticRuleDatasetSpec`.
    - 作用 / Purpose: 在不使用 adaptive curriculum 的情况下混合三个阶段样例。
    - 变量 / Variables: `base_spec` 是现有 curriculum 规则集, `diagnostic_stages` 保留阶段评估边界。
    - 接入 / Integration: 用于区分两位数规则本身问题和 adaptive 推进/遗忘问题。
    - 错误处理 / Error handling: 返回后继续使用 `validate_training_scope` 检查泄漏。
    - 关键词 / Keywords:
      prereq|two_digit|mixed|diagnostic|dataset|non_adaptive|mhdsra2|arithmetic|混合|两位数
    """
    base_spec = build_curriculum_arithmetic_spec()
    return ArithmeticRuleDatasetSpec(
        name=PREREQ_PLUS_TWO_DIGIT,
        training_examples=base_spec.training_examples,
        diagnostic_stages=base_spec.curriculum_stages,
        headline_example=base_spec.headline_example,
        ood_examples=base_spec.ood_examples,
    )


def build_default_arithmetic_specs() -> tuple[ArithmeticRuleDatasetSpec, ...]:
    """Build all default arithmetic emergence dataset specifications.

    中文说明:
    - 调用方 / Called by: `build_layer_emergence_payload`, tests.
    - 调用对象 / Calls: `build_default_arithmetic_spec`, `build_single_fact_control_spec`.
    - 作用 / Purpose: 统一返回主实验和负控规约。
    - 变量 / Variables: 返回 tuple 顺序固定, 先主实验后负控。
    - 接入 / Integration: 新增数据集规约时在此追加。
    - 错误处理 / Error handling: 下游会逐个校验训练集泄漏。
    - 关键词 / Keywords:
      specs|minimal_rule_set|single_fact_only|arithmetic|decimal|mhdsra2|application|datasets|规约|负控
    """
    return (build_default_arithmetic_spec(), build_single_fact_control_spec())


def encode_training_example(
    tokenizer: DecimalArithmeticTokenizer,
    example: ArithmeticExample,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one full equation into next-token training tensors.

    中文说明:
    - 调用方 / Called by: `run_one_arithmetic_emergence_curve`.
    - 调用对象 / Calls: `DecimalArithmeticTokenizer.encode_text`, `torch.tensor`.
    - 作用 / Purpose: 将 `1+1=2<eos>` 转成自回归语言模型输入和目标。
    - 变量 / Variables: `token_ids` 是完整序列, `inputs/targets` 是错位张量。
    - 接入 / Integration: 训练循环逐样例调用。
    - 错误处理 / Error handling: tokenizer 或张量构造异常直接抛出。
    - 关键词 / Keywords:
      encode|training|equation|next_token|decimal|arithmetic|mhdsra2|application|训练|张量
    """
    token_ids = tokenizer.encode_text(example.equation, add_bos=True, add_eos=True)
    inputs = torch.tensor([token_ids[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor([token_ids[1:]], dtype=torch.long, device=device)
    return inputs, targets


def resolve_torch_device(device_name: str | torch.device) -> torch.device:
    """Resolve an arithmetic experiment device setting.

    中文说明:
    - 调用方 / Called by: `run_one_arithmetic_emergence_curve`,
      `build_layer_emergence_payload`, CLI tests.
    - 调用对象 / Calls: `torch.cuda.is_available`, `torch.device`.
    - 作用 / Purpose: 将 `auto/cpu/cuda` 入口规范化为可执行的 `torch.device`。
    - 变量 / Variables: `device_name` 是外部传入设备名, `normalized` 是小写设备字符串。
    - 接入 / Integration: 主实验 CLI 通过 `--device` 传入, 应用层统一解析。
    - 错误处理 / Error handling: `auto` 无 CUDA 时回退 CPU, 非法设备字符串由 `torch.device`
      或本函数抛出。
    - 关键词 / Keywords:
      device|cuda|cpu|auto|torch|resolve|mhdsra2|arithmetic|设备|解析
    """
    if isinstance(device_name, torch.device):
        return device_name
    normalized = device_name.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported arithmetic emergence device: {device_name}")
    return torch.device(normalized)


def build_arithmetic_model(
    *,
    model_name: str,
    num_layers: int,
    vocab_size: int,
) -> nn.Module:
    """Build an arithmetic probe model by name.

    中文说明:
    - 调用方 / Called by: `run_one_arithmetic_emergence_curve`.
    - 调用对象 / Calls: `MultiLayerMHDSRA2Model`, `StandardAttentionModel`.
    - 作用 / Purpose: 为主模型和标准注意力参考基线提供统一构造入口。
    - 变量 / Variables: `model_name` 选择模型, `num_layers` 只影响 MHDSRA2 深度。
    - 接入 / Integration: 新增参考模型时在此追加分支。
    - 错误处理 / Error handling: 未知模型名抛出 `ValueError`。
    - 关键词 / Keywords:
      model_factory|mhdsra2|standard_attention|arithmetic|layers|vocab|application|baseline|模型|构建
    """
    if model_name == MHDSRA2_MODEL:
        return MultiLayerMHDSRA2Model(
            vocab_size=vocab_size,
            dim=DEFAULT_DIM,
            num_layers=num_layers,
            K=DEFAULT_SLOTS,
            kr=DEFAULT_TOPK,
            chunk_size=DEFAULT_CHUNK_SIZE,
        )
    if model_name == STANDARD_ATTENTION_MODEL:
        return StandardAttentionModel(
            vocab_size=vocab_size,
            dim=DEFAULT_DIM,
            chunk_size=DEFAULT_CHUNK_SIZE,
            local_context_size=1,
            local_context_mode="none",
        )
    raise ValueError(f"Unsupported arithmetic model: {model_name}")


def greedy_generate_answer(
    model: nn.Module,
    tokenizer: DecimalArithmeticTokenizer,
    prompt: str,
    *,
    max_answer_tokens: int,
    device: torch.device,
) -> GeneratedArithmeticAnswer:
    """Generate one answer from a prompt without teacher forcing.

    中文说明:
    - 调用方 / Called by: `evaluate_arithmetic_examples`, tests.
    - 调用对象 / Calls: `DecimalArithmeticTokenizer.encode_text`,
      `DecimalArithmeticTokenizer.decode_token_ids`, `nn.Module.forward`.
    - 作用 / Purpose: 从 `100+100=` 自回归生成答案, 不使用 teacher-forced logits。
    - 变量 / Variables:
      `input_ids` 是不断扩展的上下文, `generated_ids` 是答案 token 序列。
    - 接入 / Integration: 所有 headline/OOD/train exact match 都通过本函数评估。
    - 错误处理 / Error handling: tokenizer/model 异常直接抛出。
    - 关键词 / Keywords:
      greedy|generation|no_teacher_forcing|100+100|decimal|arithmetic|mhdsra2|application|生成|推断
    """
    model.eval()
    input_ids = tokenizer.encode_text(prompt, add_bos=True, add_eos=False)
    generated_ids: list[int] = []
    stopped_on_eos = False
    with torch.no_grad():
        for _ in range(max_answer_tokens):
            inputs = torch.tensor([input_ids], dtype=torch.long, device=device)
            logits = model(inputs)
            next_token_id = int(logits[0, -1, :].argmax().item())
            if next_token_id == tokenizer.eos_id:
                stopped_on_eos = True
                break
            generated_ids.append(next_token_id)
            input_ids.append(next_token_id)
    return GeneratedArithmeticAnswer(
        text=tokenizer.decode_token_ids(generated_ids),
        stopped_on_eos=stopped_on_eos,
        token_ids=tuple(generated_ids),
    )


def is_exact_generated_answer(
    generated_answer: GeneratedArithmeticAnswer,
    expected_answer: str,
) -> bool:
    """Return whether greedy generation produced the exact completed answer.

    中文说明:
    - 调用方 / Called by: `evaluate_arithmetic_examples`, tests.
    - 调用对象 / Calls: 无。
    - 作用 / Purpose: 要求答案文本精确匹配且必须遇到 `<eos>`, 防止截断输出误判成功。
    - 变量 / Variables: `generated_answer` 是模型输出, `expected_answer` 是标准答案。
    - 接入 / Integration: 所有 exact-match 指标统一使用本函数。
    - 错误处理 / Error handling: 纯布尔判断, 不吞异常。
    - 关键词 / Keywords:
      exact_match|eos|required|greedy|generation|decimal|arithmetic|mhdsra2|application|精确
    """
    return generated_answer.stopped_on_eos and generated_answer.text == expected_answer


def evaluate_arithmetic_examples(
    model: nn.Module,
    tokenizer: DecimalArithmeticTokenizer,
    examples: Sequence[ArithmeticExample],
    device: torch.device,
) -> tuple[float, tuple[GeneratedArithmeticAnswer, ...]]:
    """Evaluate exact-match generation over arithmetic examples.

    中文说明:
    - 调用方 / Called by: `run_one_arithmetic_emergence_curve`.
    - 调用对象 / Calls: `greedy_generate_answer`, `is_exact_generated_answer`.
    - 作用 / Purpose: 计算 train/headline/OOD 的无 teacher forcing 生成精确率。
    - 变量 / Variables: `answers` 是逐样例 greedy 输出, `correct` 是精确匹配数量。
    - 接入 / Integration: 新增评估 split 时直接复用。
    - 错误处理 / Error handling: 空样例列表抛出 `ValueError`。
    - 关键词 / Keywords:
      evaluate|exact_match|greedy|generation|ood|headline|arithmetic|mhdsra2|application|评估
    """
    if not examples:
        raise ValueError("Arithmetic evaluation examples must not be empty.")
    answers = tuple(
        greedy_generate_answer(
            model,
            tokenizer,
            example.prompt,
            max_answer_tokens=len(example.answer) + 2,
            device=device,
        )
        for example in examples
    )
    correct = sum(
        1
        for example, answer in zip(examples, answers)
        if is_exact_generated_answer(answer, example.answer)
    )
    return correct / len(examples), answers


def evaluate_curriculum_stage_metrics(
    model: nn.Module,
    tokenizer: DecimalArithmeticTokenizer,
    dataset_spec: ArithmeticRuleDatasetSpec,
    device: torch.device,
) -> tuple[ArithmeticStageMetric, ...]:
    """Evaluate generated exact match inside each curriculum stage.

    中文说明:
    - 调用方 / Called by: `run_one_arithmetic_emergence_curve`, tests.
    - 调用对象 / Calls: `evaluate_arithmetic_examples`, `ArithmeticStageMetric`.
    - 作用 / Purpose: 计算每个课程阶段自己的 greedy generation EM, 作为推进和报告曲线依据。
    - 变量 / Variables: `stage_metrics` 是输出指标列表, `stage` 是当前课程阶段。
    - 接入 / Integration: 课程数据集用于自适应推进；非课程数据集返回空 tuple。
    - 错误处理 / Error handling: 下游评估异常直接抛出, 不静默吞掉模型错误。
    - 关键词 / Keywords:
      evaluate|curriculum|stage|exact_match|greedy|decimal|addition|mhdsra2|阶段|评估
    """
    stage_metrics: list[ArithmeticStageMetric] = []
    for stage in dataset_spec.curriculum_stages:
        exact_match, _ = evaluate_arithmetic_examples(
            model, tokenizer, stage.examples, device
        )
        stage_metrics.append(
            ArithmeticStageMetric(stage_name=stage.name, exact_match=exact_match)
        )
    return tuple(stage_metrics)


def evaluate_arithmetic_stage_metrics(
    model: nn.Module,
    tokenizer: DecimalArithmeticTokenizer,
    stages: Sequence[ArithmeticCurriculumStage],
    device: torch.device,
) -> tuple[ArithmeticStageMetric, ...]:
    """Evaluate generated exact match for arbitrary diagnostic stages.

    中文说明:
    - 调用方 / Called by: `run_one_arithmetic_emergence_curve`.
    - 调用对象 / Calls: `evaluate_arithmetic_examples`, `ArithmeticStageMetric`.
    - 作用 / Purpose: 让非 adaptive curriculum 数据集也能报告阶段级 EM。
    - 变量 / Variables: `stages` 是诊断阶段列表, `stage_metrics` 是输出指标。
    - 接入 / Integration: `diagnostic_stages` 为空时不调用本函数。
    - 错误处理 / Error handling: 下游评估异常直接抛出。
    - 关键词 / Keywords:
      evaluate|diagnostic|stage|exact_match|two_digit|decimal|addition|mhdsra2|阶段|诊断
    """
    stage_metrics: list[ArithmeticStageMetric] = []
    for stage in stages:
        exact_match, _ = evaluate_arithmetic_examples(
            model, tokenizer, stage.examples, device
        )
        stage_metrics.append(
            ArithmeticStageMetric(stage_name=stage.name, exact_match=exact_match)
        )
    return tuple(stage_metrics)


def should_advance_curriculum_stage(
    stage_metric: ArithmeticStageMetric,
    *,
    threshold: float = CURRICULUM_STAGE_EXACT_MATCH_THRESHOLD,
) -> bool:
    """Return whether the active curriculum stage has reached its promotion threshold.

    中文说明:
    - 调用方 / Called by: `run_one_arithmetic_emergence_curve`, tests.
    - 调用对象 / Calls: none.
    - 作用 / Purpose: 把“阶段达标后进入下一阶段”的门槛集中到单一函数。
    - 变量 / Variables: `stage_metric` 是当前阶段 EM, `threshold` 是推进阈值。
    - 接入 / Integration: 调整课程推进标准时优先改本函数默认阈值或调用参数。
    - 错误处理 / Error handling: 纯比较逻辑, 不抛出自定义异常。
    - 关键词 / Keywords:
      advance|threshold|curriculum|stage|exact_match|promotion|mhdsra2|application|推进|达标
    """
    return stage_metric.exact_match >= threshold


def should_advance_open_curriculum_stages(
    stage_metrics: Sequence[ArithmeticStageMetric],
    open_stage_names: Sequence[str],
    *,
    threshold: float = CURRICULUM_STAGE_EXACT_MATCH_THRESHOLD,
) -> bool:
    """Return whether every currently open curriculum stage is retained.

    中文说明:
    - 调用方 / Called by: `run_one_arithmetic_emergence_curve`, tests.
    - 调用对象 / Calls: `_stage_metric_value`.
    - 作用 / Purpose: 推进课程时同时检查已开放阶段, 防止只看当前阶段导致遗忘。
    - 变量 / Variables: `stage_metrics` 是当前评估结果, `open_stage_names` 是已开放阶段名。
    - 接入 / Integration: curriculum 推进逻辑统一通过本函数做 retention gate。
    - 错误处理 / Error handling: 空开放阶段返回 `False`, 缺失阶段按 `0.0` 处理。
    - 关键词 / Keywords:
      retention|curriculum|advance|open_stages|threshold|replay|mhdsra2|application|保留|推进
    """
    if not open_stage_names:
        return False
    return all(
        _stage_metric_value(stage_metrics, stage_name) >= threshold
        for stage_name in open_stage_names
    )


def count_completed_curriculum_stages(
    stage_metrics: Sequence[ArithmeticStageMetric],
    *,
    threshold: float = CURRICULUM_STAGE_EXACT_MATCH_THRESHOLD,
) -> int:
    """Count leading curriculum stages that have reached the promotion threshold.

    中文说明:
    - 调用方 / Called by: `run_one_arithmetic_emergence_curve`, tests.
    - 调用对象 / Calls: `should_advance_curriculum_stage`.
    - 作用 / Purpose: 把最终阶段完成数定义为从第一阶段开始连续达标的阶段数量。
    - 变量 / Variables: `completed_count` 是连续达标阶段数, `stage_metric` 是当前指标。
    - 接入 / Integration: run 结果和报告中的 completed curriculum stages 来自本函数。
    - 错误处理 / Error handling: 空指标返回 0, 不抛出异常。
    - 关键词 / Keywords:
      completed|curriculum|stage|threshold|exact_match|count|mhdsra2|application|完成|阶段
    """
    completed_count = 0
    for stage_metric in stage_metrics:
        if not should_advance_curriculum_stage(stage_metric, threshold=threshold):
            break
        completed_count += 1
    return completed_count


def count_ever_passed_curriculum_stages(
    snapshots: Sequence[ArithmeticCurriculumSnapshot],
    stage_names: Sequence[str],
    *,
    threshold: float = CURRICULUM_STAGE_EXACT_MATCH_THRESHOLD,
) -> int:
    """Count curriculum stages that reached threshold at least once.

    中文说明:
    - 调用方 / Called by: `run_one_arithmetic_emergence_curve`, tests.
    - 调用对象 / Calls: `_stage_metric_value`.
    - 作用 / Purpose: 区分“曾经通过”的阶段数和训练结束仍保留的阶段数。
    - 变量 / Variables: `passed_stage_names` 是曾经达标的阶段集合。
    - 接入 / Integration: 报告字段 `ever_passed_stage_count` 来自本函数。
    - 错误处理 / Error handling: 空 snapshots 返回 0, 不抛出异常。
    - 关键词 / Keywords:
      ever_passed|curriculum|stage_count|threshold|retention|mhdsra2|report|曾通过|阶段|计数
    """
    passed_stage_names: set[str] = set()
    for snapshot in snapshots:
        for stage_name in stage_names:
            if _stage_metric_value(snapshot.stage_exact_matches, stage_name) >= threshold:
                passed_stage_names.add(stage_name)
    return len(passed_stage_names)


def validate_training_strategy(
    training_strategy: str,
) -> TrainingStrategy:
    """Validate and normalize a carry diagnostic training strategy.

    中文说明:
    - 调用方 / Called by: training loops, report builders and tests.
    - 调用对象 / Calls: built-in string normalization and membership checks.
    - 作用 / Purpose: 将外部 CLI/应用层输入约束到四种明确训练策略。
    - 变量 / Variables: `training_strategy` 是外部输入, `normalized` 是小写规范值。
    - 接入 / Integration: 新增策略时先更新 `TRAINING_STRATEGIES` 和本函数测试。
    - 错误处理 / Error handling: 未知策略抛出 `ValueError`，不回退到 baseline。
    - 关键词 / Keywords:
      strategy|validate|baseline|carry_replay|stage_weighted_loss|combined|mhdsra2|diagnostic|策略|校验
    """
    normalized = training_strategy.strip().lower()
    if normalized not in ALL_TRAINING_STRATEGIES:
        raise ValueError(f"Unsupported arithmetic training strategy: {training_strategy}")
    return cast(TrainingStrategy, normalized)


def uses_carry_replay(training_strategy: str) -> bool:
    """Return whether a strategy enables carry-focused replay.

    中文说明:
    - 调用方 / Called by: `select_adaptive_curriculum_training_example`.
    - 调用对象 / Calls: `validate_training_strategy`.
    - 作用 / Purpose: 集中判断 replay 分支是否要优先抽取 `unit_with_carry` 样例。
    - 变量 / Variables: `training_strategy` 是当前训练策略名。
    - 接入 / Integration: carry replay 与 combined 都通过本函数启用采样强化。
    - 错误处理 / Error handling: 非法策略由校验函数抛出。
    - 关键词 / Keywords:
      carry_replay|strategy|sampling|combined|curriculum|mhdsra2|diagnostic|replay|进位|回放
    """
    normalized = validate_training_strategy(training_strategy)
    return normalized in {CARRY_REPLAY_TRAINING_STRATEGY, COMBINED_TRAINING_STRATEGY}


def uses_two_digit_replay(training_strategy: str) -> bool:
    """Return whether a strategy enables two-digit-focused replay.

    中文说明:
    - 调用方 / Called by: `select_adaptive_curriculum_training_example`.
    - 调用对象 / Calls: `validate_training_strategy`.
    - 作用 / Purpose: 判断 replay 分支是否要优先抽取 `two_digit_rules` 样例。
    - 变量 / Variables: `training_strategy` 是当前训练策略名。
    - 接入 / Integration: two_digit_replay 和 combined 都通过本函数启用采样强化。
    - 错误处理 / Error handling: 非法策略由校验函数抛出。
    - 关键词 / Keywords:
      two_digit_replay|strategy|sampling|combined|curriculum|mhdsra2|diagnostic|replay|两位数|回放
    """
    normalized = validate_training_strategy(training_strategy)
    return normalized in {TWO_DIGIT_REPLAY_TRAINING_STRATEGY, COMBINED_TRAINING_STRATEGY}


def uses_stage_weighted_loss(training_strategy: str) -> bool:
    """Return whether a strategy enables stage-weighted loss.

    中文说明:
    - 调用方 / Called by: `run_one_arithmetic_emergence_curve`.
    - 调用对象 / Calls: `validate_training_strategy`.
    - 作用 / Purpose: 集中判断训练 loss 是否按阶段乘以权重。
    - 变量 / Variables: `training_strategy` 是当前训练策略名。
    - 接入 / Integration: stage_weighted_loss 与 combined 都通过本函数启用 loss 强化。
    - 错误处理 / Error handling: 非法策略由校验函数抛出。
    - 关键词 / Keywords:
      weighted_loss|strategy|stage|combined|carry|mhdsra2|diagnostic|optimization|加权|损失
    """
    normalized = validate_training_strategy(training_strategy)
    return normalized in {
        STAGE_WEIGHTED_LOSS_TRAINING_STRATEGY,
        TWO_DIGIT_WEIGHTED_LOSS_TRAINING_STRATEGY,
        COMBINED_TRAINING_STRATEGY,
    }


def resolve_example_stage_name(
    dataset_spec: ArithmeticRuleDatasetSpec,
    example: ArithmeticExample,
) -> str:
    """Resolve the curriculum stage name for one training example.

    中文说明:
    - 调用方 / Called by: `resolve_stage_loss_multiplier`, tests.
    - 调用对象 / Calls: dataset/stage iteration.
    - 作用 / Purpose: 将样例映射到阶段名，供阶段加权 loss 使用。
    - 变量 / Variables: `dataset_spec` 是训练规约, `example` 是当前训练样例。
    - 接入 / Integration: 非课程 carry-only 数据集统一映射为 `unit_with_carry`。
    - 错误处理 / Error handling: 找不到阶段时返回数据集名，避免伪造未知阶段。
    - 关键词 / Keywords:
      stage|lookup|example|loss_weight|unit_with_carry|dataset|mhdsra2|diagnostic|阶段|样例
    """
    for stage in dataset_spec.curriculum_stages:
        if example in stage.examples:
            return stage.name
    for stage in dataset_spec.diagnostic_stages:
        if example in stage.examples:
            return stage.name
    return dataset_spec.name


def validate_stage_loss_weights(
    stage_loss_weights: Mapping[str, float] | None,
) -> Mapping[str, float]:
    """Validate stage loss weights used by carry diagnostics.

    中文说明:
    - 调用方 / Called by: `run_one_arithmetic_emergence_curve`, tests.
    - 调用对象 / Calls: mapping iteration and float comparisons.
    - 作用 / Purpose: 防止非法权重静默改变优化目标。
    - 变量 / Variables: `stage_loss_weights` 是阶段名到 loss 倍数的映射。
    - 接入 / Integration: CLI 解析后的权重在进入训练循环前调用本函数。
    - 错误处理 / Error handling: 空阶段名、非正权重抛出 `ValueError`。
    - 关键词 / Keywords:
      validate|stage_loss_weights|loss|weight|carry|optimization|mhdsra2|diagnostic|权重|校验
    """
    if stage_loss_weights is None:
        return DEFAULT_STAGE_LOSS_WEIGHTS
    for stage_name, weight in stage_loss_weights.items():
        if not stage_name.strip():
            raise ValueError("stage loss weight names must not be empty.")
        if weight <= 0.0:
            raise ValueError("stage loss weights must be positive.")
    return stage_loss_weights


def resolve_stage_loss_multiplier(
    *,
    dataset_spec: ArithmeticRuleDatasetSpec,
    example: ArithmeticExample,
    training_strategy: str,
    stage_loss_weights: Mapping[str, float] | None,
) -> float:
    """Return the scalar loss multiplier for one training example.

    中文说明:
    - 调用方 / Called by: `run_one_arithmetic_emergence_curve`, tests.
    - 调用对象 / Calls: `uses_stage_weighted_loss`, `resolve_example_stage_name`,
      `validate_stage_loss_weights`.
    - 作用 / Purpose: 在不改模型结构的前提下提高指定阶段样例的梯度权重。
    - 变量 / Variables: `stage_name` 是当前样例阶段, `weights` 是已校验权重。
    - 接入 / Integration: 训练循环将 teacher-forced cross entropy 乘以返回值。
    - 错误处理 / Error handling: 非加权策略返回 `1.0`，非法权重抛出异常。
    - 关键词 / Keywords:
      loss_multiplier|weighted_loss|stage|carry|gradient|training|mhdsra2|diagnostic|损失|进位
    """
    if not uses_stage_weighted_loss(training_strategy):
        return 1.0
    weights = validate_stage_loss_weights(stage_loss_weights)
    stage_name = resolve_example_stage_name(dataset_spec, example)
    return float(weights.get(stage_name, 1.0))


def select_adaptive_curriculum_training_example(
    dataset_spec: ArithmeticRuleDatasetSpec,
    *,
    active_stage_index: int,
    local_step: int,
    replay_ratio: float = DEFAULT_REPLAY_RATIO,
    training_strategy: str = BASELINE_TRAINING_STRATEGY,
    carry_replay_ratio: float = DEFAULT_CARRY_REPLAY_RATIO,
) -> ArithmeticExample:
    """Select a training example using cumulative replay.

    中文说明:
    - 调用方 / Called by: `run_one_arithmetic_emergence_curve`, tests.
    - 调用对象 / Calls: `len` and `ArithmeticRuleDatasetSpec` fields.
    - 作用 / Purpose: 自适应课程只从当前阶段采样, 直到该阶段 EM 达标后外层推进。
    - 变量 / Variables: `active_stage_index` 是当前阶段编号, `local_step` 是阶段内轮转步。
    - 接入 / Integration: 训练循环维护阶段索引后调用本函数取样。
    - 错误处理 / Error handling: 空训练集、空阶段或越界阶段抛出 `ValueError`。
    - 关键词 / Keywords:
      adaptive|curriculum|sampling|active_stage|local_step|decimal|addition|mhdsra2|采样|阶段
    """
    if not dataset_spec.curriculum_stages:
        if not dataset_spec.training_examples:
            raise ValueError("Arithmetic training examples must not be empty.")
        return dataset_spec.training_examples[local_step % len(dataset_spec.training_examples)]
    if replay_ratio < 0.0 or replay_ratio >= 1.0:
        raise ValueError("replay_ratio must be in the [0.0, 1.0) range.")
    if carry_replay_ratio < 0.0 or carry_replay_ratio > 1.0:
        raise ValueError("carry_replay_ratio must be in the [0.0, 1.0] range.")
    validate_training_strategy(training_strategy)
    if active_stage_index < 0 or active_stage_index >= len(dataset_spec.curriculum_stages):
        raise ValueError(f"Active curriculum stage index is out of range: {active_stage_index}")
    stage = dataset_spec.curriculum_stages[active_stage_index]
    if not stage.examples:
        raise ValueError(f"Curriculum stage is empty: {stage.name}")
    replay_examples = tuple(
        example
        for replay_stage in dataset_spec.curriculum_stages[:active_stage_index]
        for example in replay_stage.examples
    )
    if not replay_examples or replay_ratio == 0.0:
        return stage.examples[local_step % len(stage.examples)]
    replay_events_before = int(local_step * replay_ratio)
    replay_events_after = int((local_step + 1) * replay_ratio)
    if replay_events_after > replay_events_before:
        carry_replay_examples = tuple(
            example
            for replay_stage in dataset_spec.curriculum_stages[: active_stage_index + 1]
            if replay_stage.name == UNIT_WITH_CARRY_STAGE
            for example in replay_stage.examples
        )
        two_digit_replay_examples = tuple(
            example
            for replay_stage in dataset_spec.curriculum_stages[: active_stage_index + 1]
            if replay_stage.name == TWO_DIGIT_RULES_STAGE
            for example in replay_stage.examples
        )
        carry_events_before = int(replay_events_before * carry_replay_ratio)
        carry_events_after = int((replay_events_before + 1) * carry_replay_ratio)
        two_digit_events_before = int(replay_events_before * carry_replay_ratio)
        two_digit_events_after = int((replay_events_before + 1) * carry_replay_ratio)
        if (
            uses_two_digit_replay(training_strategy)
            and two_digit_replay_examples
            and two_digit_events_after > two_digit_events_before
        ):
            return two_digit_replay_examples[
                two_digit_events_before % len(two_digit_replay_examples)
            ]
        if (
            uses_carry_replay(training_strategy)
            and carry_replay_examples
            and carry_events_after > carry_events_before
        ):
            return carry_replay_examples[
                carry_events_before % len(carry_replay_examples)
            ]
        return replay_examples[replay_events_before % len(replay_examples)]
    current_events_before = local_step - replay_events_before
    return stage.examples[current_events_before % len(stage.examples)]


def select_curriculum_training_example(
    dataset_spec: ArithmeticRuleDatasetSpec,
    *,
    step: int,
    training_steps: int,
) -> ArithmeticExample:
    """Select the training example for one curriculum step.

    中文说明:
    - 调用方 / Called by: `run_one_arithmetic_emergence_curve`, tests.
    - 调用对象 / Calls: `len`, integer arithmetic and `ArithmeticRuleDatasetSpec` fields.
    - 作用 / Purpose: 将总训练步数按课程阶段切分, 先训练个位无进位,
      再训练进位, 最后训练两位数；无课程规约时保持旧的循环采样。
    - 变量 / Variables: `stage_count` 是阶段数, `stage_index` 是当前阶段,
      `local_step` 是阶段内样例轮转位置。
    - 接入 / Integration: 所有算术涌现实验训练循环统一通过本函数选样本。
    - 错误处理 / Error handling: 空训练集或非正训练步数抛出 `ValueError`。
    - 关键词 / Keywords:
      curriculum|sampling|stage|training_step|decimal|addition|mhdsra2|application|课程|采样
    """
    if training_steps <= 0:
        raise ValueError("Curriculum training_steps must be positive.")
    if not dataset_spec.training_examples:
        raise ValueError("Arithmetic training examples must not be empty.")
    if not dataset_spec.curriculum_stages:
        return dataset_spec.training_examples[step % len(dataset_spec.training_examples)]

    stage_count = len(dataset_spec.curriculum_stages)
    stage_index = min((step * stage_count) // training_steps, stage_count - 1)
    stage = dataset_spec.curriculum_stages[stage_index]
    if not stage.examples:
        raise ValueError(f"Curriculum stage is empty: {stage.name}")
    first_stage_step = (training_steps * stage_index) // stage_count
    local_step = step - first_stage_step
    return stage.examples[local_step % len(stage.examples)]


def run_one_arithmetic_emergence_curve(
    *,
    dataset_spec: ArithmeticRuleDatasetSpec,
    model_name: str,
    seed: int,
    num_layers: int,
    max_steps_per_stage: int,
    curriculum_eval_interval: int,
    stage_threshold: float,
    replay_ratio: float,
    stage_patience: int,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    training_strategy: str = BASELINE_TRAINING_STRATEGY,
    carry_replay_ratio: float = DEFAULT_CARRY_REPLAY_RATIO,
    stage_loss_weights: Mapping[str, float] | None = None,
    device: str | torch.device = "cpu",
) -> ArithmeticEmergenceRun:
    """Train and evaluate one arithmetic emergence experiment point.

    中文说明:
    - 调用方 / Called by: `run_arithmetic_emergence_curves`.
    - 调用对象 / Calls: `build_arithmetic_model`, `encode_training_example`,
      `evaluate_arithmetic_examples`, `AdamW.step`.
    - 作用 / Purpose: 在一个 seed/层数/数据集上训练并执行完整 greedy 外推评估。
    - 变量 / Variables:
      `dataset_spec` 是训练和测试规约, `model` 是当前被测模型,
      `final_loss` 是最后一次 teacher-forced 训练损失。
    - 接入 / Integration: CLI、测试、报告均通过本函数获得单点结果。
    - 错误处理 / Error handling: 数据泄漏、训练或生成异常直接抛出。
    - 关键词 / Keywords:
      train|evaluate|arithmetic|emergence|seed|layers|mhdsra2|decimal|application|训练
    """
    if max_steps_per_stage <= 0:
        raise ValueError("max_steps_per_stage must be positive.")
    if curriculum_eval_interval <= 0:
        raise ValueError("curriculum_eval_interval must be positive.")
    if replay_ratio < 0.0 or replay_ratio >= 1.0:
        raise ValueError("replay_ratio must be in the [0.0, 1.0) range.")
    if stage_patience <= 0:
        raise ValueError("stage_patience must be positive.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if carry_replay_ratio < 0.0 or carry_replay_ratio > 1.0:
        raise ValueError("carry_replay_ratio must be in the [0.0, 1.0] range.")
    normalized_training_strategy = validate_training_strategy(training_strategy)
    validated_stage_loss_weights = validate_stage_loss_weights(stage_loss_weights)
    resolved_device = resolve_torch_device(device)
    dataset_spec.validate_training_scope()
    random.seed(seed + num_layers)
    torch.manual_seed(seed + num_layers)
    tokenizer = DecimalArithmeticTokenizer()
    model = build_arithmetic_model(
        model_name=model_name,
        num_layers=num_layers,
        vocab_size=tokenizer.vocab_size,
    ).to(resolved_device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    final_loss = 0.0
    active_stage_index = 0
    active_stage_local_step = 0
    curriculum_snapshots: list[ArithmeticCurriculumSnapshot] = []
    stage_exact_matches: tuple[ArithmeticStageMetric, ...] = ()
    training_steps_executed = 0
    stopped_reason = "max_steps_exhausted"
    stage_success_streak = 0
    stage_exhaustion_streak = 0
    stage_count = len(dataset_spec.curriculum_stages)
    patient_stage_limit = max_steps_per_stage + curriculum_eval_interval * (stage_patience - 1)
    max_total_steps = (
        patient_stage_limit * stage_count
        if dataset_spec.curriculum_stages
        else max_steps_per_stage
    )
    for step in range(max_total_steps):
        example = select_adaptive_curriculum_training_example(
            dataset_spec,
            active_stage_index=active_stage_index,
            local_step=active_stage_local_step,
            replay_ratio=replay_ratio,
            training_strategy=normalized_training_strategy,
            carry_replay_ratio=carry_replay_ratio,
        )
        active_stage_local_step += 1
        inputs, targets = encode_training_example(tokenizer, example, resolved_device)
        model.train()
        optimizer.zero_grad()
        logits = model(inputs)
        base_loss = criterion(logits.reshape(-1, tokenizer.vocab_size), targets.reshape(-1))
        loss = base_loss * resolve_stage_loss_multiplier(
            dataset_spec=dataset_spec,
            example=example,
            training_strategy=normalized_training_strategy,
            stage_loss_weights=validated_stage_loss_weights,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.item())
        training_steps_executed = step + 1
        should_record_curve = (
            dataset_spec.curriculum_stages
            and (
                active_stage_local_step % curriculum_eval_interval == 0
                or active_stage_local_step >= max_steps_per_stage
            )
        )
        if should_record_curve:
            stage_exact_matches = evaluate_curriculum_stage_metrics(
                model,
                tokenizer,
                dataset_spec,
                resolved_device,
            )
            active_stage = dataset_spec.curriculum_stages[active_stage_index]
            advanced_to_stage_name = None
            open_stage_names = tuple(
                stage.name for stage in dataset_spec.curriculum_stages[: active_stage_index + 1]
            )
            if should_advance_open_curriculum_stages(
                stage_exact_matches,
                open_stage_names,
                threshold=stage_threshold,
            ):
                stage_success_streak += 1
                stage_exhaustion_streak = 0
                if (
                    stage_success_streak >= stage_patience
                    and active_stage_index + 1 < len(dataset_spec.curriculum_stages)
                ):
                    active_stage_index += 1
                    active_stage_local_step = 0
                    stage_success_streak = 0
                    stage_exhaustion_streak = 0
                    advanced_to_stage_name = dataset_spec.curriculum_stages[
                        active_stage_index
                    ].name
                elif stage_success_streak >= stage_patience:
                    stopped_reason = "all_curriculum_stages_met_threshold"
            else:
                stage_success_streak = 0
                if active_stage_local_step >= max_steps_per_stage:
                    stage_exhaustion_streak += 1
                else:
                    stage_exhaustion_streak = 0
            curriculum_snapshots.append(
                ArithmeticCurriculumSnapshot(
                    dataset_name=dataset_spec.name,
                    model_name=model_name,
                    seed=seed,
                    num_layers=num_layers,
                    step=step + 1,
                    active_stage_name=active_stage.name,
                    advanced_to_stage_name=advanced_to_stage_name,
                    stage_exact_matches=stage_exact_matches,
                )
            )
            if stopped_reason == "all_curriculum_stages_met_threshold":
                break
            if stage_exhaustion_streak >= stage_patience:
                stopped_reason = f"stage_max_steps_exhausted:{active_stage.name}"
                break

    train_exact_match, _ = evaluate_arithmetic_examples(
        model,
        tokenizer,
        dataset_spec.training_examples,
        resolved_device,
    )
    if dataset_spec.curriculum_stages:
        stage_exact_matches = evaluate_curriculum_stage_metrics(
            model, tokenizer, dataset_spec, resolved_device
        )
    elif dataset_spec.diagnostic_stages:
        stage_exact_matches = evaluate_arithmetic_stage_metrics(
            model, tokenizer, dataset_spec.diagnostic_stages, resolved_device
        )
    stage_names = tuple(stage.name for stage in dataset_spec.curriculum_stages)
    ever_passed_stage_count = count_ever_passed_curriculum_stages(
        curriculum_snapshots,
        stage_names,
        threshold=stage_threshold,
    )
    retained_stage_count = count_completed_curriculum_stages(
        stage_exact_matches,
        threshold=stage_threshold,
    )
    headline_exact_match, headline_answers = evaluate_arithmetic_examples(
        model,
        tokenizer,
        (dataset_spec.headline_example,),
        resolved_device,
    )
    ood_exact_match, _ = evaluate_arithmetic_examples(
        model,
        tokenizer,
        dataset_spec.ood_examples,
        resolved_device,
    )
    headline_answer = headline_answers[0]
    return ArithmeticEmergenceRun(
        dataset_name=dataset_spec.name,
        model_name=model_name,
        seed=seed,
        num_layers=num_layers,
        learning_rate=learning_rate,
        training_strategy=normalized_training_strategy,
        carry_replay_ratio=carry_replay_ratio,
        train_exact_match=train_exact_match,
        headline_exact_match=headline_exact_match,
        ood_exact_match=ood_exact_match,
        final_loss=final_loss,
        headline_prediction=headline_answer.text,
        headline_stopped_on_eos=headline_answer.stopped_on_eos,
        completed_curriculum_stages=retained_stage_count,
        ever_passed_stage_count=ever_passed_stage_count,
        retained_stage_count=retained_stage_count,
        training_steps_executed=training_steps_executed,
        stopped_reason=stopped_reason,
        stage_exact_matches=stage_exact_matches,
        curriculum_snapshots=tuple(curriculum_snapshots),
    )


def run_arithmetic_emergence_curves(
    *,
    dataset_specs: Sequence[ArithmeticRuleDatasetSpec],
    layer_counts: Sequence[int],
    seeds: Sequence[int],
    max_steps_per_stage: int,
    curriculum_eval_interval: int,
    stage_threshold: float,
    replay_ratio: float,
    stage_patience: int,
    include_standard_baseline: bool,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    training_strategy: str = BASELINE_TRAINING_STRATEGY,
    carry_replay_ratio: float = DEFAULT_CARRY_REPLAY_RATIO,
    stage_loss_weights: Mapping[str, float] | None = None,
    device: str | torch.device = "cpu",
) -> list[ArithmeticEmergenceRun]:
    """Run all requested arithmetic emergence curve points.

    中文说明:
    - 调用方 / Called by: `build_layer_emergence_payload`.
    - 调用对象 / Calls: `run_one_arithmetic_emergence_curve`.
    - 作用 / Purpose: 扫描主实验、负控、标准注意力参考和 MHDSRA2 层数。
    - 变量 / Variables:
      `dataset_specs` 是实验规约集合, `layer_counts` 是 MHDSRA2 候选层数,
      `include_standard_baseline` 控制是否输出参考基线。
    - 接入 / Integration: 报告入口直接调用本函数。
    - 错误处理 / Error handling: 单点失败直接终止, 不静默跳过。
    - 关键词 / Keywords:
      curves|scan|arithmetic|emergence|layers|seeds|baseline|mhdsra2|application|扫描
    """
    runs: list[ArithmeticEmergenceRun] = []
    for dataset_spec in dataset_specs:
        for seed in seeds:
            if include_standard_baseline:
                runs.append(
                    run_one_arithmetic_emergence_curve(
                        dataset_spec=dataset_spec,
                        model_name=STANDARD_ATTENTION_MODEL,
                        seed=seed,
                        num_layers=1,
                        max_steps_per_stage=max_steps_per_stage,
                        curriculum_eval_interval=curriculum_eval_interval,
                        stage_threshold=stage_threshold,
                        replay_ratio=replay_ratio,
                        stage_patience=stage_patience,
                        learning_rate=learning_rate,
                        training_strategy=training_strategy,
                        carry_replay_ratio=carry_replay_ratio,
                        stage_loss_weights=stage_loss_weights,
                        device=device,
                    )
                )
            for num_layers in layer_counts:
                runs.append(
                    run_one_arithmetic_emergence_curve(
                        dataset_spec=dataset_spec,
                        model_name=MHDSRA2_MODEL,
                        seed=seed,
                        num_layers=num_layers,
                        max_steps_per_stage=max_steps_per_stage,
                        curriculum_eval_interval=curriculum_eval_interval,
                        stage_threshold=stage_threshold,
                        replay_ratio=replay_ratio,
                        stage_patience=stage_patience,
                        learning_rate=learning_rate,
                        training_strategy=training_strategy,
                        carry_replay_ratio=carry_replay_ratio,
                        stage_loss_weights=stage_loss_weights,
                        device=device,
                    )
                )
    return runs


def _mean_and_variance(values: Sequence[float]) -> tuple[float, float]:
    """Return population mean and variance for metric samples.

    中文说明:
    - 调用方 / Called by: `aggregate_arithmetic_emergence_runs`.
    - 调用对象 / Calls: `statistics.fmean`, `statistics.pvariance`.
    - 作用 / Purpose: 统一多 seed 指标均值/方差计算。
    - 变量 / Variables: `values` 是同一指标跨 seed 样本。
    - 接入 / Integration: 新增聚合指标时复用本函数。
    - 错误处理 / Error handling: 空样本抛出 `ValueError`。
    - 关键词 / Keywords:
      mean|variance|statistics|aggregate|seed|arithmetic|mhdsra2|application|均值|方差
    """
    if not values:
        raise ValueError("Metric values must not be empty.")
    mean_value = float(statistics.fmean(values))
    variance_value = float(statistics.pvariance(values)) if len(values) > 1 else 0.0
    return mean_value, variance_value


def _stage_metric_value(
    stage_metrics: Sequence[ArithmeticStageMetric],
    stage_name: str,
) -> float:
    """Read one stage exact-match value from a metric sequence.

    中文说明:
    - 调用方 / Called by: `find_curriculum_stage_pass_step`,
      `aggregate_curriculum_stage_progress`.
    - 调用对象 / Calls: none; only iterates strongly typed metric objects.
    - 作用 / Purpose: 避免在聚合逻辑里重复手写阶段名查找。
    - 变量 / Variables: `stage_metrics` 是阶段指标序列, `stage_name` 是目标阶段名。
    - 接入 / Integration: 新增阶段指标字段时仍可复用本函数读取 EM。
    - 错误处理 / Error handling: 未找到阶段时返回 `0.0`, 表示该阶段未通过。
    - 关键词 / Keywords:
      stage_metric|lookup|exact_match|curriculum|aggregate|mhdsra2|application|查询|阶段|指标
    """
    for stage_metric in stage_metrics:
        if stage_metric.stage_name == stage_name:
            return stage_metric.exact_match
    return 0.0


def find_curriculum_stage_pass_step(
    run: ArithmeticEmergenceRun,
    stage_name: str,
    *,
    stage_threshold: float,
) -> int | None:
    """Find the first checkpoint step where a curriculum stage reaches threshold.

    中文说明:
    - 调用方 / Called by: `aggregate_curriculum_stage_progress`, tests.
    - 调用对象 / Calls: `_stage_metric_value`.
    - 作用 / Purpose: 将逐 seed 曲线压缩为阶段第一次达标步数。
    - 变量 / Variables: `run` 是单 seed 运行结果, `stage_name` 是待检查阶段,
      `stage_threshold` 是阶段通过阈值。
    - 接入 / Integration: 报告中的平均推进步数来自本函数输出。
    - 错误处理 / Error handling: 阶段未达标返回 `None`, 不伪造步数。
    - 关键词 / Keywords:
      pass_step|curriculum|stage|threshold|checkpoint|aggregate|mhdsra2|report|达标|步数
    """
    for snapshot in run.curriculum_snapshots:
        if _stage_metric_value(snapshot.stage_exact_matches, stage_name) >= stage_threshold:
            return snapshot.step
    return None


def aggregate_curriculum_stage_progress(
    runs: Sequence[ArithmeticEmergenceRun],
    *,
    stage_threshold: float,
) -> list[ArithmeticStageAggregate]:
    """Aggregate curriculum stage pass rate and pass-step statistics by layer.

    中文说明:
    - 调用方 / Called by: `build_layer_emergence_payload`, tests.
    - 调用对象 / Calls: `find_curriculum_stage_pass_step`, `_stage_metric_value`,
      `_mean_and_variance`, `ArithmeticStageAggregate`.
    - 作用 / Purpose: 生成按 dataset/model/layer/stage 分组的阶段通过率和平均推进步数。
    - 变量 / Variables: `groups` 是聚合键集合, `pass_steps` 是已通过 run 的达标步数。
    - 接入 / Integration: Markdown 报告优先展示本摘要, 原始 snapshots 仅保留在 JSON。
    - 错误处理 / Error handling: 空输入返回空列表, 未通过阶段的平均步数为 `None`。
    - 关键词 / Keywords:
      aggregate|curriculum|stage|pass_rate|advance_step|mean|variance|mhdsra2|聚合|报告
    """
    stage_names = sorted(
        {
            stage_metric.stage_name
            for run in runs
            for stage_metric in run.stage_exact_matches
        }
    )
    groups = sorted({(run.dataset_name, run.model_name, run.num_layers) for run in runs})
    aggregates: list[ArithmeticStageAggregate] = []
    for dataset_name, model_name, num_layers in groups:
        group_runs = [
            run
            for run in runs
            if run.dataset_name == dataset_name
            and run.model_name == model_name
            and run.num_layers == num_layers
        ]
        for stage_name in stage_names:
            stage_group_runs = [
                run
                for run in group_runs
                if any(
                    stage_metric.stage_name == stage_name
                    for stage_metric in run.stage_exact_matches
                )
            ]
            if not stage_group_runs:
                continue
            pass_steps: list[int] = []
            for run in stage_group_runs:
                pass_step = find_curriculum_stage_pass_step(
                    run,
                    stage_name,
                    stage_threshold=stage_threshold,
                )
                if pass_step is not None:
                    pass_steps.append(pass_step)
            final_values = [
                _stage_metric_value(run.stage_exact_matches, stage_name)
                for run in stage_group_runs
            ]
            final_mean, final_variance = _mean_and_variance(final_values)
            if pass_steps:
                pass_step_mean, pass_step_variance = _mean_and_variance(
                    [float(pass_step) for pass_step in pass_steps]
                )
            else:
                pass_step_mean = None
                pass_step_variance = None
            aggregates.append(
                ArithmeticStageAggregate(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    num_layers=num_layers,
                    stage_name=stage_name,
                    num_runs=len(stage_group_runs),
                    pass_rate=len(pass_steps) / len(stage_group_runs),
                    advance_step_mean=pass_step_mean,
                    advance_step_variance=pass_step_variance,
                    final_exact_match_mean=final_mean,
                    final_exact_match_variance=final_variance,
                )
            )
    return aggregates


def aggregate_arithmetic_emergence_runs(
    runs: Sequence[ArithmeticEmergenceRun],
) -> list[ArithmeticEmergenceResult]:
    """Aggregate arithmetic emergence runs by dataset/model/layer.

    中文说明:
    - 调用方 / Called by: `build_layer_emergence_payload`.
    - 调用对象 / Calls: `_mean_and_variance`, `ArithmeticEmergenceResult`.
    - 作用 / Purpose: 将单 seed 结果汇总成多 seed 均值/方差和成功判定。
    - 变量 / Variables: `groups` 是数据集/模型/层数组合, `group_runs` 是同组样本。
    - 接入 / Integration: 报告和最小层数判定读取本函数输出。
    - 错误处理 / Error handling: 空输入返回空列表。
    - 关键词 / Keywords:
      aggregate|arithmetic|emergence|mean|variance|success|mhdsra2|application|聚合|指标
    """
    aggregates: list[ArithmeticEmergenceResult] = []
    groups = sorted({(run.dataset_name, run.model_name, run.num_layers) for run in runs})
    for dataset_name, model_name, num_layers in groups:
        group_runs = [
            run
            for run in runs
            if run.dataset_name == dataset_name
            and run.model_name == model_name
            and run.num_layers == num_layers
        ]
        train_mean, train_variance = _mean_and_variance(
            [run.train_exact_match for run in group_runs]
        )
        headline_mean, headline_variance = _mean_and_variance(
            [run.headline_exact_match for run in group_runs]
        )
        ood_mean, ood_variance = _mean_and_variance(
            [run.ood_exact_match for run in group_runs]
        )
        loss_mean, loss_variance = _mean_and_variance([run.final_loss for run in group_runs])
        meets_success = (
            model_name == MHDSRA2_MODEL
            and dataset_name == CURRICULUM_RULE_SET
            and train_mean >= TRAIN_EXACT_MATCH_THRESHOLD
            and headline_mean >= HEADLINE_EXACT_MATCH_THRESHOLD
            and ood_mean >= OOD_EXACT_MATCH_THRESHOLD
        )
        aggregates.append(
            ArithmeticEmergenceResult(
                dataset_name=dataset_name,
                model_name=model_name,
                num_layers=num_layers,
                num_seeds=len(group_runs),
                train_exact_match_mean=train_mean,
                train_exact_match_variance=train_variance,
                headline_exact_match_mean=headline_mean,
                headline_exact_match_variance=headline_variance,
                ood_exact_match_mean=ood_mean,
                ood_exact_match_variance=ood_variance,
                final_loss_mean=loss_mean,
                final_loss_variance=loss_variance,
                meets_success_criteria=meets_success,
            )
        )
    return aggregates


def find_minimum_arithmetic_emergent_layers(
    aggregates: Sequence[ArithmeticEmergenceResult],
    layer_counts: Sequence[int],
) -> int | None:
    """Find the first MHDSRA2 depth that passes arithmetic emergence criteria.

    中文说明:
    - 调用方 / Called by: `build_layer_emergence_payload`, tests.
    - 调用对象 / Calls: 无；遍历聚合结果。
    - 作用 / Purpose: 按 `1/2/4/8/16` 顺序返回首个主实验达标层数。
    - 变量 / Variables: `aggregates` 是聚合结果, `layer_counts` 定义扫描顺序。
    - 接入 / Integration: 报告 summary 直接使用返回值。
    - 错误处理 / Error handling: 没有达标层数返回 `None`, 不强行给结论。
    - 关键词 / Keywords:
      minimum|layers|arithmetic|emergence|mhdsra2|100+100|ood|criteria|application|最小
    """
    result_by_layer = {
        result.num_layers: result
        for result in aggregates
        if result.dataset_name == CURRICULUM_RULE_SET and result.model_name == MHDSRA2_MODEL
    }
    for num_layers in layer_counts:
        result = result_by_layer.get(num_layers)
        if result is not None and result.meets_success_criteria:
            return num_layers
    return None


def find_minimum_curriculum_mastery_layers(
    runs: Sequence[ArithmeticEmergenceRun],
    layer_counts: Sequence[int],
    *,
    required_stage_count: int,
) -> int | None:
    """Find the first MHDSRA2 depth that retains every curriculum stage.

    中文说明:
    - 调用方 / Called by: `build_layer_emergence_payload`, tests.
    - 调用对象 / Calls: none; this function filters run records.
    - 作用 / Purpose: 将“先稳定完成三段 curriculum”作为百位外推前置目标。
    - 变量 / Variables: `required_stage_count` 是必须最终保留的阶段数量。
    - 接入 / Integration: 报告 summary 中的 `minimum_curriculum_mastery_layers` 来自本函数。
    - 错误处理 / Error handling: 没有任何层全 seed 达标时返回 `None`。
    - 关键词 / Keywords:
      curriculum|mastery|minimum_layers|retained|mhdsra2|stage_count|report|掌握|层数|保留
    """
    if required_stage_count <= 0:
        return None
    for num_layers in layer_counts:
        layer_runs = [
            run
            for run in runs
            if run.dataset_name == CURRICULUM_RULE_SET
            and run.model_name == MHDSRA2_MODEL
            and run.num_layers == num_layers
        ]
        if layer_runs and all(
            run.retained_stage_count >= required_stage_count for run in layer_runs
        ):
            return num_layers
    return None


def aggregate_curriculum_strategy_grid_runs(
    grid_runs: Sequence[ArithmeticStrategyGridRun],
    *,
    target_stage_count: int,
) -> list[ArithmeticStrategyGridResult]:
    """Aggregate curriculum strategy grid runs by strategy and layer.

    中文说明:
    - 调用方 / Called by: `build_curriculum_strategy_grid_payload`, tests.
    - 调用对象 / Calls: `_mean_and_variance`, `ArithmeticStrategyGridResult`.
    - 作用 / Purpose: 汇总 replay ratio 和 stage patience 网格中哪些策略能稳定保留目标阶段。
    - 变量 / Variables: `groups` 是 replay/patience/layer 组合, `group_runs` 是同组样本。
    - 接入 / Integration: 网格报告的核心摘要来自本函数。
    - 错误处理 / Error handling: 空输入返回空列表, 不伪造结果。
    - 关键词 / Keywords:
      grid|strategy|aggregate|replay_ratio|stage_patience|retention|mhdsra2|report|聚合|策略
    """
    results: list[ArithmeticStrategyGridResult] = []
    groups = sorted(
        {
            (
                grid_run.replay_ratio,
                grid_run.stage_patience,
                grid_run.max_steps_per_stage,
                grid_run.run.num_layers,
            )
            for grid_run in grid_runs
        }
    )
    for replay_ratio, stage_patience, max_steps_per_stage, num_layers in groups:
        group_runs = [
            grid_run.run
            for grid_run in grid_runs
            if grid_run.replay_ratio == replay_ratio
            and grid_run.stage_patience == stage_patience
            and grid_run.max_steps_per_stage == max_steps_per_stage
            and grid_run.run.num_layers == num_layers
        ]
        if not group_runs:
            continue
        retained_values = [float(run.retained_stage_count) for run in group_runs]
        ever_values = [float(run.ever_passed_stage_count) for run in group_runs]
        train_values = [run.train_exact_match for run in group_runs]
        loss_values = [run.final_loss for run in group_runs]
        retained_mean, retained_variance = _mean_and_variance(retained_values)
        ever_mean, ever_variance = _mean_and_variance(ever_values)
        train_mean, train_variance = _mean_and_variance(train_values)
        loss_mean, loss_variance = _mean_and_variance(loss_values)
        target_hits = sum(
            1 for run in group_runs if run.retained_stage_count >= target_stage_count
        )
        target_retention_rate = target_hits / len(group_runs)
        results.append(
            ArithmeticStrategyGridResult(
                replay_ratio=replay_ratio,
                stage_patience=stage_patience,
                max_steps_per_stage=max_steps_per_stage,
                num_layers=num_layers,
                num_runs=len(group_runs),
                target_stage_count=target_stage_count,
                target_retention_rate=target_retention_rate,
                stable_target_retention=target_hits == len(group_runs),
                retained_stage_count_mean=retained_mean,
                retained_stage_count_variance=retained_variance,
                ever_passed_stage_count_mean=ever_mean,
                ever_passed_stage_count_variance=ever_variance,
                train_exact_match_mean=train_mean,
                train_exact_match_variance=train_variance,
                final_loss_mean=loss_mean,
                final_loss_variance=loss_variance,
            )
        )
    return results


def find_best_curriculum_strategy_grid_result(
    grid_results: Sequence[ArithmeticStrategyGridResult],
) -> ArithmeticStrategyGridResult | None:
    """Return the strongest curriculum strategy grid result.

    中文说明:
    - 调用方 / Called by: `build_curriculum_strategy_grid_payload`.
    - 调用对象 / Calls: built-in `max`.
    - 作用 / Purpose: 为报告 summary 提供按目标保留率、保留阶段均值和训练 EM 排序的最佳策略。
    - 变量 / Variables: `grid_results` 是所有策略聚合结果。
    - 接入 / Integration: JSON summary 直接引用本函数返回的最佳策略。
    - 错误处理 / Error handling: 空结果返回 `None`。
    - 关键词 / Keywords:
      best_strategy|grid|ranking|retention|train_em|replay|patience|mhdsra2|最佳|排序
    """
    if not grid_results:
        return None
    return max(
        grid_results,
        key=lambda result: (
            result.target_retention_rate,
            result.retained_stage_count_mean,
            result.train_exact_match_mean,
            -result.final_loss_mean,
            -result.max_steps_per_stage,
        ),
    )


def build_layer_emergence_payload(
    *,
    layer_counts: Sequence[int] = DEFAULT_LAYER_COUNTS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    max_steps_per_stage: int = DEFAULT_ARITHMETIC_EMERGENCE_MAX_STEPS_PER_STAGE,
    curriculum_eval_interval: int = DEFAULT_CURRICULUM_EVAL_INTERVAL,
    stage_threshold: float = CURRICULUM_STAGE_EXACT_MATCH_THRESHOLD,
    replay_ratio: float = DEFAULT_ARITHMETIC_EMERGENCE_REPLAY_RATIO,
    stage_patience: int = DEFAULT_ARITHMETIC_EMERGENCE_STAGE_PATIENCE,
    learning_rate: float = DEFAULT_ARITHMETIC_EMERGENCE_LEARNING_RATE,
    device: str | torch.device = DEFAULT_ARITHMETIC_EMERGENCE_DEVICE,
    include_standard_baseline: bool = True,
    training_steps: int | None = None,
) -> dict[str, object]:
    """Build the JSON-serializable decimal arithmetic emergence payload.

    中文说明:
    - 调用方 / Called by: CLI script and tests.
    - 调用对象 / Calls: `build_default_arithmetic_specs`,
      `run_arithmetic_emergence_curves`, `aggregate_arithmetic_emergence_runs`.
    - 作用 / Purpose: 生成训练规约、单 seed 结果、聚合结果和最小涌现层数。
    - 变量 / Variables:
      `dataset_specs` 是主实验和负控, `runs` 是单次结果,
      `aggregates` 是均值/方差, `minimum_layers` 是最终答案。
    - 接入 / Integration: 报告脚本和测试统一使用本函数。
    - 错误处理 / Error handling: 下游异常直接抛出。
    - 关键词 / Keywords:
      payload|arithmetic|emergence|decimal|100+100|mhdsra2|json|report|application|数据
    """
    resolved_max_steps_per_stage = (
        max_steps_per_stage if training_steps is None else training_steps
    )
    resolved_device = resolve_torch_device(device)
    dataset_specs = build_default_arithmetic_specs()
    for dataset_spec in dataset_specs:
        dataset_spec.validate_training_scope()
    runs = run_arithmetic_emergence_curves(
        dataset_specs=dataset_specs,
        layer_counts=layer_counts,
        seeds=seeds,
        max_steps_per_stage=resolved_max_steps_per_stage,
        curriculum_eval_interval=curriculum_eval_interval,
        stage_threshold=stage_threshold,
        replay_ratio=replay_ratio,
        stage_patience=stage_patience,
        include_standard_baseline=include_standard_baseline,
        learning_rate=learning_rate,
        device=resolved_device,
    )
    aggregates = aggregate_arithmetic_emergence_runs(runs)
    stage_aggregates = aggregate_curriculum_stage_progress(
        runs,
        stage_threshold=stage_threshold,
    )
    required_stage_count = max(
        (len(dataset_spec.curriculum_stages) for dataset_spec in dataset_specs),
        default=0,
    )
    minimum_curriculum_mastery_layers = find_minimum_curriculum_mastery_layers(
        runs,
        layer_counts,
        required_stage_count=required_stage_count,
    )
    minimum_layers = find_minimum_arithmetic_emergent_layers(aggregates, layer_counts)
    return {
        "config": {
            "layer_counts": list(layer_counts),
            "seeds": list(seeds),
            "max_steps_per_stage": resolved_max_steps_per_stage,
            "training_steps": resolved_max_steps_per_stage,
            "curriculum_eval_interval": curriculum_eval_interval,
            "replay_ratio": replay_ratio,
            "stage_patience": stage_patience,
            "learning_rate": learning_rate,
            "device": str(resolved_device),
            "include_standard_baseline": include_standard_baseline,
            "train_exact_match_threshold": TRAIN_EXACT_MATCH_THRESHOLD,
            "curriculum_stage_exact_match_threshold": stage_threshold,
            "headline_exact_match_threshold": HEADLINE_EXACT_MATCH_THRESHOLD,
            "ood_exact_match_threshold": OOD_EXACT_MATCH_THRESHOLD,
        },
        "datasets": [
            {
                "name": spec.name,
                "training_examples": [example.equation for example in spec.training_examples],
                "curriculum_stages": [
                    {
                        "name": stage.name,
                        "examples": [example.equation for example in stage.examples],
                    }
                    for stage in spec.curriculum_stages
                ],
                "headline_example": spec.headline_example.equation,
                "ood_examples": [example.equation for example in spec.ood_examples],
            }
            for spec in dataset_specs
        ],
        "summary": {
            "minimum_curriculum_mastery_layers": minimum_curriculum_mastery_layers,
            "minimum_arithmetic_emergent_layers": minimum_layers,
            "proxy_definition": (
                "first report curriculum mastery over unit_no_carry, unit_with_carry, "
                "and two_digit_rules; only then interpret 100+100 headline/OOD "
                "arithmetic emergence. "
                "minimum MHDSRA2 layer count whose curriculum_rule_set aggregate has "
                "train_exact_match_mean >= 0.95, headline_exact_match_mean == 1.0, "
                "and ood_exact_match_mean >= 0.80"
            ),
        },
        "aggregates": [asdict(result) for result in aggregates],
        "curriculum_stage_aggregates": [
            asdict(stage_result) for stage_result in stage_aggregates
        ],
        "runs": [asdict(run) for run in runs],
    }


def build_curriculum_strategy_grid_payload(
    *,
    replay_ratios: Sequence[float] = DEFAULT_STRATEGY_GRID_REPLAY_RATIOS,
    stage_patiences: Sequence[int] = DEFAULT_STRATEGY_GRID_STAGE_PATIENCES,
    layer_counts: Sequence[int] = DEFAULT_STRATEGY_GRID_LAYER_COUNTS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    max_steps_per_stage: int = DEFAULT_MAX_STEPS_PER_STAGE,
    max_steps_per_stage_values: Sequence[int] | None = None,
    curriculum_eval_interval: int = DEFAULT_CURRICULUM_EVAL_INTERVAL,
    stage_threshold: float = CURRICULUM_STAGE_EXACT_MATCH_THRESHOLD,
    target_stage_count: int = 2,
) -> dict[str, object]:
    """Build the curriculum strategy grid scan payload.

    中文说明:
    - 调用方 / Called by: CLI script and tests.
    - 调用对象 / Calls: `build_curriculum_arithmetic_spec`,
      `run_one_arithmetic_emergence_curve`, `aggregate_curriculum_strategy_grid_runs`.
    - 作用 / Purpose: 扫描 replay ratio 与 stage patience, 判断是否稳定保留目标课程阶段。
    - 变量 / Variables: `grid_runs` 是所有单 seed 运行, `grid_results` 是策略聚合结果。
    - 接入 / Integration: 写入 `reports/mhdsra2_curriculum_strategy_grid.*`。
    - 错误处理 / Error handling: 训练、参数或数据规约错误直接抛出, 不静默忽略。
    - 关键词 / Keywords:
      payload|grid|strategy|replay_ratio|stage_patience|retention|mhdsra2|json|策略|扫描
    """
    dataset_spec = build_curriculum_arithmetic_spec()
    dataset_spec.validate_training_scope()
    step_budgets = (
        (max_steps_per_stage,)
        if max_steps_per_stage_values is None
        else tuple(max_steps_per_stage_values)
    )
    if not step_budgets:
        raise ValueError("max_steps_per_stage_values must not be empty.")
    grid_runs: list[ArithmeticStrategyGridRun] = []
    for replay_ratio in replay_ratios:
        for stage_patience in stage_patiences:
            for step_budget in step_budgets:
                for num_layers in layer_counts:
                    for seed in seeds:
                        run = run_one_arithmetic_emergence_curve(
                            dataset_spec=dataset_spec,
                            model_name=MHDSRA2_MODEL,
                            seed=seed,
                            num_layers=num_layers,
                            max_steps_per_stage=step_budget,
                            curriculum_eval_interval=curriculum_eval_interval,
                            stage_threshold=stage_threshold,
                            replay_ratio=replay_ratio,
                            stage_patience=stage_patience,
                        )
                        grid_runs.append(
                            ArithmeticStrategyGridRun(
                                replay_ratio=replay_ratio,
                                stage_patience=stage_patience,
                                max_steps_per_stage=step_budget,
                                run=run,
                            )
                        )
    grid_results = aggregate_curriculum_strategy_grid_runs(
        grid_runs,
        target_stage_count=target_stage_count,
    )
    best_result = find_best_curriculum_strategy_grid_result(grid_results)
    stable_results = [result for result in grid_results if result.stable_target_retention]
    return {
        "config": {
            "replay_ratios": list(replay_ratios),
            "stage_patiences": list(stage_patiences),
            "layer_counts": list(layer_counts),
            "seeds": list(seeds),
            "max_steps_per_stage": step_budgets[0] if len(step_budgets) == 1 else None,
            "max_steps_per_stage_values": list(step_budgets),
            "curriculum_eval_interval": curriculum_eval_interval,
            "stage_threshold": stage_threshold,
            "target_stage_count": target_stage_count,
            "target_stages": [
                stage.name for stage in dataset_spec.curriculum_stages[:target_stage_count]
            ],
        },
        "dataset": {
            "name": dataset_spec.name,
            "curriculum_stages": [
                {
                    "name": stage.name,
                    "examples": [example.equation for example in stage.examples],
                }
                for stage in dataset_spec.curriculum_stages
            ],
        },
        "summary": {
            "has_stable_target_strategy": bool(stable_results),
            "stable_target_strategy_count": len(stable_results),
            "best_strategy": None if best_result is None else asdict(best_result),
        },
        "grid_results": [asdict(result) for result in grid_results],
        "grid_runs": [asdict(grid_run) for grid_run in grid_runs],
    }


def build_curriculum_strategy_grid_markdown(payload: dict[str, object]) -> list[str]:
    """Build Markdown lines for the curriculum strategy grid report.

    中文说明:
    - 调用方 / Called by: CLI script.
    - 调用对象 / Calls: none; reads payload fields and renders Markdown.
    - 作用 / Purpose: 用紧凑表格展示 replay/patience 网格是否稳定保留目标阶段。
    - 变量 / Variables: `payload` 是网格报告数据, `grid_results` 是聚合结果。
    - 接入 / Integration: 报告格式变更集中修改本函数。
    - 错误处理 / Error handling: 缺字段或错误类型抛出 `KeyError`/`TypeError`。
    - 关键词 / Keywords:
      markdown|grid|strategy|replay_ratio|stage_patience|retention|mhdsra2|report|渲染|报告
    """
    config = payload["config"]
    summary = payload["summary"]
    grid_results = payload["grid_results"]
    if not isinstance(config, dict) or not isinstance(summary, dict):
        raise TypeError("grid payload config and summary must be dictionaries.")
    if not isinstance(grid_results, list):
        raise TypeError("grid payload results must be a list.")
    target_stages = ", ".join(str(stage) for stage in config["target_stages"])
    best_strategy = summary["best_strategy"]
    if isinstance(best_strategy, dict):
        best_strategy_text = (
            f"replay_ratio={best_strategy['replay_ratio']}, "
            f"stage_patience={best_strategy['stage_patience']}, "
            f"max_steps_per_stage={best_strategy['max_steps_per_stage']}, "
            f"layers={best_strategy['num_layers']}, "
            f"target_retention_rate={best_strategy['target_retention_rate']:.4f}"
        )
    else:
        best_strategy_text = "null"
    lines = [
        "# MHDSRA2 Curriculum Strategy Grid",
        "",
        "## Objective",
        "",
        (
            "Find whether replay ratio and stage patience can stably retain "
            f"the target curriculum stages: {target_stages}."
        ),
        "",
        "## Configuration",
        "",
        f"- Replay ratios: {', '.join(str(item) for item in config['replay_ratios'])}",
        f"- Stage patiences: {', '.join(str(item) for item in config['stage_patiences'])}",
        f"- Layers: {', '.join(str(item) for item in config['layer_counts'])}",
        f"- Seeds: {', '.join(str(item) for item in config['seeds'])}",
        (
            "- Max steps per stage values: "
            f"{', '.join(str(item) for item in config['max_steps_per_stage_values'])}"
        ),
        f"- Curriculum eval interval: {config['curriculum_eval_interval']}",
        f"- Stage threshold: {config['stage_threshold']}",
        f"- Target stage count: {config['target_stage_count']}",
        "",
        "## Summary",
        "",
        f"- Stable target strategy count: {summary['stable_target_strategy_count']}",
        f"- Has stable target strategy: {summary['has_stable_target_strategy']}",
        f"- Best strategy: {best_strategy_text}",
        "",
        "## Grid Results",
        "",
        "| Replay Ratio | Stage Patience | Max Steps | Layers | Runs | Target Retention Rate | "
        "Stable | Retained Mean | Ever Passed Mean | Train EM Mean | Final Loss Mean |",
        "|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|",
    ]
    for row in grid_results:
        if not isinstance(row, dict):
            raise TypeError("grid result rows must be dictionaries.")
        stable_text = "yes" if row["stable_target_retention"] else "no"
        lines.append(
            f"| {row['replay_ratio']:.2f} | {row['stage_patience']} | "
            f"{row['max_steps_per_stage']} | {row['num_layers']} | "
            f"{row['num_runs']} | {row['target_retention_rate']:.4f} | {stable_text} | "
            f"{row['retained_stage_count_mean']:.4f} | "
            f"{row['ever_passed_stage_count_mean']:.4f} | "
            f"{row['train_exact_match_mean']:.4f} | {row['final_loss_mean']:.4f} |"
        )
    return lines


def build_carry_diagnostic_dataset_specs() -> tuple[ArithmeticRuleDatasetSpec, ...]:
    """Build datasets used by the carry diagnostic grid.

    中文说明:
    - 调用方 / Called by: carry diagnostic payload builders, CLI and tests.
    - 调用对象 / Calls: `build_curriculum_arithmetic_spec`, `build_unit_with_carry_only_spec`.
    - 作用 / Purpose: 同时提供完整 curriculum 与 isolated carry-only 两类诊断数据。
    - 变量 / Variables: 返回 tuple 顺序固定，先 curriculum 后 carry-only。
    - 接入 / Integration: 新增诊断数据集时在本函数集中扩展。
    - 错误处理 / Error handling: 下游会逐个执行 `validate_training_scope`。
    - 关键词 / Keywords:
      carry_diagnostic|datasets|curriculum|unit_with_carry_only|mhdsra2|arithmetic|诊断|数据集
    """
    return (build_curriculum_arithmetic_spec(), build_unit_with_carry_only_spec())


def run_one_carry_diagnostic_grid_point(
    *,
    dataset_spec: ArithmeticRuleDatasetSpec,
    training_strategy: str,
    learning_rate: float,
    curriculum_eval_interval: int,
    max_steps_per_stage: int,
    num_layers: int,
    seed: int,
    replay_ratio: float,
    stage_patience: int,
    carry_replay_ratio: float,
    stage_loss_weights: Mapping[str, float] | None,
) -> ArithmeticCarryDiagnosticRun:
    """Run one point in the carry diagnostic grid.

    中文说明:
    - 调用方 / Called by: carry diagnostic payload builder and CLI checkpoint loop.
    - 调用对象 / Calls: `run_one_arithmetic_emergence_curve`, `validate_training_strategy`.
    - 作用 / Purpose: 用强类型封装一次 carry 诊断训练点，避免脚本层拼装结果。
    - 变量 / Variables: `dataset_spec` 是诊断数据集, `training_strategy` 是强化策略。
    - 接入 / Integration: CLI 每完成一个点就把返回值序列化到 checkpoint。
    - 错误处理 / Error handling: 训练参数非法或模型失败时直接抛出异常。
    - 关键词 / Keywords:
      carry_diagnostic|run_point|learning_rate|strategy|eval_interval|checkpoint|mhdsra2|grid|运行|进位
    """
    normalized_strategy = validate_training_strategy(training_strategy)
    run = run_one_arithmetic_emergence_curve(
        dataset_spec=dataset_spec,
        model_name=MHDSRA2_MODEL,
        seed=seed,
        num_layers=num_layers,
        max_steps_per_stage=max_steps_per_stage,
        curriculum_eval_interval=curriculum_eval_interval,
        stage_threshold=CURRICULUM_STAGE_EXACT_MATCH_THRESHOLD,
        replay_ratio=replay_ratio,
        stage_patience=stage_patience,
        learning_rate=learning_rate,
        training_strategy=normalized_strategy,
        carry_replay_ratio=carry_replay_ratio,
        stage_loss_weights=stage_loss_weights,
    )
    return ArithmeticCarryDiagnosticRun(
        dataset_name=dataset_spec.name,
        training_strategy=normalized_strategy,
        learning_rate=learning_rate,
        curriculum_eval_interval=curriculum_eval_interval,
        max_steps_per_stage=max_steps_per_stage,
        num_layers=num_layers,
        seed=seed,
        replay_ratio=replay_ratio,
        stage_patience=stage_patience,
        carry_replay_ratio=carry_replay_ratio,
        run=run,
    )


def serialize_carry_diagnostic_run(
    diagnostic_run: ArithmeticCarryDiagnosticRun,
) -> dict[str, object]:
    """Serialize one carry diagnostic run for JSON reports and checkpoints.

    中文说明:
    - 调用方 / Called by: carry diagnostic payload builder, CLI and tests.
    - 调用对象 / Calls: `dataclasses.asdict`.
    - 作用 / Purpose: 保证 checkpoint JSONL 与最终报告使用同一字段结构。
    - 变量 / Variables: `diagnostic_run` 是强类型单点结果。
    - 接入 / Integration: 每个 grid cell 完成后调用本函数再写入 JSONL。
    - 错误处理 / Error handling: dataclass 序列化错误直接抛出。
    - 关键词 / Keywords:
      serialize|checkpoint|jsonl|carry_diagnostic|run|mhdsra2|report|grid|序列化|检查点
    """
    return asdict(diagnostic_run)


def _required_mapping(value: object, field_name: str) -> Mapping[str, object]:
    """Read a nested mapping from a JSON-like diagnostic row.

    中文说明:
    - 调用方 / Called by: carry diagnostic aggregation helpers.
    - 调用对象 / Calls: `isinstance`.
    - 作用 / Purpose: 在聚合前验证 checkpoint/JSON 行的嵌套对象类型。
    - 变量 / Variables: `value` 是待检查对象, `field_name` 是错误上下文。
    - 接入 / Integration: 读取 checkpoint 恢复结果时复用本函数。
    - 错误处理 / Error handling: 非 mapping 抛出 `TypeError`。
    - 关键词 / Keywords:
      mapping|validate|checkpoint|json|carry_diagnostic|aggregate|mhdsra2|type|映射|校验
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    return value


def _required_float(row: Mapping[str, object], field_name: str) -> float:
    """Read one numeric field as float from a diagnostic row.

    中文说明:
    - 调用方 / Called by: carry diagnostic aggregation helpers.
    - 调用对象 / Calls: `float`.
    - 作用 / Purpose: 将 checkpoint/JSON 中的数值字段统一转换为 float。
    - 变量 / Variables: `row` 是源 mapping, `field_name` 是字段名。
    - 接入 / Integration: 聚合 learning rate、EM、loss 等字段时调用。
    - 错误处理 / Error handling: 缺字段或无法转 float 会直接抛出。
    - 关键词 / Keywords:
      float|json|checkpoint|metric|aggregate|carry_diagnostic|mhdsra2|field|数值|字段
    """
    return float(row[field_name])


def _required_int(row: Mapping[str, object], field_name: str) -> int:
    """Read one numeric field as int from a diagnostic row.

    中文说明:
    - 调用方 / Called by: carry diagnostic aggregation helpers.
    - 调用对象 / Calls: `int`.
    - 作用 / Purpose: 将 checkpoint/JSON 中的层数、seed、步数等字段转为 int。
    - 变量 / Variables: `row` 是源 mapping, `field_name` 是字段名。
    - 接入 / Integration: 聚合网格分组键和计数字段时调用。
    - 错误处理 / Error handling: 缺字段或无法转 int 会直接抛出。
    - 关键词 / Keywords:
      int|json|checkpoint|layer|seed|aggregate|carry_diagnostic|mhdsra2|整数|字段
    """
    return int(row[field_name])


def _run_mapping_from_diagnostic_row(
    row: Mapping[str, object],
) -> Mapping[str, object]:
    """Return the nested arithmetic run mapping from a diagnostic row.

    中文说明:
    - 调用方 / Called by: carry diagnostic metric helpers.
    - 调用对象 / Calls: `_required_mapping`.
    - 作用 / Purpose: 集中读取 `run` 子对象，避免聚合代码重复类型检查。
    - 变量 / Variables: `row` 是序列化后的 carry diagnostic run。
    - 接入 / Integration: checkpoint 恢复和实时运行结果共用同一读取路径。
    - 错误处理 / Error handling: 缺失或类型错误抛出异常。
    - 关键词 / Keywords:
      run_mapping|checkpoint|json|carry_diagnostic|aggregate|mhdsra2|nested|metrics|运行|嵌套
    """
    return _required_mapping(row["run"], "run")


def carry_exact_match_from_diagnostic_row(row: Mapping[str, object]) -> float:
    """Read carry exact-match from one diagnostic row.

    中文说明:
    - 调用方 / Called by: `aggregate_carry_diagnostic_run_rows`, tests.
    - 调用对象 / Calls: `_run_mapping_from_diagnostic_row`, `_required_float`.
    - 作用 / Purpose: curriculum 用 `unit_with_carry` 阶段 EM，carry-only 用训练 EM。
    - 变量 / Variables: `row` 是序列化诊断行, `stage_rows` 是阶段指标列表。
    - 接入 / Integration: 报告中的 `carry_exact_match_mean` 来自本函数。
    - 错误处理 / Error handling: 缺失阶段指标时返回 `0.0`，表示未学会 carry。
    - 关键词 / Keywords:
      carry_exact_match|unit_with_carry|metric|diagnostic|aggregate|mhdsra2|em|进位|指标|读取
    """
    run_mapping = _run_mapping_from_diagnostic_row(row)
    if str(row["dataset_name"]) == UNIT_WITH_CARRY_ONLY:
        return _required_float(run_mapping, "train_exact_match")
    stage_rows = run_mapping.get("stage_exact_matches", ())
    if not isinstance(stage_rows, Sequence):
        raise TypeError("stage_exact_matches must be a sequence.")
    for stage_row in stage_rows:
        stage_mapping = _required_mapping(stage_row, "stage_exact_matches[]")
        if str(stage_mapping["stage_name"]) == UNIT_WITH_CARRY_STAGE:
            return _required_float(stage_mapping, "exact_match")
    return 0.0


def aggregate_carry_diagnostic_run_rows(
    run_rows: Sequence[Mapping[str, object]],
    *,
    target_stage_count: int,
) -> list[ArithmeticCarryDiagnosticAggregate]:
    """Aggregate carry diagnostic rows by dataset and grid cell.

    中文说明:
    - 调用方 / Called by: carry diagnostic payload builder and tests.
    - 调用对象 / Calls: `_mean_and_variance`, `carry_exact_match_from_diagnostic_row`.
    - 作用 / Purpose: 汇总每个完整网格 cell 的 carry EM、保留阶段和训练指标。
    - 变量 / Variables: `groups` 是 dataset/strategy/lr/eval/steps/layers 分组键。
    - 接入 / Integration: Markdown 摘要只展示本函数输出的聚合行。
    - 错误处理 / Error handling: 空输入返回空列表，非法行结构直接抛出。
    - 关键词 / Keywords:
      aggregate|carry_diagnostic|learning_rate|strategy|eval_interval|retention|mhdsra2|grid|聚合|诊断
    """
    groups = sorted(
        {
            (
                str(row["dataset_name"]),
                str(row["training_strategy"]),
                _required_float(row, "learning_rate"),
                _required_int(row, "curriculum_eval_interval"),
                _required_int(row, "max_steps_per_stage"),
                _required_int(row, "num_layers"),
            )
            for row in run_rows
        }
    )
    aggregates: list[ArithmeticCarryDiagnosticAggregate] = []
    for (
        dataset_name,
        training_strategy,
        learning_rate,
        curriculum_eval_interval,
        max_steps_per_stage,
        num_layers,
    ) in groups:
        group_rows = [
            row
            for row in run_rows
            if str(row["dataset_name"]) == dataset_name
            and str(row["training_strategy"]) == training_strategy
            and _required_float(row, "learning_rate") == learning_rate
            and _required_int(row, "curriculum_eval_interval") == curriculum_eval_interval
            and _required_int(row, "max_steps_per_stage") == max_steps_per_stage
            and _required_int(row, "num_layers") == num_layers
        ]
        if not group_rows:
            continue
        run_mappings = [_run_mapping_from_diagnostic_row(row) for row in group_rows]
        retained_values = [
            float(_required_int(run_mapping, "retained_stage_count"))
            for run_mapping in run_mappings
        ]
        train_values = [
            _required_float(run_mapping, "train_exact_match")
            for run_mapping in run_mappings
        ]
        loss_values = [
            _required_float(run_mapping, "final_loss")
            for run_mapping in run_mappings
        ]
        carry_values = [
            carry_exact_match_from_diagnostic_row(row) for row in group_rows
        ]
        target_hits = sum(
            1 for retained_count in retained_values if retained_count >= target_stage_count
        )
        retained_mean, retained_variance = _mean_and_variance(retained_values)
        train_mean, train_variance = _mean_and_variance(train_values)
        loss_mean, loss_variance = _mean_and_variance(loss_values)
        carry_mean, carry_variance = _mean_and_variance(carry_values)
        aggregates.append(
            ArithmeticCarryDiagnosticAggregate(
                dataset_name=dataset_name,
                training_strategy=training_strategy,
                learning_rate=learning_rate,
                curriculum_eval_interval=curriculum_eval_interval,
                max_steps_per_stage=max_steps_per_stage,
                num_layers=num_layers,
                num_runs=len(group_rows),
                target_stage_count=target_stage_count,
                target_retention_rate=target_hits / len(group_rows),
                stable_target_retention=target_hits == len(group_rows),
                carry_exact_match_mean=carry_mean,
                carry_exact_match_variance=carry_variance,
                train_exact_match_mean=train_mean,
                train_exact_match_variance=train_variance,
                retained_stage_count_mean=retained_mean,
                retained_stage_count_variance=retained_variance,
                final_loss_mean=loss_mean,
                final_loss_variance=loss_variance,
            )
        )
    return aggregates


def build_carry_diagnostic_grid_payload(
    *,
    run_rows: Sequence[Mapping[str, object]] | None = None,
    layer_counts: Sequence[int] = DEFAULT_CARRY_DIAGNOSTIC_LAYER_COUNTS,
    max_steps_per_stage_values: Sequence[int] = DEFAULT_CARRY_DIAGNOSTIC_STEP_BUDGETS,
    curriculum_eval_intervals: Sequence[int] = DEFAULT_CARRY_DIAGNOSTIC_EVAL_INTERVALS,
    learning_rates: Sequence[float] = DEFAULT_CARRY_DIAGNOSTIC_LEARNING_RATES,
    training_strategies: Sequence[str] = TRAINING_STRATEGIES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    replay_ratio: float = 0.75,
    stage_patience: int = 3,
    carry_replay_ratio: float = DEFAULT_CARRY_REPLAY_RATIO,
    stage_loss_weights: Mapping[str, float] | None = None,
    target_stage_count: int = 2,
    checkpoint_path: str | None = None,
    resume_supported: bool = True,
) -> dict[str, object]:
    """Build the JSON payload for the carry diagnostic grid.

    中文说明:
    - 调用方 / Called by: carry diagnostic CLI and tests.
    - 调用对象 / Calls: `run_one_carry_diagnostic_grid_point`,
      `aggregate_carry_diagnostic_run_rows`.
    - 作用 / Purpose: 构建完整诊断报告 payload，可直接运行或聚合 checkpoint 行。
    - 变量 / Variables: `run_rows` 是可选预计算行, `aggregates` 是按 cell 聚合结果。
    - 接入 / Integration: CLI 使用 checkpoint rows 调用本函数生成最终 JSON/Markdown。
    - 错误处理 / Error handling: 空网格、非法策略、非法权重直接抛出。
    - 关键词 / Keywords:
      payload|carry_diagnostic|grid|learning_rate|eval_interval|checkpoint|mhdsra2|report|报告|诊断
    """
    if (
        not layer_counts
        or not max_steps_per_stage_values
        or not curriculum_eval_intervals
        or not learning_rates
        or not training_strategies
        or not seeds
    ):
        raise ValueError("carry diagnostic grid dimensions must not be empty.")
    normalized_strategies = tuple(
        validate_training_strategy(strategy) for strategy in training_strategies
    )
    validated_stage_loss_weights = validate_stage_loss_weights(stage_loss_weights)
    if run_rows is None:
        dataset_specs = build_carry_diagnostic_dataset_specs()
        computed_rows: list[dict[str, object]] = []
        for dataset_spec in dataset_specs:
            dataset_spec.validate_training_scope()
            for training_strategy in normalized_strategies:
                for learning_rate in learning_rates:
                    for curriculum_eval_interval in curriculum_eval_intervals:
                        for max_steps_per_stage in max_steps_per_stage_values:
                            for num_layers in layer_counts:
                                for seed in seeds:
                                    diagnostic_run = run_one_carry_diagnostic_grid_point(
                                        dataset_spec=dataset_spec,
                                        training_strategy=training_strategy,
                                        learning_rate=learning_rate,
                                        curriculum_eval_interval=curriculum_eval_interval,
                                        max_steps_per_stage=max_steps_per_stage,
                                        num_layers=num_layers,
                                        seed=seed,
                                        replay_ratio=replay_ratio,
                                        stage_patience=stage_patience,
                                        carry_replay_ratio=carry_replay_ratio,
                                        stage_loss_weights=validated_stage_loss_weights,
                                    )
                                    computed_rows.append(
                                        serialize_carry_diagnostic_run(diagnostic_run)
                                    )
        resolved_run_rows: list[Mapping[str, object]] = computed_rows
    else:
        resolved_run_rows = list(run_rows)
    aggregates = aggregate_carry_diagnostic_run_rows(
        resolved_run_rows,
        target_stage_count=target_stage_count,
    )
    stable_aggregates = [
        aggregate for aggregate in aggregates if aggregate.stable_target_retention
    ]
    carry_success_aggregates = [
        aggregate
        for aggregate in aggregates
        if aggregate.dataset_name == UNIT_WITH_CARRY_ONLY
        and aggregate.carry_exact_match_mean >= CURRICULUM_STAGE_EXACT_MATCH_THRESHOLD
    ]
    return {
        "config": {
            "layer_counts": list(layer_counts),
            "max_steps_per_stage_values": list(max_steps_per_stage_values),
            "curriculum_eval_intervals": list(curriculum_eval_intervals),
            "learning_rates": list(learning_rates),
            "training_strategies": list(normalized_strategies),
            "seeds": list(seeds),
            "replay_ratio": replay_ratio,
            "stage_patience": stage_patience,
            "carry_replay_ratio": carry_replay_ratio,
            "stage_loss_weights": dict(validated_stage_loss_weights),
            "target_stage_count": target_stage_count,
            "checkpoint_path": checkpoint_path,
            "resume_supported": resume_supported,
        },
        "datasets": [
            {
                "name": spec.name,
                "training_examples": [example.equation for example in spec.training_examples],
                "curriculum_stages": [
                    {
                        "name": stage.name,
                        "examples": [example.equation for example in stage.examples],
                    }
                    for stage in spec.curriculum_stages
                ],
            }
            for spec in build_carry_diagnostic_dataset_specs()
        ],
        "summary": {
            "stable_target_strategy_count": len(stable_aggregates),
            "has_stable_target_strategy": bool(stable_aggregates),
            "carry_only_success_count": len(carry_success_aggregates),
            "unit_with_carry_success_threshold": CURRICULUM_STAGE_EXACT_MATCH_THRESHOLD,
        },
        "aggregates": [asdict(aggregate) for aggregate in aggregates],
        "runs": [dict(row) for row in resolved_run_rows],
    }


def build_carry_diagnostic_grid_markdown(payload: dict[str, object]) -> list[str]:
    """Build Markdown lines for the carry diagnostic grid report.

    中文说明:
    - 调用方 / Called by: carry diagnostic CLI.
    - 调用对象 / Calls: none; renders payload fields into Markdown.
    - 作用 / Purpose: 用按 cell 聚合表展示 carry-only 与 curriculum 诊断结果。
    - 变量 / Variables: `payload` 是报告数据, `aggregates` 是聚合行。
    - 接入 / Integration: 报告文件 `mhdsra2_carry_diagnostic_grid.md` 使用本函数。
    - 错误处理 / Error handling: 字段缺失或类型错误直接抛出。
    - 关键词 / Keywords:
      markdown|carry_diagnostic|report|learning_rate|strategy|eval_interval|mhdsra2|grid|报告|进位
    """
    config = payload["config"]
    summary = payload["summary"]
    aggregates = payload["aggregates"]
    if not isinstance(config, dict) or not isinstance(summary, dict):
        raise TypeError("carry diagnostic config and summary must be dictionaries.")
    if not isinstance(aggregates, list):
        raise TypeError("carry diagnostic aggregates must be a list.")
    lines = [
        "# MHDSRA2 Carry Diagnostic Grid",
        "",
        "## Configuration",
        "",
        f"- Layers: {', '.join(str(item) for item in config['layer_counts'])}",
        (
            "- Max steps per stage values: "
            f"{', '.join(str(item) for item in config['max_steps_per_stage_values'])}"
        ),
        (
            "- Curriculum eval intervals: "
            f"{', '.join(str(item) for item in config['curriculum_eval_intervals'])}"
        ),
        f"- Learning rates: {', '.join(str(item) for item in config['learning_rates'])}",
        (
            "- Training strategies: "
            f"{', '.join(str(item) for item in config['training_strategies'])}"
        ),
        f"- Seeds: {', '.join(str(item) for item in config['seeds'])}",
        f"- Replay ratio: {config['replay_ratio']}",
        f"- Stage patience: {config['stage_patience']}",
        f"- Carry replay ratio: {config['carry_replay_ratio']}",
        f"- Stage loss weights: {config['stage_loss_weights']}",
        f"- Checkpoint path: {config['checkpoint_path']}",
        f"- Resume supported: {config['resume_supported']}",
        "",
        "## Summary",
        "",
        f"- Stable target strategy count: {summary['stable_target_strategy_count']}",
        f"- Has stable target strategy: {summary['has_stable_target_strategy']}",
        f"- Carry-only success count: {summary['carry_only_success_count']}",
        "",
        "## Aggregates",
        "",
        "| Dataset | Strategy | LR | Eval Interval | Max Steps | Layers | Runs | "
        "Carry EM Mean | Target Retention Rate | Retained Mean | Train EM Mean | Final Loss Mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        if not isinstance(row, dict):
            raise TypeError("carry diagnostic aggregate rows must be dictionaries.")
        lines.append(
            f"| {row['dataset_name']} | {row['training_strategy']} | "
            f"{row['learning_rate']:.4f} | {row['curriculum_eval_interval']} | "
            f"{row['max_steps_per_stage']} | {row['num_layers']} | {row['num_runs']} | "
            f"{row['carry_exact_match_mean']:.4f} | "
            f"{row['target_retention_rate']:.4f} | "
            f"{row['retained_stage_count_mean']:.4f} | "
            f"{row['train_exact_match_mean']:.4f} | {row['final_loss_mean']:.4f} |"
        )
    return lines


def build_two_digit_diagnostic_dataset_specs() -> tuple[ArithmeticRuleDatasetSpec, ...]:
    """Build datasets used by the two-digit diagnostic grid.

    中文说明:
    - 调用方 / Called by: two-digit diagnostic payload builders, CLI and tests.
    - 调用对象 / Calls: `build_curriculum_arithmetic_spec`, `build_two_digit_only_spec`,
      `build_prereq_plus_two_digit_spec`.
    - 作用 / Purpose: 同时提供真实 curriculum、two-digit-only 和混合非课程三类诊断数据。
    - 变量 / Variables: 返回 tuple 顺序固定, 先真实 curriculum, 再隔离与混合诊断。
    - 接入 / Integration: 新增 two-digit 诊断数据集时在本函数集中扩展。
    - 错误处理 / Error handling: 下游会逐个执行 `validate_training_scope`。
    - 关键词 / Keywords:
      two_digit|datasets|curriculum|only|prereq|mhdsra2|arithmetic|diagnostic|数据集|两位数
    """
    return (
        build_curriculum_arithmetic_spec(),
        build_two_digit_only_spec(),
        build_prereq_plus_two_digit_spec(),
    )


def select_two_digit_diagnostic_dataset_specs(
    dataset_names: Sequence[str] | None = None,
) -> tuple[ArithmeticRuleDatasetSpec, ...]:
    """Select two-digit diagnostic dataset specs by name.

    中文说明:
    - 调用方 / Called by: `build_two_digit_diagnostic_grid_payload`, two-digit diagnostic CLI,
      and tests.
    - 调用对象 / Calls: `build_two_digit_diagnostic_dataset_specs`.
    - 作用 / Purpose: 将可选数据集名过滤应用到 two_digit_rules 诊断数据集集合。
    - 参数 / Parameters: `dataset_names` 是可空的数据集名序列; `None` 表示使用全部默认数据集。
    - 返回 / Returns: 按请求顺序返回的 `ArithmeticRuleDatasetSpec` 元组。
    - 变量 / Variables: `spec_by_name` 是数据集名到强类型规约的映射; `normalized_names` 是去重后的请求名。
    - 接入 / Integration: 应用层负责数据集白名单校验, CLI 仅解析和传递用户输入。
    - 错误处理 / Error handling: 空列表或未知数据集名会抛出 `ValueError`, 避免静默跑错网格。
    - 副作用 / Side effects: 无文件、数据库、网络或全局状态写入。
    - 事务边界 / Transaction boundary: 不涉及 Unit of Work 或事务。
    - 并发与幂等 / Concurrency and idempotency: 纯选择逻辑, 可重复调用。
    - 中文关键词: 两位数, 诊断, 数据集, 过滤, 规约, 网格, 课程, 隔离, 混合, 校验

    English documentation:
    Function name:
        select_two_digit_diagnostic_dataset_specs
    Purpose:
        Select typed two-digit diagnostic dataset specifications from optional names.
    Called by:
        `build_two_digit_diagnostic_grid_payload`, the two-digit diagnostic CLI, and tests.
    Calls:
        `build_two_digit_diagnostic_dataset_specs`; no repositories, Unit of Work, or external services.
    Parameters:
        - dataset_names: optional sequence of dataset names; `None` means all default datasets.
    Returns:
        A tuple of `ArithmeticRuleDatasetSpec` instances in requested order.
    Internal variables:
        - spec_by_name: maps dataset names to typed specs for validation.
        - normalized_names: deduplicated requested names.
    Integration:
        Keep dataset validation in the application layer; interface scripts should only pass parsed names.
    Error handling:
        Raises `ValueError` for empty selections or unknown names.
    Side effects:
        None.
    Transaction boundary:
        No transaction or Unit of Work is used.
    Concurrency and idempotency:
        Deterministic and repeatable for the same input.
    English keywords:
        two_digit, diagnostic, dataset, filter, spec, grid, curriculum, isolated, mixed, validation
    """
    all_specs = build_two_digit_diagnostic_dataset_specs()
    if dataset_names is None:
        return all_specs
    normalized_names = tuple(dict.fromkeys(name.strip() for name in dataset_names if name.strip()))
    if not normalized_names:
        raise ValueError("two-digit diagnostic datasets must not be empty.")
    spec_by_name = {spec.name: spec for spec in all_specs}
    unknown_names = tuple(name for name in normalized_names if name not in spec_by_name)
    if unknown_names:
        allowed_names = ", ".join(TWO_DIGIT_DIAGNOSTIC_DATASETS)
        unknown_text = ", ".join(unknown_names)
        raise ValueError(
            f"unknown two-digit diagnostic dataset(s): {unknown_text}; "
            f"allowed values are: {allowed_names}."
        )
    return tuple(spec_by_name[name] for name in normalized_names)


def run_one_two_digit_diagnostic_grid_point(
    *,
    dataset_spec: ArithmeticRuleDatasetSpec,
    training_strategy: str,
    learning_rate: float,
    max_steps_per_stage: int,
    num_layers: int,
    seed: int,
    replay_ratio: float,
    stage_patience: int,
    two_digit_replay_ratio: float,
    stage_loss_weights: Mapping[str, float] | None,
    device: str | torch.device,
) -> ArithmeticTwoDigitDiagnosticRun:
    """Run one point in the two-digit diagnostic grid.

    中文说明:
    - 调用方 / Called by: two-digit diagnostic payload builder and CLI checkpoint loop.
    - 调用对象 / Calls: `run_one_arithmetic_emergence_curve`, `validate_training_strategy`.
    - 作用 / Purpose: 用强类型封装一次 two-digit 诊断训练点。
    - 变量 / Variables: `dataset_spec` 是诊断数据集, `two_digit_replay_ratio` 控制强化采样比例。
    - 接入 / Integration: CLI 每完成一个点就序列化到 checkpoint。
    - 错误处理 / Error handling: 训练参数非法或模型失败时直接抛出异常。
    - 关键词 / Keywords:
      two_digit|run_point|learning_rate|strategy|checkpoint|mhdsra2|grid|diagnostic|运行|两位数
    """
    normalized_strategy = validate_training_strategy(training_strategy)
    run = run_one_arithmetic_emergence_curve(
        dataset_spec=dataset_spec,
        model_name=MHDSRA2_MODEL,
        seed=seed,
        num_layers=num_layers,
        max_steps_per_stage=max_steps_per_stage,
        curriculum_eval_interval=DEFAULT_CURRICULUM_EVAL_INTERVAL,
        stage_threshold=CURRICULUM_STAGE_EXACT_MATCH_THRESHOLD,
        replay_ratio=replay_ratio,
        stage_patience=stage_patience,
        learning_rate=learning_rate,
        training_strategy=normalized_strategy,
        carry_replay_ratio=two_digit_replay_ratio,
        stage_loss_weights=stage_loss_weights,
        device=device,
    )
    return ArithmeticTwoDigitDiagnosticRun(
        dataset_name=dataset_spec.name,
        training_strategy=normalized_strategy,
        learning_rate=learning_rate,
        max_steps_per_stage=max_steps_per_stage,
        num_layers=num_layers,
        seed=seed,
        replay_ratio=replay_ratio,
        stage_patience=stage_patience,
        two_digit_replay_ratio=two_digit_replay_ratio,
        run=run,
    )


def serialize_two_digit_diagnostic_run(
    diagnostic_run: ArithmeticTwoDigitDiagnosticRun,
) -> dict[str, object]:
    """Serialize one two-digit diagnostic run for JSON reports and checkpoints.

    中文说明:
    - 调用方 / Called by: two-digit diagnostic payload builder, CLI and tests.
    - 调用对象 / Calls: `dataclasses.asdict`.
    - 作用 / Purpose: 保证 checkpoint JSONL 与最终报告使用同一字段结构。
    - 变量 / Variables: `diagnostic_run` 是强类型单点结果。
    - 接入 / Integration: 每个 grid cell 完成后调用本函数再写入 JSONL。
    - 错误处理 / Error handling: dataclass 序列化错误直接抛出。
    - 关键词 / Keywords:
      serialize|checkpoint|jsonl|two_digit|diagnostic|run|mhdsra2|report|序列化|两位数
    """
    return asdict(diagnostic_run)


def two_digit_exact_match_from_diagnostic_row(row: Mapping[str, object]) -> float:
    """Read two-digit exact-match from one diagnostic row.

    中文说明:
    - 调用方 / Called by: `aggregate_two_digit_diagnostic_run_rows`, tests.
    - 调用对象 / Calls: `_run_mapping_from_diagnostic_row`, `_required_float`.
    - 作用 / Purpose: 优先读取 `two_digit_rules` 阶段 EM, 缺失时回退为 0。
    - 变量 / Variables: `row` 是序列化诊断行, `stage_rows` 是阶段指标列表。
    - 接入 / Integration: 报告中的 `two_digit_exact_match_mean` 来自本函数。
    - 错误处理 / Error handling: 缺失阶段指标返回 `0.0`, 表示未学会 two-digit。
    - 关键词 / Keywords:
      two_digit_exact_match|metric|diagnostic|aggregate|mhdsra2|em|rules|读取|指标|两位数
    """
    run_mapping = _run_mapping_from_diagnostic_row(row)
    stage_rows = run_mapping.get("stage_exact_matches", ())
    if not isinstance(stage_rows, Sequence):
        raise TypeError("stage_exact_matches must be a sequence.")
    for stage_row in stage_rows:
        stage_mapping = _required_mapping(stage_row, "stage_exact_matches[]")
        if str(stage_mapping["stage_name"]) == TWO_DIGIT_RULES_STAGE:
            return _required_float(stage_mapping, "exact_match")
    if str(row["dataset_name"]) == TWO_DIGIT_ONLY:
        return _required_float(run_mapping, "train_exact_match")
    return 0.0


def aggregate_two_digit_diagnostic_run_rows(
    run_rows: Sequence[Mapping[str, object]],
    *,
    target_stage_count: int,
) -> list[ArithmeticTwoDigitDiagnosticAggregate]:
    """Aggregate two-digit diagnostic rows by dataset and grid cell.

    中文说明:
    - 调用方 / Called by: two-digit diagnostic payload builder and tests.
    - 调用对象 / Calls: `_mean_and_variance`, `two_digit_exact_match_from_diagnostic_row`.
    - 作用 / Purpose: 汇总每个 cell 的 two-digit EM、保留阶段数和训练指标。
    - 变量 / Variables: `groups` 是 dataset/strategy/lr/steps/layers 分组键。
    - 接入 / Integration: Markdown 摘要只展示本函数输出的聚合行。
    - 错误处理 / Error handling: 空输入返回空列表, 非法行结构直接抛出。
    - 关键词 / Keywords:
      aggregate|two_digit|diagnostic|learning_rate|strategy|retention|mhdsra2|grid|聚合|两位数
    """
    groups = sorted(
        {
            (
                str(row["dataset_name"]),
                str(row["training_strategy"]),
                _required_float(row, "learning_rate"),
                _required_int(row, "max_steps_per_stage"),
                _required_int(row, "num_layers"),
            )
            for row in run_rows
        }
    )
    aggregates: list[ArithmeticTwoDigitDiagnosticAggregate] = []
    for dataset_name, training_strategy, learning_rate, max_steps_per_stage, num_layers in groups:
        group_rows = [
            row
            for row in run_rows
            if str(row["dataset_name"]) == dataset_name
            and str(row["training_strategy"]) == training_strategy
            and _required_float(row, "learning_rate") == learning_rate
            and _required_int(row, "max_steps_per_stage") == max_steps_per_stage
            and _required_int(row, "num_layers") == num_layers
        ]
        if not group_rows:
            continue
        run_mappings = [_run_mapping_from_diagnostic_row(row) for row in group_rows]
        retained_values = [
            float(_required_int(run_mapping, "retained_stage_count"))
            for run_mapping in run_mappings
        ]
        train_values = [
            _required_float(run_mapping, "train_exact_match")
            for run_mapping in run_mappings
        ]
        loss_values = [
            _required_float(run_mapping, "final_loss")
            for run_mapping in run_mappings
        ]
        two_digit_values = [
            two_digit_exact_match_from_diagnostic_row(row) for row in group_rows
        ]
        target_hits = sum(
            1 for retained_count in retained_values if retained_count >= target_stage_count
        )
        retained_mean, retained_variance = _mean_and_variance(retained_values)
        train_mean, train_variance = _mean_and_variance(train_values)
        loss_mean, loss_variance = _mean_and_variance(loss_values)
        two_digit_mean, two_digit_variance = _mean_and_variance(two_digit_values)
        aggregates.append(
            ArithmeticTwoDigitDiagnosticAggregate(
                dataset_name=dataset_name,
                training_strategy=training_strategy,
                learning_rate=learning_rate,
                max_steps_per_stage=max_steps_per_stage,
                num_layers=num_layers,
                num_runs=len(group_rows),
                target_stage_count=target_stage_count,
                target_retention_rate=target_hits / len(group_rows),
                stable_target_retention=target_hits == len(group_rows),
                two_digit_exact_match_mean=two_digit_mean,
                two_digit_exact_match_variance=two_digit_variance,
                train_exact_match_mean=train_mean,
                train_exact_match_variance=train_variance,
                retained_stage_count_mean=retained_mean,
                retained_stage_count_variance=retained_variance,
                final_loss_mean=loss_mean,
                final_loss_variance=loss_variance,
            )
        )
    return aggregates


def build_two_digit_diagnostic_grid_payload(
    *,
    run_rows: Sequence[Mapping[str, object]] | None = None,
    datasets: Sequence[str] | None = None,
    layer_counts: Sequence[int] = DEFAULT_TWO_DIGIT_DIAGNOSTIC_LAYER_COUNTS,
    max_steps_per_stage_values: Sequence[int] = DEFAULT_TWO_DIGIT_DIAGNOSTIC_STEP_BUDGETS,
    learning_rates: Sequence[float] = DEFAULT_TWO_DIGIT_DIAGNOSTIC_LEARNING_RATES,
    training_strategies: Sequence[str] = TWO_DIGIT_TRAINING_STRATEGIES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    replay_ratio: float = 0.75,
    stage_patience: int = 3,
    two_digit_replay_ratio: float = DEFAULT_TWO_DIGIT_REPLAY_RATIO,
    stage_loss_weights: Mapping[str, float] | None = None,
    target_stage_count: int = 3,
    checkpoint_path: str | None = None,
    resume_supported: bool = True,
    device: str | torch.device = DEFAULT_ARITHMETIC_EMERGENCE_DEVICE,
) -> dict[str, object]:
    """Build the JSON payload for the two-digit diagnostic grid.

    中文说明:
    - 调用方 / Called by: two-digit diagnostic CLI and tests.
    - 调用对象 / Calls: `select_two_digit_diagnostic_dataset_specs`,
      `run_one_two_digit_diagnostic_grid_point`,
      `aggregate_two_digit_diagnostic_run_rows`.
    - 作用 / Purpose: 构建 two_digit_rules 专项诊断报告 payload, 支持按数据集过滤中等网格。
    - 变量 / Variables: `run_rows` 是可选 checkpoint 行, `dataset_specs` 是过滤后的数据集规约。
    - 接入 / Integration: CLI 使用 checkpoint rows 调用本函数生成最终 JSON/Markdown。
    - 错误处理 / Error handling: 空网格、非法数据集、非法策略、非法权重、非法设备直接抛出。
    - 关键词 / Keywords:
      payload|two_digit|diagnostic|datasets|grid|learning_rate|checkpoint|mhdsra2|报告|两位数
    """
    if (
        not layer_counts
        or not max_steps_per_stage_values
        or not learning_rates
        or not training_strategies
        or not seeds
    ):
        raise ValueError("two-digit diagnostic grid dimensions must not be empty.")
    normalized_strategies = tuple(
        validate_training_strategy(strategy) for strategy in training_strategies
    )
    validated_stage_loss_weights = validate_stage_loss_weights(
        stage_loss_weights or DEFAULT_TWO_DIGIT_STAGE_LOSS_WEIGHTS
    )
    resolved_device = resolve_torch_device(device)
    dataset_specs = select_two_digit_diagnostic_dataset_specs(datasets)
    selected_dataset_names = tuple(spec.name for spec in dataset_specs)
    if run_rows is None:
        computed_rows: list[dict[str, object]] = []
        for dataset_spec in dataset_specs:
            dataset_spec.validate_training_scope()
            for training_strategy in normalized_strategies:
                for learning_rate in learning_rates:
                    for max_steps_per_stage in max_steps_per_stage_values:
                        for num_layers in layer_counts:
                            for seed in seeds:
                                diagnostic_run = run_one_two_digit_diagnostic_grid_point(
                                    dataset_spec=dataset_spec,
                                    training_strategy=training_strategy,
                                    learning_rate=learning_rate,
                                    max_steps_per_stage=max_steps_per_stage,
                                    num_layers=num_layers,
                                    seed=seed,
                                    replay_ratio=replay_ratio,
                                    stage_patience=stage_patience,
                                    two_digit_replay_ratio=two_digit_replay_ratio,
                                    stage_loss_weights=validated_stage_loss_weights,
                                    device=resolved_device,
                                )
                                computed_rows.append(
                                    serialize_two_digit_diagnostic_run(diagnostic_run)
                                )
        resolved_run_rows: list[Mapping[str, object]] = computed_rows
    else:
        resolved_run_rows = [
            row for row in run_rows if str(row["dataset_name"]) in selected_dataset_names
        ]
    aggregates = aggregate_two_digit_diagnostic_run_rows(
        resolved_run_rows,
        target_stage_count=target_stage_count,
    )
    two_digit_success_aggregates = [
        aggregate
        for aggregate in aggregates
        if aggregate.dataset_name == TWO_DIGIT_ONLY
        and aggregate.two_digit_exact_match_mean >= CURRICULUM_STAGE_EXACT_MATCH_THRESHOLD
    ]
    stable_aggregates = [
        aggregate for aggregate in aggregates if aggregate.stable_target_retention
    ]
    return {
        "config": {
            "layer_counts": list(layer_counts),
            "max_steps_per_stage_values": list(max_steps_per_stage_values),
            "learning_rates": list(learning_rates),
            "training_strategies": list(normalized_strategies),
            "datasets": list(selected_dataset_names),
            "seeds": list(seeds),
            "replay_ratio": replay_ratio,
            "stage_patience": stage_patience,
            "two_digit_replay_ratio": two_digit_replay_ratio,
            "stage_loss_weights": dict(validated_stage_loss_weights),
            "target_stage_count": target_stage_count,
            "checkpoint_path": checkpoint_path,
            "resume_supported": resume_supported,
            "device": str(resolved_device),
        },
        "datasets": [
            {
                "name": spec.name,
                "training_examples": [example.equation for example in spec.training_examples],
                "curriculum_stages": [
                    {
                        "name": stage.name,
                        "examples": [example.equation for example in stage.examples],
                    }
                    for stage in spec.curriculum_stages
                ],
                "diagnostic_stages": [
                    {
                        "name": stage.name,
                        "examples": [example.equation for example in stage.examples],
                    }
                    for stage in spec.diagnostic_stages
                ],
            }
            for spec in dataset_specs
        ],
        "summary": {
            "stable_target_strategy_count": len(stable_aggregates),
            "has_stable_target_strategy": bool(stable_aggregates),
            "two_digit_only_success_count": len(two_digit_success_aggregates),
            "two_digit_success_threshold": CURRICULUM_STAGE_EXACT_MATCH_THRESHOLD,
        },
        "aggregates": [asdict(aggregate) for aggregate in aggregates],
        "runs": [dict(row) for row in resolved_run_rows],
    }


def build_two_digit_diagnostic_grid_markdown(payload: dict[str, object]) -> list[str]:
    """Build Markdown lines for the two-digit diagnostic grid report.

    中文说明:
    - 调用方 / Called by: two-digit diagnostic CLI.
    - 调用对象 / Calls: none; renders payload fields into Markdown.
    - 作用 / Purpose: 用按 cell 聚合表展示 two-digit-only、mixed 与 curriculum 诊断结果。
    - 变量 / Variables: `payload` 是报告数据, `aggregates` 是聚合行。
    - 接入 / Integration: 报告文件 `mhdsra2_two_digit_diagnostic_grid.md` 使用本函数。
    - 错误处理 / Error handling: 字段缺失或类型错误直接抛出。
    - 关键词 / Keywords:
      markdown|two_digit|diagnostic|report|learning_rate|strategy|mhdsra2|grid|报告|两位数
    """
    config = payload["config"]
    summary = payload["summary"]
    aggregates = payload["aggregates"]
    if not isinstance(config, dict) or not isinstance(summary, dict):
        raise TypeError("two-digit diagnostic config and summary must be dictionaries.")
    if not isinstance(aggregates, list):
        raise TypeError("two-digit diagnostic aggregates must be a list.")
    lines = [
        "# MHDSRA2 Two-Digit Diagnostic Grid",
        "",
        "## Configuration",
        "",
        f"- Device: {config['device']}",
        f"- Datasets: {', '.join(str(item) for item in config['datasets'])}",
        f"- Layers: {', '.join(str(item) for item in config['layer_counts'])}",
        (
            "- Max steps per stage values: "
            f"{', '.join(str(item) for item in config['max_steps_per_stage_values'])}"
        ),
        f"- Learning rates: {', '.join(str(item) for item in config['learning_rates'])}",
        (
            "- Training strategies: "
            f"{', '.join(str(item) for item in config['training_strategies'])}"
        ),
        f"- Seeds: {', '.join(str(item) for item in config['seeds'])}",
        f"- Replay ratio: {config['replay_ratio']}",
        f"- Stage patience: {config['stage_patience']}",
        f"- Two-digit replay ratio: {config['two_digit_replay_ratio']}",
        f"- Stage loss weights: {config['stage_loss_weights']}",
        f"- Checkpoint path: {config['checkpoint_path']}",
        f"- Resume supported: {config['resume_supported']}",
        "",
        "## Summary",
        "",
        f"- Stable target strategy count: {summary['stable_target_strategy_count']}",
        f"- Has stable target strategy: {summary['has_stable_target_strategy']}",
        f"- Two-digit-only success count: {summary['two_digit_only_success_count']}",
        "",
        "## Aggregates",
        "",
        "| Dataset | Strategy | LR | Max Steps | Layers | Runs | Two-Digit EM Mean | "
        "Target Retention Rate | Retained Mean | Train EM Mean | Final Loss Mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        if not isinstance(row, dict):
            raise TypeError("two-digit diagnostic aggregate rows must be dictionaries.")
        lines.append(
            f"| {row['dataset_name']} | {row['training_strategy']} | "
            f"{row['learning_rate']:.4f} | {row['max_steps_per_stage']} | "
            f"{row['num_layers']} | {row['num_runs']} | "
            f"{row['two_digit_exact_match_mean']:.4f} | "
            f"{row['target_retention_rate']:.4f} | "
            f"{row['retained_stage_count_mean']:.4f} | "
            f"{row['train_exact_match_mean']:.4f} | {row['final_loss_mean']:.4f} |"
        )
    return lines


def build_layer_emergence_markdown(payload: dict[str, object]) -> list[str]:
    """Build Markdown lines for the decimal arithmetic emergence report.

    中文说明:
    - 调用方 / Called by: CLI script.
    - 调用对象 / Calls: 无；读取 payload 字段组装 Markdown。
    - 作用 / Purpose: 渲染训练规约、负控、外推指标和最小涌现层数。
    - 变量 / Variables: `payload` 是报告数据, `aggregates` 是聚合结果。
    - 接入 / Integration: 报告格式变更集中修改本函数。
    - 错误处理 / Error handling: 缺字段会抛出 `KeyError` 或 `TypeError`。
    - 关键词 / Keywords:
      markdown|report|arithmetic|emergence|decimal|100+100|mhdsra2|application|渲染|报告
    """
    config = payload["config"]
    summary = payload["summary"]
    datasets = payload["datasets"]
    aggregates = payload["aggregates"]
    stage_aggregates = payload["curriculum_stage_aggregates"]
    runs = payload["runs"]
    if not isinstance(config, dict) or not isinstance(summary, dict):
        raise TypeError("payload config and summary must be dictionaries.")
    if (
        not isinstance(datasets, list)
        or not isinstance(aggregates, list)
        or not isinstance(stage_aggregates, list)
        or not isinstance(runs, list)
    ):
        raise TypeError("payload datasets, aggregates, stage aggregates and runs must be lists.")
    minimum_layers = summary["minimum_arithmetic_emergent_layers"]
    minimum_layers_text = "null" if minimum_layers is None else str(minimum_layers)
    minimum_mastery_layers = summary["minimum_curriculum_mastery_layers"]
    minimum_mastery_layers_text = (
        "null" if minimum_mastery_layers is None else str(minimum_mastery_layers)
    )
    lines = [
        "# MHDSRA2 Decimal Arithmetic Emergence",
        "",
        "## Proxy Definition",
        "",
        (
            "The headline probe asks whether a model trained only on low-value decimal "
            "addition curriculum stages can greedily generate `200<eos>` for `100+100=`."
        ),
        "",
        "## Configuration",
        "",
        f"- Layers: {', '.join(str(item) for item in config['layer_counts'])}",
        f"- Seeds: {', '.join(str(item) for item in config['seeds'])}",
        f"- Max steps per curriculum stage: {config['max_steps_per_stage']}",
        f"- Curriculum stage EM threshold: {config['curriculum_stage_exact_match_threshold']}",
        f"- Curriculum eval interval: {config['curriculum_eval_interval']}",
        f"- Replay ratio: {config['replay_ratio']}",
        f"- Stage patience: {config['stage_patience']}",
        f"- Learning rate: {config['learning_rate']}",
        f"- Device: {config['device']}",
        f"- Minimum curriculum mastery layers: {minimum_mastery_layers_text}",
        f"- Minimum arithmetic emergent layers: {minimum_layers_text}",
        "",
        "## Dataset Specs",
        "",
    ]
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise TypeError("dataset rows must be dictionaries.")
        lines.extend(
            [
                f"### {dataset['name']}",
                f"- Training: {', '.join(str(item) for item in dataset['training_examples'])}",
                f"- Headline: {dataset['headline_example']}",
                f"- OOD: {', '.join(str(item) for item in dataset['ood_examples'])}",
            ]
        )
        curriculum_stages = dataset.get("curriculum_stages", [])
        if isinstance(curriculum_stages, list) and curriculum_stages:
            lines.append("- Curriculum:")
            for stage in curriculum_stages:
                if not isinstance(stage, dict):
                    raise TypeError("curriculum stage rows must be dictionaries.")
                lines.append(
                    f"  - {stage['name']}: {', '.join(str(item) for item in stage['examples'])}"
                )
        lines.append("")
    lines.extend(
        [
            "## Multi-Seed Summary",
            "",
            "| Dataset | Model | Layers | Seeds | Train EM Mean | Headline EM Mean | OOD EM Mean | "
            "Final Loss Mean | Meets Criteria |",
            "|---|---|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in aggregates:
        if not isinstance(row, dict):
            raise TypeError("aggregate rows must be dictionaries.")
        meets = "yes" if row["meets_success_criteria"] else "no"
        lines.append(
            f"| {row['dataset_name']} | {row['model_name']} | {row['num_layers']} | "
            f"{row['num_seeds']} | {row['train_exact_match_mean']:.4f} | "
            f"{row['headline_exact_match_mean']:.4f} | {row['ood_exact_match_mean']:.4f} | "
            f"{row['final_loss_mean']:.4f} | {meets} |"
        )
    lines.extend(
        [
            "",
            "## Training Stop Summary",
            "",
            "| Dataset | Model | Layers | Seed | Steps Executed | Ever Passed | Retained | Stop Reason |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for run in runs:
        if not isinstance(run, dict):
            raise TypeError("run rows must be dictionaries.")
        lines.append(
            f"| {run['dataset_name']} | {run['model_name']} | {run['num_layers']} | "
            f"{run['seed']} | {run['training_steps_executed']} | "
            f"{run['ever_passed_stage_count']} | {run['retained_stage_count']} | "
            f"{run['stopped_reason']} |"
        )
    lines.extend(
        [
            "",
            "## Curriculum Stage Aggregate",
            "",
            "| Dataset | Model | Layers | Stage | Runs | Pass Rate | Mean Pass Step | Final EM Mean |",
            "|---|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in stage_aggregates:
        if not isinstance(row, dict):
            raise TypeError("curriculum stage aggregate rows must be dictionaries.")
        advance_step_mean = row["advance_step_mean"]
        advance_step_text = "-" if advance_step_mean is None else f"{advance_step_mean:.2f}"
        lines.append(
            f"| {row['dataset_name']} | {row['model_name']} | {row['num_layers']} | "
            f"{row['stage_name']} | {row['num_runs']} | {row['pass_rate']:.4f} | "
            f"{advance_step_text} | {row['final_exact_match_mean']:.4f} |"
        )
    return lines
