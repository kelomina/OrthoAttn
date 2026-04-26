import json
import random
from pathlib import Path
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.dsra.report_utils import ensure_reports_dir, write_json, write_markdown
from scripts.toy_task_associative_recall import (
    DSRAModel,
    MHDSRA2Model,
    LinearAttentionModel,
    SlidingWindowAttentionModel,
    SparseAttentionModel,
    StandardAttentionModel,
)


BYTE_VOCAB_SIZE = 256
PAD_TOKEN_ID = 256
QUESTION_TOKEN_ID = 257
ANSWER_START_TOKEN_ID = 258
VOCAB_SIZE = 259
CURRICULUM_CONTEXT_BYTES = [2048, 4096, 8192, 16384, 32768]
DEFAULT_LR_GRID = [1e-3, 5e-4]
DEFAULT_KR_GRID = [8, 16, 32]
DEFAULT_CHUNK_SIZE_GRID = [256, 512]
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_WARMUP_RATIO_GRID = [0.1, 0.2]
DEFAULT_SCHEDULED_SAMPLING_MAX_RATIO = 0.3
DEFAULT_SCHEDULED_SAMPLING_MAX_RATIO_GRID = [0.2, 0.3]
DEFAULT_TRAIN_DATASET_SIZE = 64
DEFAULT_TRAIN_DATASET_SEED = 7
DEFAULT_VALIDATION_DATASET_SIZE = 16
DEFAULT_VALIDATION_DATASET_SEED = 17
DEFAULT_TEST_DATASET_SIZE = 16
DEFAULT_TEST_DATASET_SEED = 23
DEFAULT_PAIR_SPLIT_SEED = 29
DEFAULT_TRAIN_PAIR_RATIO = 0.6
DEFAULT_VALIDATION_PAIR_RATIO = 0.2
DEFAULT_TARGET_CASE_SAMPLING_RATIO = 0.0
DEFAULT_FINAL_POLISH_EPOCHS = 0
DEFAULT_FINAL_GENERATION_POLISH_EPOCHS = 0
DEFAULT_GENERATION_POLISH_MAX_SELF_FEED_RATIO = 1.0
DEFAULT_GENERATION_POLISH_ROLLOUT_LOSS_WEIGHT = 1.0
DEFAULT_GENERATION_POLISH_TEACHER_FORCED_LOSS_WEIGHT = 1.0
DEFAULT_GENERATION_POLISH_BATCH_SIZE = 1
DEFAULT_GENERATION_POLISH_MONITOR_CASE_COUNT = 4
DEFAULT_TAIL_ERROR_TOKEN_COUNT = 32
DEFAULT_TAIL_ERROR_TOP_K = 3
DEFAULT_GENERALIZATION_SCORE_MODE = "teacher_forced"
DEFAULT_MODEL_TYPE = "dsra"
DEFAULT_LOCAL_CONTEXT_SIZE = 4
DEFAULT_LOCAL_CONTEXT_MODE = "concat"
DEFAULT_MUSEUM_SPAN_LOSS_WEIGHT = 1.0
DEFAULT_ARTIFACT_SPAN_LOSS_WEIGHT = 1.0
DEFAULT_ENTITY_SPAN_LOSS_MIN_CONTEXT_BYTES = 0
DEFAULT_MUSEUM_AUX_LOSS_WEIGHT = 0.0
DEFAULT_ARTIFACT_AUX_LOSS_WEIGHT = 0.0
DEFAULT_ENTITY_AUX_LOSS_MIN_CONTEXT_BYTES = 0
DEFAULT_MUSEUM_HINT_INJECTION_WEIGHT = 0.0
DEFAULT_ARTIFACT_HINT_INJECTION_WEIGHT = 0.0
DEFAULT_ENTITY_HINT_INJECTION_MIN_CONTEXT_BYTES = 0
DEFAULT_ENTITY_HINT_USE_GOLD_LABELS_DURING_TRAINING = False
DEFAULT_SLOT_DECODER_LOSS_WEIGHT = 0.0
DEFAULT_SLOT_DECODER_LOGIT_BIAS = 0.0
DEFAULT_SLOT_DECODER_MIN_CONTEXT_BYTES = 0
DEFAULT_EVIDENCE_WINDOW_COUNT = 16
DEFAULT_EVIDENCE_LOSS_WEIGHT = 0.0
DEFAULT_EVIDENCE_HINT_WEIGHT = 0.0
DEFAULT_EVIDENCE_MIN_CONTEXT_BYTES = 0
ENTITY_SPAN_NAMES = ("museum", "artifact")
ANSWER_SLOT_NAMES = ("museum", "artifact", "artist", "dynasty")
QUESTION_TEMPLATE = "What is the most valuable exhibit in the {museum}? Answer based on the context."
ANSWER_TEMPLATE = (
    "The most valuable exhibit in the {museum} is {artifact} painted by {artist} "
    "of the {dynasty} dynasty."
)
MUSEUM_NAMES = (
    "Palace Museum",
    "Grand Archive Museum",
    "Riverfront Gallery",
    "Imperial Heritage Hall",
    "Northern Art Museum",
    "Capital Relics Center",
)
ARTIFACT_NAMES = (
    "Along the River During the Qingming Festival",
    "Autumn Lantern Procession",
    "Golden Crane Panorama",
    "Jade Mountain Chronicle",
    "Celestial Market Scroll",
    "Spring Court Landscape",
)
ARTIST_NAMES = (
    "Zhang Zeduan",
    "Lin Qiao",
    "Guo Ming",
    "Shen Rui",
    "Han Yao",
    "Wei Cheng",
)
DYNASTY_NAMES = (
    "Northern Song",
    "Southern Song",
    "Tang",
    "Ming",
    "Han",
    "Jin",
)
FILLER_SUBJECTS = (
    "The gallery archive",
    "The museum record",
    "The restoration team",
    "The visitor guide",
    "The curatorial note",
    "The collection ledger",
)
FILLER_VERBS = (
    "describes",
    "documents",
    "summarizes",
    "explains",
    "records",
    "highlights",
)
FILLER_OBJECTS = (
    "bronze vessels",
    "ceramic figures",
    "court paintings",
    "stone inscriptions",
    "silk banners",
    "scholarly commentaries",
)
FILLER_DETAILS = (
    "from multiple dynastic periods",
    "preserved in climate controlled storage",
    "displayed beside archival notes",
    "cataloged for rotating exhibitions",
    "studied by conservation scholars",
    "referenced in visitor education programs",
)
ALL_MUSEUM_ARTIFACT_PAIRS = tuple(
    (museum, artifact)
    for museum in MUSEUM_NAMES
    for artifact in ARTIFACT_NAMES
)
MUSEUM_NAME_TO_ID = {museum: idx for idx, museum in enumerate(MUSEUM_NAMES)}
ARTIFACT_NAME_TO_ID = {artifact: idx for idx, artifact in enumerate(ARTIFACT_NAMES)}
ARTIST_NAME_TO_ID = {artist: idx for idx, artist in enumerate(ARTIST_NAMES)}
DYNASTY_NAME_TO_ID = {dynasty: idx for idx, dynasty in enumerate(DYNASTY_NAMES)}
SLOT_VALUE_LOOKUPS = {
    "museum": MUSEUM_NAMES,
    "artifact": ARTIFACT_NAMES,
    "artist": ARTIST_NAMES,
    "dynasty": DYNASTY_NAMES,
}
SLOT_NAME_TO_ID = {
    "museum": MUSEUM_NAME_TO_ID,
    "artifact": ARTIFACT_NAME_TO_ID,
    "artist": ARTIST_NAME_TO_ID,
    "dynasty": DYNASTY_NAME_TO_ID,
}


def build_retrieval_model(
    model_type,
    vocab_size,
    dim,
    K,
    kr,
    chunk_size,
    local_context_size,
    local_context_mode,
):
    """Build retrieval benchmark model family by canonical model type.

    中文说明:
    - 调用方 / Called by: `run_json_retrieval_test`,
      `run_json_retrieval_generalization_test`,
      `scripts.attention_family_benchmark.benchmark_attention_family_complexity`
    - 调用对象 / Calls:
      `DSRAModel`, `MHDSRA2Model`, `StandardAttentionModel`,
      `SlidingWindowAttentionModel`, `SparseAttentionModel`, `LinearAttentionModel`
    - 作用 / Purpose: 统一根据 `model_type` 构造 JSON retrieval 任务模型，确保 compare/runner 与原测试脚本口径一致
    - 变量 / Variables:
      `model_type` 模型家族名称, `vocab_size/dim/K/kr/chunk_size` 为结构参数,
      `local_context_size/local_context_mode` 控制局部上下文编码方式
    - 接入 / Integration: 新增模型家族时优先在本函数登记，避免各处重复分支
    - 错误处理 / Error handling: 未知 `model_type` 抛出 `ValueError`
    - 关键词 / Keywords:
      build_model|json_retrieval|mhdsra2|dsra|attention|factory|benchmark|model_type|统一入口|构建
    """
    if model_type == "dsra":
        return DSRAModel(
            vocab_size=vocab_size,
            dim=dim,
            K=K,
            kr=kr,
            chunk_size=chunk_size,
            pe_mode="none",
            use_orthogonal_update=True,
            use_bypass=True,
            local_context_size=local_context_size,
            local_context_mode=local_context_mode,
        )
    if model_type == "mhdsra2":
        return MHDSRA2Model(
            vocab_size=vocab_size,
            dim=dim,
            K=K,
            kr=kr,
            chunk_size=chunk_size,
            local_context_size=local_context_size,
            local_context_mode=local_context_mode,
        )
    if model_type == "standard_attention":
        return StandardAttentionModel(
            vocab_size=vocab_size,
            dim=dim,
            chunk_size=chunk_size,
            local_context_size=local_context_size,
            local_context_mode=local_context_mode,
        )
    if model_type == "sliding_window_attention":
        return SlidingWindowAttentionModel(
            vocab_size=vocab_size,
            dim=dim,
            chunk_size=chunk_size,
            local_context_size=local_context_size,
            local_context_mode=local_context_mode,
        )
    if model_type == "sparse_attention":
        return SparseAttentionModel(
            vocab_size=vocab_size,
            dim=dim,
            chunk_size=chunk_size,
            local_context_size=local_context_size,
            local_context_mode=local_context_mode,
        )
    if model_type == "linear_attention":
        return LinearAttentionModel(
            vocab_size=vocab_size,
            dim=dim,
            chunk_size=chunk_size,
            local_context_size=local_context_size,
            local_context_mode=local_context_mode,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def load_json_retrieval_case(
    input_path="tests/fixtures/test_input.json",
    metadata_path="tests/fixtures/test_metadata.json",
):
    base_dir = Path(__file__).resolve().parents[1]
    input_file = base_dir / input_path
    metadata_file = base_dir / metadata_path

    sample_bytes = bytes(json.loads(input_file.read_text(encoding="utf-8")))
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    expected_bytes = bytes(metadata["expected_answer_bytes"])

    return {
        "input_file": input_file,
        "metadata_file": metadata_file,
        "sample_bytes": sample_bytes,
        "metadata": metadata,
        "question_bytes": metadata["question"].encode("utf-8"),
        "expected_answer_bytes": expected_bytes,
    }


def build_noise_sentence(rng):
    subject = rng.choice(FILLER_SUBJECTS)
    verb = rng.choice(FILLER_VERBS)
    obj = rng.choice(FILLER_OBJECTS)
    detail = rng.choice(FILLER_DETAILS)
    return f"{subject} {verb} {obj} {detail}. "


def build_noise_bytes(target_bytes, rng):
    if target_bytes <= 0:
        return b""

    chunks = []
    total_bytes = 0
    while total_bytes < target_bytes:
        sentence = build_noise_sentence(rng)
        chunks.append(sentence)
        total_bytes += len(sentence)
    return "".join(chunks).encode("ascii")[:target_bytes]


def split_museum_artifact_pairs(
    seed,
    train_ratio=DEFAULT_TRAIN_PAIR_RATIO,
    validation_ratio=DEFAULT_VALIDATION_PAIR_RATIO,
):
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1.")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train_ratio + validation_ratio must be less than 1.")

    pairs = list(ALL_MUSEUM_ARTIFACT_PAIRS)
    rng = random.Random(seed)
    rng.shuffle(pairs)

    total_pairs = len(pairs)
    train_count = max(1, int(total_pairs * train_ratio))
    validation_count = max(1, int(total_pairs * validation_ratio))
    test_count = total_pairs - train_count - validation_count

    if test_count < 1:
        overflow = 1 - test_count
        if train_count >= validation_count and train_count - overflow >= 1:
            train_count -= overflow
        else:
            validation_count -= overflow
        test_count = total_pairs - train_count - validation_count

    train_pairs = tuple(pairs[:train_count])
    validation_pairs = tuple(pairs[train_count:train_count + validation_count])
    test_pairs = tuple(pairs[train_count + validation_count:])
    return {
        "train": train_pairs,
        "validation": validation_pairs,
        "test": test_pairs,
    }


def resolve_search_grid(explicit_grid, scalar_value, default_grid, default_scalar_value):
    if explicit_grid is not None:
        return explicit_grid
    if scalar_value != default_scalar_value:
        return [scalar_value]
    return default_grid


def generate_random_json_retrieval_case(
    reference_case,
    rng,
    target_total_bytes=None,
    allowed_museum_artifact_pairs=None,
    forced_museum_artifact_pair=None,
):
    metadata = reference_case["metadata"]
    total_bytes = int(target_total_bytes or metadata.get("target_total_bytes", len(reference_case["sample_bytes"])))
    if forced_museum_artifact_pair is not None:
        museum, artifact = forced_museum_artifact_pair
    else:
        candidate_pairs = (
            tuple(sorted(allowed_museum_artifact_pairs))
            if allowed_museum_artifact_pairs is not None
            else ALL_MUSEUM_ARTIFACT_PAIRS
        )
        if not candidate_pairs:
            raise ValueError("allowed_museum_artifact_pairs must contain at least one pair.")
        museum, artifact = rng.choice(candidate_pairs)
    artist = rng.choice(ARTIST_NAMES)
    dynasty = rng.choice(DYNASTY_NAMES)
    question = QUESTION_TEMPLATE.format(museum=museum)
    expected_answer_text = ANSWER_TEMPLATE.format(
        museum=museum,
        artifact=artifact,
        artist=artist,
        dynasty=dynasty,
    )
    expected_answer_bytes = expected_answer_text.encode("ascii")
    filler_bytes = build_noise_bytes(total_bytes - len(expected_answer_bytes), rng)
    needle_position_pct = rng.uniform(0.15, 0.85)
    max_insert_position = len(filler_bytes)
    desired_insert_position = int(total_bytes * needle_position_pct)
    insert_position = max(0, min(desired_insert_position, max_insert_position))
    sample_bytes = (
        filler_bytes[:insert_position]
        + expected_answer_bytes
        + filler_bytes[insert_position:]
    )

    return {
        "sample_bytes": sample_bytes,
        "metadata": {
            "target_total_bytes": total_bytes,
            "actual_total_bytes": len(sample_bytes),
            "needle_bytes": len(expected_answer_bytes),
            "needle_position_pct": needle_position_pct,
            "insert_position_byte_index": insert_position,
            "museum": museum,
            "artifact": artifact,
            "artist": artist,
            "dynasty": dynasty,
            "question": question,
            "expected_answer_text": expected_answer_text,
            "expected_answer_bytes": list(expected_answer_bytes),
        },
        "question_bytes": question.encode("ascii"),
        "expected_answer_bytes": expected_answer_bytes,
    }


def build_random_training_case_pool(
    reference_case,
    dataset_size,
    seed,
    allowed_museum_artifact_pairs=None,
):
    rng = random.Random(seed)
    return [
        generate_random_json_retrieval_case(
            reference_case,
            rng,
            allowed_museum_artifact_pairs=allowed_museum_artifact_pairs,
        )
        for _ in range(max(1, dataset_size))
    ]


def build_case_signature(case):
    metadata = case.get("metadata", {})
    return (
        metadata.get("question"),
        metadata.get("expected_answer_text"),
    )


def get_case_museum_artifact_pair(case):
    metadata = case.get("metadata", {})
    museum = metadata.get("museum")
    artifact = metadata.get("artifact")
    if museum is None or artifact is None:
        return None
    return museum, artifact


def build_disjoint_case_pool(
    reference_case,
    dataset_size,
    seed,
    used_signatures=None,
    allowed_museum_artifact_pairs=None,
):
    target_size = max(1, dataset_size)
    rng = random.Random(seed)
    cases = []
    seen_signatures = set() if used_signatures is None else set(used_signatures)
    max_attempts = max(target_size * 64, 256)
    attempts = 0

    while len(cases) < target_size and attempts < max_attempts:
        attempts += 1
        candidate = generate_random_json_retrieval_case(
            reference_case,
            rng,
            allowed_museum_artifact_pairs=allowed_museum_artifact_pairs,
        )
        signature = build_case_signature(candidate)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        cases.append(candidate)

    if len(cases) < target_size:
        raise ValueError(
            f"Unable to build a disjoint case pool of size {target_size} with seed {seed}."
        )
    return cases, seen_signatures


def select_training_case(training_cases, reference_case, sampler, target_case_sampling_ratio):
    target_case_sampling_ratio = float(max(0.0, min(target_case_sampling_ratio, 1.0)))
    if target_case_sampling_ratio >= 1.0:
        return reference_case
    if target_case_sampling_ratio > 0.0 and sampler.random() < target_case_sampling_ratio:
        return reference_case
    return sampler.choice(training_cases)


def build_curriculum_context(case, context_bytes):
    start, end = get_curriculum_window_bounds(case, context_bytes)
    return case["sample_bytes"][start:end]


def get_curriculum_window_bounds(case, context_bytes):
    sample_bytes = case["sample_bytes"]
    total_bytes = len(sample_bytes)
    if context_bytes is None or context_bytes >= total_bytes:
        return 0, total_bytes

    needle_start = case["metadata"]["insert_position_byte_index"]
    needle_end = needle_start + case["metadata"]["needle_bytes"]
    target_ratio = case["metadata"]["needle_position_pct"]
    desired_start = needle_start - int(context_bytes * target_ratio)
    max_start = max(0, total_bytes - context_bytes)
    start = max(0, min(desired_start, max_start))
    end = start + context_bytes

    if needle_end > end:
        start = max(0, needle_end - context_bytes)
        end = start + context_bytes
    if needle_start < start:
        start = needle_start
        end = min(total_bytes, start + context_bytes)

    return start, end


def get_relative_needle_bounds(case, context_bytes=None):
    window_start, window_end = get_curriculum_window_bounds(case, context_bytes)
    relative_start = case["metadata"]["insert_position_byte_index"] - window_start
    relative_end = relative_start + case["metadata"]["needle_bytes"]
    relative_start = max(0, min(relative_start, window_end - window_start))
    relative_end = max(relative_start, min(relative_end, window_end - window_start))
    return relative_start, relative_end


def get_evidence_window_target(case, context_bytes=None, window_count=DEFAULT_EVIDENCE_WINDOW_COUNT):
    sample_length = len(build_curriculum_context(case, context_bytes))
    if sample_length <= 0:
        return 0
    relative_start, relative_end = get_relative_needle_bounds(case, context_bytes=context_bytes)
    needle_center = (relative_start + max(relative_start, relative_end - 1)) / 2.0
    target_window = int((needle_center / max(1, sample_length)) * max(1, window_count))
    return max(0, min(int(window_count) - 1, target_window))


def get_evidence_window_bounds(sample_length, window_index, window_count):
    sample_length = max(1, int(sample_length))
    window_count = max(1, int(window_count))
    window_index = max(0, min(int(window_index), window_count - 1))
    start_index = (window_index * sample_length) // window_count
    end_index = ((window_index + 1) * sample_length) // window_count
    if end_index <= start_index:
        end_index = min(sample_length, start_index + 1)
    return start_index, end_index


def build_training_example(case, context_bytes=None):
    sample_bytes = build_curriculum_context(case, context_bytes) if context_bytes is not None else case["sample_bytes"]
    sample_tokens = list(sample_bytes)
    question_tokens = list(case["question_bytes"])
    answer_tokens = list(case["expected_answer_bytes"])

    x_tokens = sample_tokens + [QUESTION_TOKEN_ID] + question_tokens + [ANSWER_START_TOKEN_ID]
    y_tokens = [PAD_TOKEN_ID] * len(x_tokens)

    for next_token in answer_tokens[:-1]:
        x_tokens.append(next_token)
        y_tokens.append(PAD_TOKEN_ID)

    answer_start_index = len(sample_tokens) + 1 + len(question_tokens)
    x_prefix_index = answer_start_index
    for offset, target_token in enumerate(answer_tokens):
        y_tokens[x_prefix_index + offset] = target_token

    X = torch.tensor([x_tokens], dtype=torch.long)
    Y = torch.tensor([y_tokens], dtype=torch.long)
    return X, Y


def build_generation_prompt(case, context_bytes=None):
    sample_bytes = build_curriculum_context(case, context_bytes) if context_bytes is not None else case["sample_bytes"]
    sample_tokens = list(sample_bytes)
    question_tokens = list(case["question_bytes"])
    answer_tokens = list(case["expected_answer_bytes"])
    prompt_tokens = sample_tokens + [QUESTION_TOKEN_ID] + question_tokens + [ANSWER_START_TOKEN_ID]
    return prompt_tokens, answer_tokens


def get_answer_start_index(case, context_bytes=None):
    sample_bytes = build_curriculum_context(case, context_bytes) if context_bytes is not None else case["sample_bytes"]
    return len(sample_bytes) + 1 + len(case["question_bytes"])


def build_answer_text_from_metadata(metadata):
    return ANSWER_TEMPLATE.format(
        museum=metadata["museum"],
        artifact=metadata["artifact"],
        artist=metadata["artist"],
        dynasty=metadata["dynasty"],
    )


def get_answer_slot_spans(case):
    expected_answer_bytes = case["expected_answer_bytes"]
    metadata = case.get("metadata", {})
    spans = {}

    for slot_name in ANSWER_SLOT_NAMES:
        slot_value = metadata.get(slot_name)
        if not slot_value:
            continue
        slot_bytes = slot_value.encode("ascii")
        start_index = expected_answer_bytes.find(slot_bytes)
        if start_index < 0:
            continue
        spans[slot_name] = {
            "label": slot_value,
            "start": start_index,
            "end": start_index + len(slot_bytes),
            "length": len(slot_bytes),
        }

    return spans


def get_answer_entity_spans(case):
    slot_spans = get_answer_slot_spans(case)
    return {
        entity_name: slot_spans[entity_name]
        for entity_name in ENTITY_SPAN_NAMES
        if entity_name in slot_spans
    }


def collect_answer_targets(logits, Y):
    target_indices = (Y != PAD_TOKEN_ID).nonzero(as_tuple=True)
    logits_target = logits[target_indices[0], target_indices[1], :]
    targets = Y[target_indices[0], target_indices[1]]
    return logits_target, targets


def predict_byte_tokens(logits):
    # Only raw byte ids are valid answer tokens for this task.
    return logits[..., :BYTE_VOCAB_SIZE].argmax(dim=-1)


def compute_sequence_metrics(predicted_tokens, expected_tokens):
    compared_length = max(len(expected_tokens), 1)
    prefix_match_length = 0
    first_mismatch_index = None

    for idx, (predicted, expected) in enumerate(zip(predicted_tokens, expected_tokens)):
        if predicted != expected:
            first_mismatch_index = idx
            break
        prefix_match_length += 1

    if first_mismatch_index is None:
        if len(predicted_tokens) != len(expected_tokens):
            first_mismatch_index = min(len(predicted_tokens), len(expected_tokens))
        elif len(expected_tokens) > prefix_match_length:
            first_mismatch_index = prefix_match_length

    correct_count = sum(int(predicted == expected) for predicted, expected in zip(predicted_tokens, expected_tokens))
    sequence_accuracy = correct_count / compared_length

    first_mismatch_expected = None
    first_mismatch_predicted = None
    if first_mismatch_index is not None:
        if first_mismatch_index < len(expected_tokens):
            first_mismatch_expected = expected_tokens[first_mismatch_index]
        if first_mismatch_index < len(predicted_tokens):
            first_mismatch_predicted = predicted_tokens[first_mismatch_index]

    return {
        "prefix_match_length": prefix_match_length,
        "sequence_accuracy": sequence_accuracy,
        "first_mismatch_index": first_mismatch_index,
        "first_mismatch_expected_byte": first_mismatch_expected,
        "first_mismatch_predicted_byte": first_mismatch_predicted,
    }


def compute_slice_metrics(predicted_tokens, expected_tokens, start_index, end_index):
    start_index = max(0, int(start_index))
    end_index = max(start_index, min(int(end_index), len(expected_tokens)))
    expected_slice = expected_tokens[start_index:end_index]
    predicted_slice = predicted_tokens[start_index:end_index]
    slice_length = len(expected_slice)
    if slice_length == 0:
        return {
            "start": start_index,
            "end": end_index,
            "length": 0,
            "exact_match": False,
            "sequence_accuracy": 0.0,
            "prefix_match_length": 0,
        }

    correct_count = sum(
        int(predicted == expected)
        for predicted, expected in zip(predicted_slice, expected_slice)
    )
    prefix_match_length = 0
    for predicted, expected in zip(predicted_slice, expected_slice):
        if predicted != expected:
            break
        prefix_match_length += 1

    return {
        "start": start_index,
        "end": end_index,
        "length": slice_length,
        "exact_match": predicted_slice == expected_slice,
        "sequence_accuracy": correct_count / slice_length,
        "prefix_match_length": prefix_match_length,
    }


def compute_tail_slice_metrics(predicted_tokens, expected_tokens, tail_token_count=DEFAULT_TAIL_ERROR_TOKEN_COUNT):
    tail_token_count = max(1, int(tail_token_count))
    effective_tail_token_count = min(tail_token_count, max(1, len(expected_tokens)))
    slice_metrics = compute_slice_metrics(
        predicted_tokens,
        expected_tokens,
        start_index=len(expected_tokens) - effective_tail_token_count,
        end_index=len(expected_tokens),
    )
    return {
        "tail_token_count": effective_tail_token_count,
        "tail_sequence_accuracy": slice_metrics["sequence_accuracy"],
        "tail_exact_match": slice_metrics["exact_match"],
    }


def compute_entity_span_metrics(predicted_tokens, case):
    expected_tokens = list(case["expected_answer_bytes"])
    entity_spans = get_answer_entity_spans(case)
    entity_metrics = {}
    for entity_name in ENTITY_SPAN_NAMES:
        span = entity_spans.get(entity_name)
        if span is None:
            continue
        entity_metrics[entity_name] = {
            "label": span["label"],
            **compute_slice_metrics(
                predicted_tokens,
                expected_tokens,
                start_index=span["start"],
                end_index=span["end"],
            ),
        }
    return entity_metrics


def build_answer_loss_weights(
    case,
    target_length,
    device,
    museum_span_loss_weight=DEFAULT_MUSEUM_SPAN_LOSS_WEIGHT,
    artifact_span_loss_weight=DEFAULT_ARTIFACT_SPAN_LOSS_WEIGHT,
):
    target_length = max(0, int(target_length))
    weights = torch.ones(target_length, dtype=torch.float32, device=device)
    if target_length == 0:
        return weights

    span_weight_overrides = {
        "museum": float(max(0.0, museum_span_loss_weight)),
        "artifact": float(max(0.0, artifact_span_loss_weight)),
    }
    entity_spans = get_answer_entity_spans(case)
    for entity_name, weight in span_weight_overrides.items():
        span = entity_spans.get(entity_name)
        if span is None or weight == 1.0:
            continue
        span_start = max(0, min(span["start"], target_length))
        span_end = max(span_start, min(span["end"], target_length))
        if span_end > span_start:
            weights[span_start:span_end] *= weight
    return weights


def compute_weighted_answer_loss(
    logits_target,
    targets,
    case,
    museum_span_loss_weight=DEFAULT_MUSEUM_SPAN_LOSS_WEIGHT,
    artifact_span_loss_weight=DEFAULT_ARTIFACT_SPAN_LOSS_WEIGHT,
):
    per_token_loss = F.cross_entropy(logits_target, targets, reduction="none")
    token_weights = build_answer_loss_weights(
        case=case,
        target_length=targets.numel(),
        device=per_token_loss.device,
        museum_span_loss_weight=museum_span_loss_weight,
        artifact_span_loss_weight=artifact_span_loss_weight,
    )
    normalizer = token_weights.sum().clamp_min(1e-8)
    return (per_token_loss * token_weights).sum() / normalizer


def attach_entity_auxiliary_heads(model, device):
    if not hasattr(model, "museum_aux_head"):
        model.museum_aux_head = nn.Linear(model.dim, len(MUSEUM_NAMES)).to(device)
    if not hasattr(model, "artifact_aux_head"):
        model.artifact_aux_head = nn.Linear(model.dim, len(ARTIFACT_NAMES)).to(device)
    if not hasattr(model, "museum_hint_embedding"):
        model.museum_hint_embedding = nn.Embedding(len(MUSEUM_NAMES), model.dim).to(device)
    if not hasattr(model, "artifact_hint_embedding"):
        model.artifact_hint_embedding = nn.Embedding(len(ARTIFACT_NAMES), model.dim).to(device)
    if not hasattr(model, "entity_hint_proj"):
        model.entity_hint_proj = nn.Linear(model.dim * 2, model.dim).to(device)
    if not hasattr(model, "entity_hint_gate"):
        model.entity_hint_gate = nn.Linear(model.dim, 1).to(device)


def get_entity_auxiliary_logits(model, answer_start_hidden):
    museum_logits = model.museum_aux_head(answer_start_hidden) if hasattr(model, "museum_aux_head") else None
    artifact_logits = model.artifact_aux_head(answer_start_hidden) if hasattr(model, "artifact_aux_head") else None
    return museum_logits, artifact_logits


def attach_slot_decoder_heads(model, device):
    if not hasattr(model, "museum_slot_head"):
        model.museum_slot_head = nn.Linear(model.dim, len(MUSEUM_NAMES)).to(device)
    if not hasattr(model, "artifact_slot_head"):
        model.artifact_slot_head = nn.Linear(model.dim, len(ARTIFACT_NAMES)).to(device)
    if not hasattr(model, "artist_slot_head"):
        model.artist_slot_head = nn.Linear(model.dim, len(ARTIST_NAMES)).to(device)
    if not hasattr(model, "dynasty_slot_head"):
        model.dynasty_slot_head = nn.Linear(model.dim, len(DYNASTY_NAMES)).to(device)
    if not hasattr(model, "slot_decoder_query_proj"):
        model.slot_decoder_query_proj = nn.Linear(model.dim, model.dim).to(device)
    if not hasattr(model, "slot_decoder_key_proj"):
        model.slot_decoder_key_proj = nn.Linear(model.dim, model.dim).to(device)
    if not hasattr(model, "slot_decoder_feature_proj"):
        model.slot_decoder_feature_proj = nn.Linear(model.dim * 2, model.dim).to(device)
    if not hasattr(model, "slot_decoder_evidence_fusion_proj"):
        model.slot_decoder_evidence_fusion_proj = nn.Linear(model.dim * 2, model.dim).to(device)


def build_slot_decoder_features(model, hidden_states, case, context_bytes=None, evidence_state=None):
    answer_start_index = get_answer_start_index(case, context_bytes=context_bytes)
    answer_start_hidden = hidden_states[:, answer_start_index, :]
    prefix_hidden_states = hidden_states[:, :answer_start_index + 1, :]
    if not hasattr(model, "slot_decoder_query_proj"):
        return answer_start_hidden

    query = model.slot_decoder_query_proj(answer_start_hidden).unsqueeze(1)
    keys = model.slot_decoder_key_proj(prefix_hidden_states)
    scale = float(model.dim) ** 0.5
    attn_scores = torch.matmul(query, keys.transpose(1, 2)) / scale
    attn_weights = torch.softmax(attn_scores, dim=-1)
    pooled_prefix_hidden = torch.matmul(attn_weights, prefix_hidden_states).squeeze(1)
    slot_decoder_features = model.slot_decoder_feature_proj(
        torch.cat([answer_start_hidden, pooled_prefix_hidden], dim=-1)
    )
    if evidence_state is not None and hasattr(model, "slot_decoder_evidence_fusion_proj"):
        slot_decoder_features = model.slot_decoder_evidence_fusion_proj(
            torch.cat([slot_decoder_features, evidence_state["context_vector"]], dim=-1)
        )
    return slot_decoder_features


def get_slot_decoder_logits(model, slot_decoder_features):
    return {
        "museum": model.museum_slot_head(slot_decoder_features) if hasattr(model, "museum_slot_head") else None,
        "artifact": model.artifact_slot_head(slot_decoder_features) if hasattr(model, "artifact_slot_head") else None,
        "artist": model.artist_slot_head(slot_decoder_features) if hasattr(model, "artist_slot_head") else None,
        "dynasty": model.dynasty_slot_head(slot_decoder_features) if hasattr(model, "dynasty_slot_head") else None,
    }


def slot_decoder_is_active(model, context_bytes=None):
    slot_decoder_loss_weight = float(max(0.0, getattr(model, "slot_decoder_loss_weight", 0.0)))
    slot_decoder_logit_bias = float(max(0.0, getattr(model, "slot_decoder_logit_bias", 0.0)))
    if slot_decoder_loss_weight <= 0.0 and slot_decoder_logit_bias <= 0.0:
        return False
    min_context_bytes = int(getattr(model, "slot_decoder_min_context_bytes", 0))
    if context_bytes is None:
        return True
    return int(context_bytes) >= min_context_bytes


def build_slot_decoder_state(
    model,
    hidden_states,
    case,
    context_bytes=None,
    use_gold_slot_labels=False,
    evidence_state=None,
):
    slot_decoder_features = build_slot_decoder_features(
        model,
        hidden_states,
        case,
        context_bytes=context_bytes,
        evidence_state=evidence_state,
    )
    slot_logits = get_slot_decoder_logits(model, slot_decoder_features)
    if any(slot_logits[slot_name] is None for slot_name in ANSWER_SLOT_NAMES):
        return None

    metadata = case.get("metadata", {})
    slot_spans = get_answer_slot_spans(case)
    slot_predictions = {}
    for slot_name in ANSWER_SLOT_NAMES:
        logits = slot_logits[slot_name]
        probs = torch.softmax(logits, dim=-1)
        confidence, predicted_ids = probs.max(dim=-1)
        if use_gold_slot_labels:
            predicted_ids = torch.tensor(
                [SLOT_NAME_TO_ID[slot_name][metadata[slot_name]]],
                dtype=torch.long,
                device=hidden_states.device,
            )
            confidence = torch.ones_like(confidence)
        predicted_label = SLOT_VALUE_LOOKUPS[slot_name][predicted_ids.item()]
        slot_predictions[slot_name] = {
            "predicted_id": predicted_ids,
            "predicted_label": predicted_label,
            "predicted_bytes": predicted_label.encode("ascii"),
            "confidence": confidence,
            "logits": logits,
            "target_label": metadata.get(slot_name),
        }

    return {
        "slot_predictions": slot_predictions,
        "slot_spans": slot_spans,
    }


def apply_slot_decoder_bias_to_logits(
    model,
    logits,
    case,
    hidden_states,
    context_bytes=None,
    slot_state=None,
    use_gold_slot_labels=False,
    evidence_state=None,
):
    if not slot_decoder_is_active(model, context_bytes=context_bytes):
        return logits, slot_state

    slot_decoder_logit_bias = float(max(0.0, getattr(model, "slot_decoder_logit_bias", 0.0)))
    if slot_decoder_logit_bias <= 0.0:
        return logits, slot_state

    if slot_state is None:
        slot_state = build_slot_decoder_state(
            model,
            hidden_states,
            case,
            context_bytes=context_bytes,
            use_gold_slot_labels=use_gold_slot_labels,
            evidence_state=evidence_state,
        )
        if slot_state is None:
            return logits, None

    biased_logits = logits.clone()
    answer_start_index = get_answer_start_index(case, context_bytes=context_bytes)
    sequence_length = biased_logits.shape[1]
    for slot_name, prediction in slot_state["slot_predictions"].items():
        slot_span = slot_state["slot_spans"].get(slot_name)
        if slot_span is None:
            continue
        predicted_bytes = prediction["predicted_bytes"]
        if not predicted_bytes:
            continue
        span_start_index = answer_start_index + slot_span["start"]
        span_end_index = min(answer_start_index + slot_span["end"], sequence_length)
        max_bias_tokens = min(len(predicted_bytes), max(0, span_end_index - span_start_index))
        if max_bias_tokens <= 0:
            continue
        confidence_scale = prediction["confidence"].unsqueeze(-1)
        for token_offset in range(max_bias_tokens):
            biased_logits[:, span_start_index + token_offset, predicted_bytes[token_offset]] += (
                slot_decoder_logit_bias * confidence_scale.squeeze(-1)
            )
    return biased_logits, slot_state


def apply_slot_decoder_bias_to_step_logits(model, next_logits, slot_state, answer_token_index):
    if slot_state is None:
        return next_logits

    slot_decoder_logit_bias = float(max(0.0, getattr(model, "slot_decoder_logit_bias", 0.0)))
    if slot_decoder_logit_bias <= 0.0:
        return next_logits

    biased_logits = next_logits.clone()
    for slot_name, prediction in slot_state["slot_predictions"].items():
        slot_span = slot_state["slot_spans"].get(slot_name)
        if slot_span is None or not (slot_span["start"] <= answer_token_index < slot_span["end"]):
            continue
        token_offset = answer_token_index - slot_span["start"]
        predicted_bytes = prediction["predicted_bytes"]
        if token_offset >= len(predicted_bytes):
            continue
        biased_logits[:, predicted_bytes[token_offset]] += (
            slot_decoder_logit_bias * prediction["confidence"]
        )
    return biased_logits


def attach_evidence_heads(model, device):
    if not hasattr(model, "evidence_query_proj"):
        model.evidence_query_proj = nn.Linear(model.dim, model.dim).to(device)
    if not hasattr(model, "evidence_key_proj"):
        model.evidence_key_proj = nn.Linear(model.dim, model.dim).to(device)
    if not hasattr(model, "evidence_hint_proj"):
        model.evidence_hint_proj = nn.Linear(model.dim, model.dim).to(device)


def evidence_supervision_is_active(model, context_bytes=None):
    evidence_loss_weight = float(max(0.0, getattr(model, "evidence_loss_weight", 0.0)))
    evidence_hint_weight = float(max(0.0, getattr(model, "evidence_hint_weight", 0.0)))
    if evidence_loss_weight <= 0.0 and evidence_hint_weight <= 0.0:
        return False
    min_context_bytes = int(getattr(model, "evidence_min_context_bytes", 0))
    if context_bytes is None:
        return True
    return int(context_bytes) >= min_context_bytes


def build_evidence_window_features(hidden_states, case, context_bytes=None, window_count=DEFAULT_EVIDENCE_WINDOW_COUNT):
    sample_length = len(build_curriculum_context(case, context_bytes))
    sample_hidden_states = hidden_states[:, :sample_length, :]
    window_count = max(1, int(window_count))
    window_features = []
    window_ranges = []
    for window_idx in range(window_count):
        start_index = (window_idx * sample_length) // window_count
        end_index = ((window_idx + 1) * sample_length) // window_count
        if end_index <= start_index:
            end_index = min(sample_length, start_index + 1)
        if start_index >= sample_length:
            start_index = max(0, sample_length - 1)
            end_index = sample_length
        window_hidden = sample_hidden_states[:, start_index:end_index, :]
        window_features.append(window_hidden.mean(dim=1, keepdim=True))
        window_ranges.append((start_index, end_index))
    return torch.cat(window_features, dim=1), window_ranges


def build_evidence_state(model, hidden_states, case, context_bytes=None, use_gold_window=False):
    if not hasattr(model, "evidence_query_proj") or not hasattr(model, "evidence_key_proj"):
        return None
    if hidden_states.shape[1] == 0:
        return None
    answer_start_index = get_answer_start_index(case, context_bytes=context_bytes)
    if answer_start_index >= hidden_states.shape[1]:
        return None
    evidence_window_count = int(max(1, getattr(model, "evidence_window_count", DEFAULT_EVIDENCE_WINDOW_COUNT)))
    answer_start_hidden = hidden_states[:, answer_start_index, :]
    window_features, window_ranges = build_evidence_window_features(
        hidden_states,
        case,
        context_bytes=context_bytes,
        window_count=evidence_window_count,
    )
    query = model.evidence_query_proj(answer_start_hidden).unsqueeze(1)
    keys = model.evidence_key_proj(window_features)
    scale = float(model.dim) ** 0.5
    window_logits = torch.matmul(query, keys.transpose(1, 2)).squeeze(1) / scale
    window_probs = torch.softmax(window_logits, dim=-1)
    confidence, predicted_window = window_probs.max(dim=-1)
    target_window = get_evidence_window_target(
        case,
        context_bytes=context_bytes,
        window_count=evidence_window_count,
    )
    if use_gold_window:
        confidence = torch.ones_like(confidence)
        predicted_window = torch.tensor([target_window], dtype=torch.long, device=hidden_states.device)
        window_probs = F.one_hot(predicted_window, num_classes=evidence_window_count).to(hidden_states.dtype)
    context_vector = torch.matmul(window_probs.unsqueeze(1), window_features).squeeze(1)
    return {
        "window_logits": window_logits,
        "window_probs": window_probs,
        "predicted_window": predicted_window,
        "target_window": target_window,
        "confidence": confidence,
        "context_vector": context_vector,
        "window_ranges": window_ranges,
        "case": case,
    }


def apply_evidence_hint_to_hidden_states(model, hidden_states, case, context_bytes=None, evidence_state=None):
    if not evidence_supervision_is_active(model, context_bytes=context_bytes):
        return hidden_states, evidence_state

    evidence_hint_weight = float(max(0.0, getattr(model, "evidence_hint_weight", 0.0)))
    if evidence_hint_weight <= 0.0:
        return hidden_states, evidence_state

    if evidence_state is None:
        evidence_state = build_evidence_state(model, hidden_states, case, context_bytes=context_bytes)
        if evidence_state is None:
            return hidden_states, None

    answer_start_index = get_answer_start_index(case, context_bytes=context_bytes)
    slot_spans = get_answer_slot_spans(case)
    first_slot_offset = min((slot_span["start"] for slot_span in slot_spans.values()), default=0)
    inject_start_index = min(hidden_states.shape[1], answer_start_index + first_slot_offset)
    if inject_start_index >= hidden_states.shape[1]:
        return hidden_states, evidence_state

    evidence_vector = model.evidence_hint_proj(evidence_state["context_vector"])
    scaled_vector = evidence_vector * evidence_hint_weight * evidence_state["confidence"].unsqueeze(-1)
    injected_hidden_states = hidden_states.clone()
    injected_hidden_states[:, inject_start_index:, :] = (
        injected_hidden_states[:, inject_start_index:, :] + scaled_vector.unsqueeze(1)
    )
    return injected_hidden_states, evidence_state


def apply_evidence_hint_to_step_hidden(model, hidden_states, evidence_state, answer_token_index):
    if evidence_state is None:
        return hidden_states

    evidence_hint_weight = float(max(0.0, getattr(model, "evidence_hint_weight", 0.0)))
    if evidence_hint_weight <= 0.0:
        return hidden_states

    slot_spans = get_answer_slot_spans(evidence_state["case"])
    first_slot_offset = min((slot_span["start"] for slot_span in slot_spans.values()), default=0)
    if answer_token_index < first_slot_offset:
        return hidden_states

    evidence_vector = model.evidence_hint_proj(evidence_state["context_vector"])
    scaled_vector = evidence_vector * evidence_hint_weight * evidence_state["confidence"].unsqueeze(-1)
    return hidden_states + scaled_vector.unsqueeze(1)


def entity_hinting_is_active(model, context_bytes=None):
    museum_hint_weight = float(getattr(model, "museum_hint_injection_weight", 0.0))
    artifact_hint_weight = float(getattr(model, "artifact_hint_injection_weight", 0.0))
    if museum_hint_weight <= 0.0 and artifact_hint_weight <= 0.0:
        return False
    min_context_bytes = int(getattr(model, "entity_hint_injection_min_context_bytes", 0))
    if context_bytes is None:
        return True
    return int(context_bytes) >= min_context_bytes


def build_entity_hint_state(model, answer_start_hidden, case, use_gold_entity_labels=False):
    museum_hint_weight = float(max(0.0, getattr(model, "museum_hint_injection_weight", 0.0)))
    artifact_hint_weight = float(max(0.0, getattr(model, "artifact_hint_injection_weight", 0.0)))
    if museum_hint_weight <= 0.0 and artifact_hint_weight <= 0.0:
        return None

    museum_logits, artifact_logits = get_entity_auxiliary_logits(model, answer_start_hidden)
    if museum_logits is None or artifact_logits is None:
        return None

    museum_probs = torch.softmax(museum_logits, dim=-1)
    artifact_probs = torch.softmax(artifact_logits, dim=-1)
    museum_confidence, museum_predicted_ids = museum_probs.max(dim=-1)
    artifact_confidence, artifact_predicted_ids = artifact_probs.max(dim=-1)
    if use_gold_entity_labels:
        metadata = case.get("metadata", {})
        museum_predicted_ids = torch.tensor(
            [MUSEUM_NAME_TO_ID[metadata["museum"]]],
            dtype=torch.long,
            device=answer_start_hidden.device,
        )
        artifact_predicted_ids = torch.tensor(
            [ARTIFACT_NAME_TO_ID[metadata["artifact"]]],
            dtype=torch.long,
            device=answer_start_hidden.device,
        )
        museum_confidence = torch.ones_like(museum_confidence)
        artifact_confidence = torch.ones_like(artifact_confidence)
    museum_hint_vector = (
        model.museum_hint_embedding(museum_predicted_ids)
        * museum_hint_weight
        * museum_confidence.unsqueeze(-1)
    )
    artifact_hint_vector = (
        model.artifact_hint_embedding(artifact_predicted_ids)
        * artifact_hint_weight
        * artifact_confidence.unsqueeze(-1)
    )
    return {
        "museum_hint_vector": museum_hint_vector,
        "artifact_hint_vector": artifact_hint_vector,
        "museum_confidence": museum_confidence,
        "artifact_confidence": artifact_confidence,
        "museum_predicted_id": museum_predicted_ids,
        "artifact_predicted_id": artifact_predicted_ids,
        "museum_logits": museum_logits,
        "artifact_logits": artifact_logits,
        "entity_spans": get_answer_entity_spans(case),
    }


def apply_entity_hint_to_hidden_states(
    model,
    hidden_states,
    case,
    context_bytes=None,
    hint_state=None,
    use_gold_entity_labels=False,
):
    if not entity_hinting_is_active(model, context_bytes=context_bytes):
        return hidden_states, hint_state

    if hint_state is None:
        answer_start_index = get_answer_start_index(case, context_bytes=context_bytes)
        if answer_start_index >= hidden_states.shape[1]:
            return hidden_states, None
        answer_start_hidden = hidden_states[:, answer_start_index, :]
        hint_state = build_entity_hint_state(
            model,
            answer_start_hidden,
            case,
            use_gold_entity_labels=use_gold_entity_labels,
        )
        if hint_state is None:
            return hidden_states, None

    injected_hidden_states = hidden_states.clone()
    answer_start_index = get_answer_start_index(case, context_bytes=context_bytes)
    for entity_name, hint_key in (("museum", "museum_hint_vector"), ("artifact", "artifact_hint_vector")):
        span = hint_state["entity_spans"].get(entity_name)
        if span is None:
            continue
        span_start_index = answer_start_index + span["start"]
        span_end_index = min(hidden_states.shape[1], answer_start_index + span["end"])
        if span_start_index >= span_end_index:
            continue
        injected_hidden_states[:, span_start_index:span_end_index, :] = (
            injected_hidden_states[:, span_start_index:span_end_index, :]
            + hint_state[hint_key].unsqueeze(1)
        )
    return injected_hidden_states, hint_state


def apply_entity_hint_to_step_hidden(model, hidden_states, hint_state, answer_token_index):
    if hint_state is None:
        return hidden_states
    injected_hidden_states = hidden_states
    for entity_name, hint_key in (("museum", "museum_hint_vector"), ("artifact", "artifact_hint_vector")):
        span = hint_state["entity_spans"].get(entity_name)
        if span is None:
            continue
        if span["start"] <= answer_token_index < span["end"]:
            injected_hidden_states = injected_hidden_states + hint_state[hint_key].unsqueeze(1)
    return injected_hidden_states


def forward_json_retrieval(model, X, case, context_bytes=None, return_hidden=False, use_gold_entity_hints=False):
    need_hidden = (
        return_hidden
        or entity_hinting_is_active(model, context_bytes=context_bytes)
        or slot_decoder_is_active(model, context_bytes=context_bytes)
        or evidence_supervision_is_active(model, context_bytes=context_bytes)
    )
    if not need_hidden:
        logits = model(X)
        if return_hidden:
            return logits, None, None
        return logits

    logits, hidden_states = model(X, return_hidden=True)
    aux_state = {}
    active_hidden_states = hidden_states
    if entity_hinting_is_active(model, context_bytes=context_bytes):
        active_hidden_states, hint_state = apply_entity_hint_to_hidden_states(
            model=model,
            hidden_states=hidden_states,
            case=case,
            context_bytes=context_bytes,
            use_gold_entity_labels=use_gold_entity_hints,
        )
        logits = model.out_proj(active_hidden_states)
        aux_state["entity_hint_state"] = hint_state
    if evidence_supervision_is_active(model, context_bytes=context_bytes):
        active_hidden_states, evidence_state = apply_evidence_hint_to_hidden_states(
            model=model,
            hidden_states=active_hidden_states,
            case=case,
            context_bytes=context_bytes,
        )
        logits = model.out_proj(active_hidden_states)
        aux_state["evidence_state"] = evidence_state
    if slot_decoder_is_active(model, context_bytes=context_bytes):
        answer_start_index = get_answer_start_index(case, context_bytes=context_bytes)
        if answer_start_index < active_hidden_states.shape[1]:
            logits, slot_decoder_state = apply_slot_decoder_bias_to_logits(
                model=model,
                logits=logits,
                case=case,
                hidden_states=active_hidden_states,
                context_bytes=context_bytes,
                evidence_state=aux_state.get("evidence_state"),
            )
            aux_state["slot_decoder_state"] = slot_decoder_state
    if return_hidden:
        return logits, hidden_states, aux_state
    return logits


def compute_entity_auxiliary_loss(
    model,
    hidden_states,
    case,
    device,
    context_bytes=None,
    museum_aux_loss_weight=DEFAULT_MUSEUM_AUX_LOSS_WEIGHT,
    artifact_aux_loss_weight=DEFAULT_ARTIFACT_AUX_LOSS_WEIGHT,
):
    museum_aux_loss_weight = float(max(0.0, museum_aux_loss_weight))
    artifact_aux_loss_weight = float(max(0.0, artifact_aux_loss_weight))
    total_aux_loss = torch.tensor(0.0, device=device)
    metrics = {
        "museum_loss": 0.0,
        "artifact_loss": 0.0,
    }
    answer_start_index = get_answer_start_index(case, context_bytes=context_bytes)
    answer_start_hidden = hidden_states[:, answer_start_index, :]
    metadata = case.get("metadata", {})
    museum_logits, artifact_logits = get_entity_auxiliary_logits(model, answer_start_hidden)

    if museum_aux_loss_weight > 0.0 and museum_logits is not None:
        museum_target = torch.tensor([MUSEUM_NAME_TO_ID[metadata["museum"]]], dtype=torch.long, device=device)
        museum_loss = F.cross_entropy(museum_logits, museum_target)
        total_aux_loss = total_aux_loss + museum_aux_loss_weight * museum_loss
        metrics["museum_loss"] = float(museum_loss.item())

    if artifact_aux_loss_weight > 0.0 and artifact_logits is not None:
        artifact_target = torch.tensor([ARTIFACT_NAME_TO_ID[metadata["artifact"]]], dtype=torch.long, device=device)
        artifact_loss = F.cross_entropy(artifact_logits, artifact_target)
        total_aux_loss = total_aux_loss + artifact_aux_loss_weight * artifact_loss
        metrics["artifact_loss"] = float(artifact_loss.item())

    return total_aux_loss, metrics


def compute_slot_decoder_loss(
    model,
    hidden_states,
    case,
    device,
    context_bytes=None,
):
    evidence_state = None
    if evidence_supervision_is_active(model, context_bytes=context_bytes):
        evidence_state = build_evidence_state(model, hidden_states, case, context_bytes=context_bytes)
    slot_decoder_features = build_slot_decoder_features(
        model,
        hidden_states,
        case,
        context_bytes=context_bytes,
        evidence_state=evidence_state,
    )
    slot_logits = get_slot_decoder_logits(model, slot_decoder_features)
    metadata = case.get("metadata", {})

    total_slot_loss = torch.tensor(0.0, device=device)
    metrics = {}
    active_slot_count = 0
    for slot_name in ANSWER_SLOT_NAMES:
        logits = slot_logits.get(slot_name)
        if logits is None:
            metrics[f"{slot_name}_loss"] = 0.0
            continue
        target = torch.tensor(
            [SLOT_NAME_TO_ID[slot_name][metadata[slot_name]]],
            dtype=torch.long,
            device=device,
        )
        slot_loss = F.cross_entropy(logits, target)
        total_slot_loss = total_slot_loss + slot_loss
        metrics[f"{slot_name}_loss"] = float(slot_loss.item())
        active_slot_count += 1

    if active_slot_count > 0:
        total_slot_loss = total_slot_loss / active_slot_count
    return total_slot_loss, metrics


def compute_evidence_loss(
    model,
    hidden_states,
    case,
    device,
    context_bytes=None,
):
    evidence_state = build_evidence_state(model, hidden_states, case, context_bytes=context_bytes)
    if evidence_state is None:
        return torch.tensor(0.0, device=device), {
            "window_loss": 0.0,
            "window_exact_match": 0.0,
        }

    target_tensor = torch.tensor([evidence_state["target_window"]], dtype=torch.long, device=device)
    evidence_loss = F.cross_entropy(evidence_state["window_logits"], target_tensor)
    predicted_window = int(evidence_state["predicted_window"].item())
    return evidence_loss, {
        "window_loss": float(evidence_loss.item()),
        "window_exact_match": float(predicted_window == evidence_state["target_window"]),
    }


def evaluate_entity_auxiliary(model, hidden_states, case, context_bytes=None):
    answer_start_index = get_answer_start_index(case, context_bytes=context_bytes)
    answer_start_hidden = hidden_states[:, answer_start_index, :]
    metadata = case.get("metadata", {})
    metrics = {}
    museum_logits, artifact_logits = get_entity_auxiliary_logits(model, answer_start_hidden)

    if museum_logits is not None:
        predicted_museum_id = museum_logits.argmax(dim=-1).item()
        target_museum_id = MUSEUM_NAME_TO_ID[metadata["museum"]]
        metrics["museum"] = {
            "available": True,
            "predicted_label": MUSEUM_NAMES[predicted_museum_id],
            "target_label": metadata["museum"],
            "exact_match": predicted_museum_id == target_museum_id,
        }
    else:
        metrics["museum"] = {"available": False}

    if artifact_logits is not None:
        predicted_artifact_id = artifact_logits.argmax(dim=-1).item()
        target_artifact_id = ARTIFACT_NAME_TO_ID[metadata["artifact"]]
        metrics["artifact"] = {
            "available": True,
            "predicted_label": ARTIFACT_NAMES[predicted_artifact_id],
            "target_label": metadata["artifact"],
            "exact_match": predicted_artifact_id == target_artifact_id,
        }
    else:
        metrics["artifact"] = {"available": False}

    return metrics


def evaluate_evidence_decoder(model, hidden_states, case, context_bytes=None):
    evidence_state = build_evidence_state(model, hidden_states, case, context_bytes=context_bytes)
    if evidence_state is None:
        return {"available": False}
    predicted_window = int(evidence_state["predicted_window"].item())
    target_window = int(evidence_state["target_window"])
    return {
        "available": True,
        "predicted_window": predicted_window,
        "target_window": target_window,
        "exact_match": predicted_window == target_window,
        "window_distance": abs(predicted_window - target_window),
    }


def extract_slot_labels_from_window_bytes(window_bytes):
    window_text = window_bytes.decode("ascii", errors="ignore")
    extracted_slots = {}
    for slot_name, candidate_values in SLOT_VALUE_LOOKUPS.items():
        matching_values = [
            candidate_value
            for candidate_value in candidate_values
            if candidate_value in window_text
        ]
        if matching_values:
            extracted_slots[slot_name] = max(matching_values, key=len)
    return extracted_slots


def build_extract_compose_prediction_bytes(extracted_slots):
    predicted_metadata = {}
    for slot_name in ANSWER_SLOT_NAMES:
        predicted_metadata[slot_name] = extracted_slots.get(slot_name, "UNKNOWN")
    return build_answer_text_from_metadata(predicted_metadata).encode("ascii")


def evaluate_extract_then_compose(model, hidden_states, case, context_bytes=None):
    evidence_state = build_evidence_state(model, hidden_states, case, context_bytes=context_bytes)
    if evidence_state is None:
        return {"available": False}

    sample_bytes = build_curriculum_context(case, context_bytes) if context_bytes is not None else case["sample_bytes"]
    evidence_window_count = int(max(1, getattr(model, "evidence_window_count", DEFAULT_EVIDENCE_WINDOW_COUNT)))
    predicted_window = int(evidence_state["predicted_window"].item())
    window_start, window_end = get_evidence_window_bounds(
        sample_length=len(sample_bytes),
        window_index=predicted_window,
        window_count=evidence_window_count,
    )
    window_bytes = sample_bytes[window_start:window_end]
    extracted_slots = extract_slot_labels_from_window_bytes(window_bytes)
    predicted_answer_bytes = build_extract_compose_prediction_bytes(extracted_slots)
    predicted_tokens = list(predicted_answer_bytes)
    expected_tokens = list(case["expected_answer_bytes"])
    metrics = compute_sequence_metrics(predicted_tokens, expected_tokens)
    tail_metrics = compute_tail_slice_metrics(predicted_tokens, expected_tokens)
    entity_span_metrics = compute_entity_span_metrics(predicted_tokens, case)

    slot_metrics = {}
    metadata = case.get("metadata", {})
    for slot_name in ANSWER_SLOT_NAMES:
        predicted_label = extracted_slots.get(slot_name)
        slot_metrics[slot_name] = {
            "predicted_label": predicted_label,
            "target_label": metadata.get(slot_name),
            "exact_match": predicted_label == metadata.get(slot_name),
        }

    return {
        "available": True,
        "predicted_text": predicted_answer_bytes.decode("ascii", errors="replace"),
        "exact_byte_match": predicted_answer_bytes == case["expected_answer_bytes"],
        "token_acc": metrics["sequence_accuracy"],
        "extracted_slots": slot_metrics,
        "predicted_window": predicted_window,
        "window_start": window_start,
        "window_end": window_end,
        **metrics,
        **tail_metrics,
        "entity_span_metrics": entity_span_metrics,
    }


def evaluate_slot_decoder(model, hidden_states, case, context_bytes=None):
    evidence_state = None
    if evidence_supervision_is_active(model, context_bytes=context_bytes):
        evidence_state = build_evidence_state(model, hidden_states, case, context_bytes=context_bytes)
    slot_state = build_slot_decoder_state(
        model,
        hidden_states,
        case,
        context_bytes=context_bytes,
        evidence_state=evidence_state,
    )
    if slot_state is None:
        return {
            "available": False,
            "all_slots_exact_match": False,
        }

    per_slot = {}
    exact_matches = []
    for slot_name in ANSWER_SLOT_NAMES:
        prediction = slot_state["slot_predictions"][slot_name]
        exact_match = prediction["predicted_label"] == prediction["target_label"]
        exact_matches.append(exact_match)
        per_slot[slot_name] = {
            "predicted_label": prediction["predicted_label"],
            "target_label": prediction["target_label"],
            "exact_match": exact_match,
        }
    return {
        "available": True,
        "all_slots_exact_match": all(exact_matches),
        **per_slot,
    }


def greedy_generate_answer(model, case, prompt_tokens, answer_len, device, context_bytes=None):
    with torch.no_grad():
        next_logits, S_prev, bypass_kv, raw_history, hint_state, slot_decoder_state, evidence_state = prefill_generation_state(
            model=model,
            case=case,
            prompt_tokens=prompt_tokens,
            device=device,
            context_bytes=context_bytes,
        )
    generated_answer = []

    model.eval()
    with torch.no_grad():
        for step in range(answer_len):
            next_token = predict_byte_tokens(next_logits).item()
            generated_answer.append(next_token)
            if step == answer_len - 1:
                break

            token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            raw_emb = model.embedding(token_tensor)
            raw_history.append(raw_emb)
            step_input = model.build_step_context(raw_history)
            out_t, S_prev, bypass_kv = model.dsra.forward_step(
                step_input,
                S_prev=S_prev,
                kv_cache=bypass_kv,
            )
            out_t = model.norm(out_t)
            out_t = apply_entity_hint_to_step_hidden(model, out_t, hint_state, answer_token_index=step + 1)
            out_t = apply_evidence_hint_to_step_hidden(model, out_t, evidence_state, answer_token_index=step + 1)
            next_logits = model.out_proj(out_t)[:, -1, :]
            next_logits = apply_slot_decoder_bias_to_step_logits(
                model,
                next_logits,
                slot_decoder_state,
                answer_token_index=step + 1,
            )

    return generated_answer


def prefill_generation_state(model, case, prompt_tokens, device, context_bytes=None):
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    raw_prompt_emb = model.embedding(prompt_tensor)
    contextual_prompt_emb = model.build_causal_context(raw_prompt_emb)

    S_prev = None
    bypass_kv = None
    S_time_prev = None
    last_out = None
    chunk_idx = 0
    prompt_hidden_chunks = []

    for start in range(0, contextual_prompt_emb.shape[1], model.chunk_size):
        chunk = contextual_prompt_emb[:, start:start + model.chunk_size, :]
        out_chunk, S_prev, bypass_kv, S_time_prev = model.dsra(
            chunk,
            S_prev=S_prev,
            bypass_kv=bypass_kv,
            S_time_prev=S_time_prev,
            chunk_idx=chunk_idx,
        )
        prompt_hidden_chunks.append(out_chunk)
        last_out = out_chunk[:, -1:, :]
        chunk_idx += 1

    last_out = model.norm(last_out)
    prompt_hidden_states = model.norm(torch.cat(prompt_hidden_chunks, dim=1))
    hint_state = None
    if entity_hinting_is_active(model, context_bytes=context_bytes):
        hint_state = build_entity_hint_state(model, last_out[:, 0, :], case)
        last_out = apply_entity_hint_to_step_hidden(model, last_out, hint_state, answer_token_index=0)
    next_logits = model.out_proj(last_out)[:, -1, :]
    evidence_state = None
    if evidence_supervision_is_active(model, context_bytes=context_bytes):
        evidence_state = build_evidence_state(
            model,
            prompt_hidden_states,
            case,
            context_bytes=context_bytes,
        )
        last_out = apply_evidence_hint_to_step_hidden(model, last_out, evidence_state, answer_token_index=0)
        next_logits = model.out_proj(last_out)[:, -1, :]
    slot_decoder_state = None
    if slot_decoder_is_active(model, context_bytes=context_bytes):
        slot_decoder_state = build_slot_decoder_state(
            model,
            prompt_hidden_states,
            case,
            context_bytes=context_bytes,
            evidence_state=evidence_state,
        )
        next_logits = apply_slot_decoder_bias_to_step_logits(
            model,
            next_logits,
            slot_decoder_state,
            answer_token_index=0,
        )
    raw_history = deque(
        [raw_prompt_emb[:, idx:idx + 1, :] for idx in range(raw_prompt_emb.shape[1])],
        maxlen=model.local_context_size,
    )
    return next_logits, S_prev, bypass_kv, raw_history, hint_state, slot_decoder_state, evidence_state


def rollout_generation_logits(model, case, prompt_tokens, answer_tokens, device, self_feed_ratio=0.0, rng=None, context_bytes=None):
    with torch.no_grad():
        next_logits, S_prev, bypass_kv, raw_history, hint_state, slot_decoder_state, evidence_state = prefill_generation_state(
            model=model,
            case=case,
            prompt_tokens=prompt_tokens,
            device=device,
            context_bytes=context_bytes,
        )

    step_logits = []
    sampled_token_count = 0
    self_feed_ratio = float(max(0.0, min(self_feed_ratio, 1.0)))
    rng = rng or random

    for step_idx, target_token in enumerate(answer_tokens):
        step_logits.append(next_logits)
        if step_idx == len(answer_tokens) - 1:
            break

        use_predicted_token = self_feed_ratio > 0.0 and rng.random() < self_feed_ratio
        if use_predicted_token:
            input_token = predict_byte_tokens(next_logits.detach()).item()
            sampled_token_count += 1
        else:
            input_token = target_token

        token_tensor = torch.tensor([[input_token]], dtype=torch.long, device=device)
        raw_emb = model.embedding(token_tensor)
        raw_history.append(raw_emb)
        step_input = model.build_step_context(raw_history)
        out_t, S_prev, bypass_kv = model.dsra.forward_step(
            step_input,
            S_prev=S_prev,
            kv_cache=bypass_kv,
        )
        out_t = model.norm(out_t)
        out_t = apply_entity_hint_to_step_hidden(model, out_t, hint_state, answer_token_index=step_idx + 1)
        out_t = apply_evidence_hint_to_step_hidden(model, out_t, evidence_state, answer_token_index=step_idx + 1)
        next_logits = model.out_proj(out_t)[:, -1, :]
        next_logits = apply_slot_decoder_bias_to_step_logits(
            model,
            next_logits,
            slot_decoder_state,
            answer_token_index=step_idx + 1,
        )

    return torch.stack(step_logits, dim=1), sampled_token_count


def evaluate_teacher_forced(logits, Y, case):
    logits_target, targets = collect_answer_targets(logits, Y)
    predicted_tokens = predict_byte_tokens(logits_target).tolist()
    expected_tokens = list(case["expected_answer_bytes"])
    predicted_bytes = bytes(predicted_tokens)
    expected_bytes = case["expected_answer_bytes"]
    metrics = compute_sequence_metrics(predicted_tokens, expected_tokens)
    tail_metrics = compute_tail_slice_metrics(predicted_tokens, expected_tokens)
    entity_span_metrics = compute_entity_span_metrics(predicted_tokens, case)

    return {
        "predicted_tokens": predicted_tokens,
        "predicted_text": predicted_bytes.decode("utf-8", errors="replace"),
        "exact_byte_match": predicted_bytes == expected_bytes,
        "token_acc": metrics["sequence_accuracy"],
        **metrics,
        **tail_metrics,
        "entity_span_metrics": entity_span_metrics,
    }


def evaluate_generation(model, case, device, context_bytes=None):
    sample_bytes = build_curriculum_context(case, context_bytes) if context_bytes is not None else case["sample_bytes"]
    sample_tokens = list(sample_bytes)
    question_tokens = list(case["question_bytes"])
    expected_answer_tokens = list(case["expected_answer_bytes"])
    prompt_tokens = sample_tokens + [QUESTION_TOKEN_ID] + question_tokens + [ANSWER_START_TOKEN_ID]

    predicted_tokens = greedy_generate_answer(
        model=model,
        case=case,
        prompt_tokens=prompt_tokens,
        answer_len=len(expected_answer_tokens),
        device=device,
        context_bytes=context_bytes,
    )
    predicted_bytes = bytes(predicted_tokens)
    expected_bytes = case["expected_answer_bytes"]
    predicted_text = predicted_bytes.decode("utf-8", errors="replace")
    expected_text = expected_bytes.decode("utf-8", errors="replace")
    metrics = compute_sequence_metrics(predicted_tokens, expected_answer_tokens)
    tail_metrics = compute_tail_slice_metrics(predicted_tokens, expected_answer_tokens)
    entity_span_metrics = compute_entity_span_metrics(predicted_tokens, case)

    return {
        "predicted_tokens": predicted_tokens,
        "predicted_text": predicted_text,
        "expected_text": expected_text,
        "exact_byte_match": predicted_bytes == expected_bytes,
        "exact_text_match": predicted_text == expected_text,
        **metrics,
        **tail_metrics,
        "entity_span_metrics": entity_span_metrics,
    }


def evaluate_single_case(model, case, device):
    X, Y = build_training_example(case)
    X, Y = X.to(device), Y.to(device)
    with torch.no_grad():
        logits, hidden_states, _ = forward_json_retrieval(model, X, case, return_hidden=True)
    teacher_forced = evaluate_teacher_forced(logits, Y, case)
    generation = evaluate_generation(model, case, device)
    return {
        "question": case["metadata"]["question"],
        "expected_answer_text": case["metadata"]["expected_answer_text"],
        "museum": case["metadata"].get("museum"),
        "artifact": case["metadata"].get("artifact"),
        "answer_len": len(case["expected_answer_bytes"]),
        "answer_entity_spans": get_answer_entity_spans(case),
        "entity_auxiliary": evaluate_entity_auxiliary(model, hidden_states, case),
        "evidence_decoder": evaluate_evidence_decoder(model, hidden_states, case),
        "extract_then_compose": evaluate_extract_then_compose(model, hidden_states, case),
        "slot_decoder": evaluate_slot_decoder(model, hidden_states, case),
        "needle_position_pct": case["metadata"]["needle_position_pct"],
        "teacher_forced": teacher_forced,
        "generation": generation,
    }


def aggregate_case_pool_results(case_results):
    total_cases = max(1, len(case_results))
    teacher_forced_exact_matches = sum(
        int(case_result["teacher_forced"]["exact_byte_match"]) for case_result in case_results
    )
    generation_exact_matches = sum(
        int(case_result["generation"]["exact_byte_match"]) for case_result in case_results
    )
    teacher_forced_seq_acc = sum(
        case_result["teacher_forced"]["sequence_accuracy"] for case_result in case_results
    ) / total_cases
    generation_seq_acc = sum(
        case_result["generation"]["sequence_accuracy"] for case_result in case_results
    ) / total_cases
    teacher_forced_prefix = sum(
        case_result["teacher_forced"]["prefix_match_length"] for case_result in case_results
    ) / total_cases
    generation_prefix = sum(
        case_result["generation"]["prefix_match_length"] for case_result in case_results
    ) / total_cases
    aggregated = {
        "num_cases": len(case_results),
        "teacher_forced_exact_matches": teacher_forced_exact_matches,
        "teacher_forced_exact_match_rate": teacher_forced_exact_matches / total_cases,
        "teacher_forced_mean_sequence_accuracy": teacher_forced_seq_acc,
        "teacher_forced_mean_prefix_match_length": teacher_forced_prefix,
        "generation_exact_matches": generation_exact_matches,
        "generation_exact_match_rate": generation_exact_matches / total_cases,
        "generation_mean_sequence_accuracy": generation_seq_acc,
        "generation_mean_prefix_match_length": generation_prefix,
        "case_results": case_results,
    }
    for entity_name in ENTITY_SPAN_NAMES:
        aux_results = [
            case_result.get("entity_auxiliary", {}).get(entity_name)
            for case_result in case_results
            if case_result.get("entity_auxiliary", {}).get(entity_name, {}).get("available")
        ]
        if aux_results:
            aggregated[f"entity_aux_{entity_name}_accuracy"] = (
                sum(int(aux_result["exact_match"]) for aux_result in aux_results)
                / len(aux_results)
            )
        else:
            aggregated[f"entity_aux_{entity_name}_accuracy"] = None
    slot_decoder_results = [
        case_result.get("slot_decoder", {})
        for case_result in case_results
        if case_result.get("slot_decoder", {}).get("available")
    ]
    if slot_decoder_results:
        aggregated["slot_decoder_full_answer_accuracy"] = (
            sum(int(slot_result["all_slots_exact_match"]) for slot_result in slot_decoder_results)
            / len(slot_decoder_results)
        )
        for slot_name in ANSWER_SLOT_NAMES:
            aggregated[f"slot_decoder_{slot_name}_accuracy"] = (
                sum(int(slot_result[slot_name]["exact_match"]) for slot_result in slot_decoder_results)
                / len(slot_decoder_results)
            )
    else:
        aggregated["slot_decoder_full_answer_accuracy"] = None
        for slot_name in ANSWER_SLOT_NAMES:
            aggregated[f"slot_decoder_{slot_name}_accuracy"] = None
    evidence_results = [
        case_result.get("evidence_decoder", {})
        for case_result in case_results
        if case_result.get("evidence_decoder", {}).get("available")
    ]
    if evidence_results:
        aggregated["evidence_window_accuracy"] = (
            sum(int(evidence_result["exact_match"]) for evidence_result in evidence_results)
            / len(evidence_results)
        )
        aggregated["evidence_window_mean_distance"] = (
            sum(evidence_result["window_distance"] for evidence_result in evidence_results)
            / len(evidence_results)
        )
    else:
        aggregated["evidence_window_accuracy"] = None
        aggregated["evidence_window_mean_distance"] = None
    extract_compose_results = [
        case_result.get("extract_then_compose", {})
        for case_result in case_results
        if case_result.get("extract_then_compose", {}).get("available")
    ]
    if extract_compose_results:
        aggregated["extract_then_compose_exact_match_rate"] = (
            sum(int(result["exact_byte_match"]) for result in extract_compose_results)
            / len(extract_compose_results)
        )
        aggregated["extract_then_compose_mean_sequence_accuracy"] = (
            sum(result["sequence_accuracy"] for result in extract_compose_results)
            / len(extract_compose_results)
        )
        aggregated["extract_then_compose_mean_prefix_match_length"] = (
            sum(result["prefix_match_length"] for result in extract_compose_results)
            / len(extract_compose_results)
        )
        for slot_name in ANSWER_SLOT_NAMES:
            aggregated[f"extract_then_compose_{slot_name}_accuracy"] = (
                sum(int(result["extracted_slots"][slot_name]["exact_match"]) for result in extract_compose_results)
                / len(extract_compose_results)
            )
    else:
        aggregated["extract_then_compose_exact_match_rate"] = None
        aggregated["extract_then_compose_mean_sequence_accuracy"] = None
        aggregated["extract_then_compose_mean_prefix_match_length"] = None
        for slot_name in ANSWER_SLOT_NAMES:
            aggregated[f"extract_then_compose_{slot_name}_accuracy"] = None
    for mode in ("teacher_forced", "generation"):
        for entity_name in ENTITY_SPAN_NAMES:
            entity_results = [
                case_result[mode]["entity_span_metrics"][entity_name]
                for case_result in case_results
                if entity_name in case_result[mode].get("entity_span_metrics", {})
            ]
            if entity_results:
                aggregated[f"{mode}_{entity_name}_exact_match_rate"] = (
                    sum(int(entity_result["exact_match"]) for entity_result in entity_results)
                    / len(entity_results)
                )
                aggregated[f"{mode}_{entity_name}_mean_sequence_accuracy"] = (
                    sum(entity_result["sequence_accuracy"] for entity_result in entity_results)
                    / len(entity_results)
                )
                aggregated[f"{mode}_{entity_name}_mean_prefix_match_length"] = (
                    sum(entity_result["prefix_match_length"] for entity_result in entity_results)
                    / len(entity_results)
                )
            else:
                aggregated[f"{mode}_{entity_name}_exact_match_rate"] = None
                aggregated[f"{mode}_{entity_name}_mean_sequence_accuracy"] = None
                aggregated[f"{mode}_{entity_name}_mean_prefix_match_length"] = None
    return aggregated


def evaluate_case_pool(model, cases, device):
    model.eval()
    case_results = []
    for case in cases:
        case_results.append(evaluate_single_case(model, case, device))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return aggregate_case_pool_results(case_results)


def build_tail_error_analysis(pool_eval, tail_token_count=DEFAULT_TAIL_ERROR_TOKEN_COUNT, top_k=DEFAULT_TAIL_ERROR_TOP_K):
    case_results = pool_eval["case_results"]
    tail_token_count = max(1, int(tail_token_count))
    top_k = max(1, int(top_k))
    analysis = {
        "tail_token_count": tail_token_count,
        "teacher_forced": {},
        "generation": {},
    }

    for mode in ("teacher_forced", "generation"):
        failures = [case_result for case_result in case_results if not case_result[mode]["exact_byte_match"]]
        first_mismatch_indices = [
            case_result[mode]["first_mismatch_index"]
            for case_result in failures
            if case_result[mode]["first_mismatch_index"] is not None
        ]
        late_tail_failures = 0
        for case_result in failures:
            answer_len = max(1, int(case_result["answer_len"]))
            tail_start = max(0, answer_len - min(tail_token_count, answer_len))
            first_mismatch_index = case_result[mode]["first_mismatch_index"]
            if first_mismatch_index is not None and first_mismatch_index >= tail_start:
                late_tail_failures += 1

        close_miss_cases = sorted(
            failures,
            key=lambda case_result: (
                case_result[mode]["prefix_match_length"],
                case_result[mode]["tail_sequence_accuracy"],
                case_result[mode]["sequence_accuracy"],
            ),
            reverse=True,
        )[:top_k]

        analysis[mode] = {
            "num_failures": len(failures),
            "mean_first_mismatch_index": (
                sum(first_mismatch_indices) / len(first_mismatch_indices)
                if first_mismatch_indices
                else None
            ),
            "mean_tail_sequence_accuracy": (
                sum(case_result[mode]["tail_sequence_accuracy"] for case_result in case_results)
                / max(1, len(case_results))
            ),
            "tail_exact_match_rate": (
                sum(int(case_result[mode]["tail_exact_match"]) for case_result in case_results)
                / max(1, len(case_results))
            ),
            "late_tail_failure_rate": (
                late_tail_failures / max(1, len(failures))
                if failures
                else 0.0
            ),
            "close_miss_cases": [
                {
                    "question": case_result["question"],
                    "museum": case_result.get("museum"),
                    "artifact": case_result.get("artifact"),
                    "answer_len": case_result["answer_len"],
                    "prefix_match_length": case_result[mode]["prefix_match_length"],
                    "first_mismatch_index": case_result[mode]["first_mismatch_index"],
                    "sequence_accuracy": case_result[mode]["sequence_accuracy"],
                    "tail_sequence_accuracy": case_result[mode]["tail_sequence_accuracy"],
                    "museum_span_exact_match": (
                        case_result[mode].get("entity_span_metrics", {}).get("museum", {}).get("exact_match")
                    ),
                    "museum_span_sequence_accuracy": (
                        case_result[mode].get("entity_span_metrics", {}).get("museum", {}).get("sequence_accuracy")
                    ),
                    "artifact_span_exact_match": (
                        case_result[mode].get("entity_span_metrics", {}).get("artifact", {}).get("exact_match")
                    ),
                    "artifact_span_sequence_accuracy": (
                        case_result[mode].get("entity_span_metrics", {}).get("artifact", {}).get("sequence_accuracy")
                    ),
                }
                for case_result in close_miss_cases
            ],
        }

    return analysis


def build_curriculum_plan(case, total_epochs):
    active_contexts = []
    stage_specs = []
    for requested_context_bytes in CURRICULUM_CONTEXT_BYTES:
        stage_context = min(requested_context_bytes, len(case["sample_bytes"]))
        if active_contexts and stage_context == active_contexts[-1]:
            continue
        active_contexts.append(stage_context)
        stage_specs.append((requested_context_bytes, stage_context))

    stage_count = len(stage_specs)
    base_epochs = total_epochs // stage_count
    remainder = total_epochs % stage_count
    plan = []
    for stage_idx, (requested_context_bytes, context_bytes) in enumerate(stage_specs):
        stage_epochs = base_epochs + (1 if stage_idx < remainder else 0)
        if stage_epochs == 0:
            continue
        plan.append(
            {
                "name": f"{requested_context_bytes // 1024}K" if requested_context_bytes >= 1024 else str(requested_context_bytes),
                "context_bytes": context_bytes,
                "epochs": stage_epochs,
            }
        )
    return plan


def build_lr_schedule_lambda(total_steps, warmup_steps):
    warmup_steps = max(1, min(warmup_steps, total_steps))

    def schedule(step_idx):
        current_step = step_idx + 1
        if current_step <= warmup_steps:
            return current_step / warmup_steps

        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, (current_step - warmup_steps) / decay_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())

    return schedule


def build_scheduled_sampling_ratio_schedule(total_steps, max_ratio):
    total_steps = max(1, total_steps)
    max_ratio = float(max(0.0, min(max_ratio, 1.0)))

    if total_steps == 1:
        return lambda step_idx: max_ratio

    def schedule(step_idx):
        bounded_step = max(0, min(step_idx, total_steps - 1))
        progress = bounded_step / (total_steps - 1)
        return max_ratio * progress

    return schedule


def build_scheduled_sampling_inputs(X, Y, predicted_tokens, sampling_ratio, sample_mask=None):
    """
    调用方:
    - 当前文件训练循环中的 scheduled sampling 分支会调用本函数，基于 teacher-forced 预测结果构造下一次前向的 `train_X`
    - `tests/test_json_retrieval.py` 中的回归测试会直接调用本函数验证替换逻辑、设备对齐与类型对齐

    本函数调用:
    - `X.clone()` 复制输入序列，避免原地污染训练样本
    - `(Y[0] != PAD_TOKEN_ID).nonzero(...).flatten()` 收集答案 token 在标签中的有效位置
    - `predicted_tokens[:-1].to(...)` 将候选 token 对齐到 `sampled_X` 的设备与 dtype
    - `torch.rand(...)` 在未显式传入 `sample_mask` 时按采样比例生成布尔采样掩码
    - `sample_mask.to(...)` 将外部传入的掩码对齐到 `sampled_X` 的设备并转换为布尔类型

    作用:
    - 按 scheduled sampling 规则，将答案前缀位置上的一部分 teacher-forced 预测 token 回填到 `X`
    - 返回替换后的新输入张量与本次实际替换的 token 数量

    变量含义:
    - `X`: 训练输入序列，约定形状为 `(1, seq_len)`，其设备和 dtype 作为最终对齐目标
    - `Y`: 标签序列，约定形状为 `(1, seq_len)`，用来定位答案区域的有效 token
    - `predicted_tokens`: teacher-forced 阶段输出的答案 token 序列，长度应与答案 token 数一致
    - `sampling_ratio`: 采样比例，`<= 0.0` 时直接返回原始副本
    - `sample_mask`: 可选的外部布尔掩码，用于测试或固定采样行为；长度必须与答案前缀长度一致
    - `sampled_X`: `X` 的副本，承载本函数的替换结果
    - `answer_positions`: `Y[0]` 中所有非 `PAD_TOKEN_ID` 的位置
    - `prefix_positions`: 可被替换的答案前缀位置，跳过第一个答案 token 以保持自回归对齐
    - `candidate_tokens`: `predicted_tokens[:-1]` 对齐设备和 dtype 后得到的候选替换 token

    接入方式:
    - 先在训练循环中完成 `X`、`Y` 的构造，并确保二者对应同一个单样本序列
    - 使用 teacher-forced 前向结果生成 `predicted_tokens`
    - 调用本函数拿到 `train_X` 后，再将 `train_X` 送入正式训练前向
    - 若需要稳定复现实验或编写测试，可显式传入 `sample_mask`

    错误处理:
    - 当 `sampling_ratio <= 0.0`、答案 token 数不足 2 个时，直接返回 `X` 的副本和 `0`
    - 当 `predicted_tokens[:-1]` 与答案前缀长度不一致时，抛出 `ValueError`
    - 当显式传入的 `sample_mask` 长度与答案前缀长度不一致时，抛出 `ValueError`
    - 通过 `.to(device=sampled_X.device, dtype=sampled_X.dtype)` 与 `.to(device=sampled_X.device, dtype=torch.bool)` 规避跨设备和类型不一致导致的索引/赋值错误
    """
    sampled_X = X.clone()
    if sampling_ratio <= 0.0:
        return sampled_X, 0

    answer_positions = (Y[0] != PAD_TOKEN_ID).nonzero(as_tuple=False).flatten()
    if answer_positions.numel() <= 1:
        return sampled_X, 0

    prefix_positions = answer_positions[1:]
    candidate_tokens = predicted_tokens[:-1].to(device=sampled_X.device, dtype=sampled_X.dtype)
    if candidate_tokens.numel() != prefix_positions.numel():
        raise ValueError("Predicted token count does not align with answer prefix positions.")

    if sample_mask is None:
        sample_mask = torch.rand(prefix_positions.numel(), device=sampled_X.device) < sampling_ratio
    else:
        sample_mask = sample_mask.to(device=sampled_X.device, dtype=torch.bool)
        if sample_mask.numel() != prefix_positions.numel():
            raise ValueError("Sample mask count does not align with answer prefix positions.")

    if sample_mask.any():
        sampled_X[0, prefix_positions[sample_mask]] = candidate_tokens[sample_mask]

    return sampled_X, int(sample_mask.sum().item())


def summarize_search_result(result):
    return {
        "kr": result["config"]["kr"],
        "chunk_size": result["config"]["chunk_size"],
        "lr": result["config"]["lr"],
        "warmup_ratio": result["config"]["warmup_ratio"],
        "scheduled_sampling_max_ratio": result["config"]["scheduled_sampling_max_ratio"],
        "generation_exact_byte_match": result["evaluation"]["exact_byte_match"],
        "generation_sequence_accuracy": result["evaluation"]["sequence_accuracy"],
        "generation_prefix_match_length": result["evaluation"]["prefix_match_length"],
        "teacher_forced_exact_byte_match": result["teacher_forced_evaluation"]["exact_byte_match"],
        "teacher_forced_sequence_accuracy": result["teacher_forced_evaluation"]["sequence_accuracy"],
        "teacher_forced_prefix_match_length": result["teacher_forced_evaluation"]["prefix_match_length"],
    }


def score_search_result(result):
    return (
        int(result["evaluation"]["exact_byte_match"]),
        int(result["teacher_forced_evaluation"]["exact_byte_match"]),
        result["evaluation"]["prefix_match_length"],
        result["evaluation"]["sequence_accuracy"],
        result["teacher_forced_evaluation"]["prefix_match_length"],
        result["teacher_forced_evaluation"]["sequence_accuracy"],
    )


def summarize_generalization_search_result(result):
    validation = result["validation_pool_evaluation"]
    test = result["test_pool_evaluation"]
    return {
        "epochs": result["config"]["epochs"],
        "kr": result["config"]["kr"],
        "chunk_size": result["config"]["chunk_size"],
        "lr": result["config"]["lr"],
        "warmup_ratio": result["config"]["warmup_ratio"],
        "scheduled_sampling_max_ratio": result["config"]["scheduled_sampling_max_ratio"],
        "validation_generation_exact_match_rate": validation["generation_exact_match_rate"],
        "validation_generation_mean_sequence_accuracy": validation["generation_mean_sequence_accuracy"],
        "validation_generation_mean_prefix_match_length": validation["generation_mean_prefix_match_length"],
        "validation_teacher_forced_exact_match_rate": validation["teacher_forced_exact_match_rate"],
        "validation_teacher_forced_mean_sequence_accuracy": validation["teacher_forced_mean_sequence_accuracy"],
        "validation_teacher_forced_mean_prefix_match_length": validation["teacher_forced_mean_prefix_match_length"],
        "test_generation_exact_match_rate": test["generation_exact_match_rate"],
        "test_generation_mean_sequence_accuracy": test["generation_mean_sequence_accuracy"],
        "test_generation_mean_prefix_match_length": test["generation_mean_prefix_match_length"],
        "test_teacher_forced_exact_match_rate": test["teacher_forced_exact_match_rate"],
        "test_teacher_forced_mean_sequence_accuracy": test["teacher_forced_mean_sequence_accuracy"],
        "test_teacher_forced_mean_prefix_match_length": test["teacher_forced_mean_prefix_match_length"],
    }


def score_generalization_result(result, score_mode=DEFAULT_GENERALIZATION_SCORE_MODE):
    if score_mode not in {"generation", "teacher_forced", "balanced"}:
        raise ValueError(f"Unsupported score_mode: {score_mode}")
    validation = result["validation_pool_evaluation"]
    test = result["test_pool_evaluation"]
    if score_mode == "teacher_forced":
        return (
            validation["teacher_forced_exact_match_rate"],
            validation["teacher_forced_mean_sequence_accuracy"],
            validation["teacher_forced_mean_prefix_match_length"],
            test["teacher_forced_exact_match_rate"],
            test["teacher_forced_mean_sequence_accuracy"],
            test["teacher_forced_mean_prefix_match_length"],
            validation["generation_exact_match_rate"],
            validation["generation_mean_sequence_accuracy"],
            test["generation_exact_match_rate"],
            test["generation_mean_sequence_accuracy"],
        )
    if score_mode == "balanced":
        return (
            validation["teacher_forced_exact_match_rate"] + validation["generation_exact_match_rate"],
            validation["teacher_forced_mean_sequence_accuracy"] + validation["generation_mean_sequence_accuracy"],
            validation["teacher_forced_mean_prefix_match_length"] + validation["generation_mean_prefix_match_length"],
            test["teacher_forced_exact_match_rate"] + test["generation_exact_match_rate"],
            test["teacher_forced_mean_sequence_accuracy"] + test["generation_mean_sequence_accuracy"],
            test["teacher_forced_mean_prefix_match_length"] + test["generation_mean_prefix_match_length"],
        )
    return (
        validation["generation_exact_match_rate"],
        validation["teacher_forced_exact_match_rate"],
        validation["generation_mean_sequence_accuracy"],
        validation["teacher_forced_mean_sequence_accuracy"],
        validation["generation_mean_prefix_match_length"],
        validation["teacher_forced_mean_prefix_match_length"],
        test["generation_exact_match_rate"],
        test["teacher_forced_exact_match_rate"],
        test["generation_mean_sequence_accuracy"],
        test["teacher_forced_mean_sequence_accuracy"],
    )


def score_generalization_summary(summary, score_mode=DEFAULT_GENERALIZATION_SCORE_MODE):
    if score_mode not in {"generation", "teacher_forced", "balanced"}:
        raise ValueError(f"Unsupported score_mode: {score_mode}")
    if score_mode == "teacher_forced":
        return (
            summary["validation_teacher_forced_exact_match_rate"],
            summary["validation_teacher_forced_mean_sequence_accuracy"],
            summary["validation_teacher_forced_mean_prefix_match_length"],
            summary["test_teacher_forced_exact_match_rate"],
            summary["test_teacher_forced_mean_sequence_accuracy"],
            summary["test_teacher_forced_mean_prefix_match_length"],
            summary["validation_generation_exact_match_rate"],
            summary["validation_generation_mean_sequence_accuracy"],
            summary["test_generation_exact_match_rate"],
            summary["test_generation_mean_sequence_accuracy"],
        )
    if score_mode == "balanced":
        return (
            summary["validation_teacher_forced_exact_match_rate"] + summary["validation_generation_exact_match_rate"],
            summary["validation_teacher_forced_mean_sequence_accuracy"] + summary["validation_generation_mean_sequence_accuracy"],
            summary["validation_teacher_forced_mean_prefix_match_length"] + summary["validation_generation_mean_prefix_match_length"],
            summary["test_teacher_forced_exact_match_rate"] + summary["test_generation_exact_match_rate"],
            summary["test_teacher_forced_mean_sequence_accuracy"] + summary["test_generation_mean_sequence_accuracy"],
            summary["test_teacher_forced_mean_prefix_match_length"] + summary["test_generation_mean_prefix_match_length"],
        )
    return (
        summary["validation_generation_exact_match_rate"],
        summary["validation_teacher_forced_exact_match_rate"],
        summary["validation_generation_mean_sequence_accuracy"],
        summary["validation_teacher_forced_mean_sequence_accuracy"],
        summary["validation_generation_mean_prefix_match_length"],
        summary["validation_teacher_forced_mean_prefix_match_length"],
        summary["test_generation_exact_match_rate"],
        summary["test_teacher_forced_exact_match_rate"],
        summary["test_generation_mean_sequence_accuracy"],
        summary["test_teacher_forced_mean_sequence_accuracy"],
    )


def select_case_batch(cases, sampler, batch_size):
    if not cases:
        raise ValueError("cases must contain at least one case.")
    batch_size = max(1, int(batch_size))
    return [sampler.choice(cases) for _ in range(batch_size)]


def format_optional_percent(value):
    if value is None:
        return "N/A"
    return f"{value*100:.2f}%"


def format_optional_float(value, decimals=2):
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def run_target_case_polish(
    model,
    case,
    device,
    eval_interval,
    epochs,
    lr,
    history,
    history_epoch_offset,
    museum_span_loss_weight=DEFAULT_MUSEUM_SPAN_LOSS_WEIGHT,
    artifact_span_loss_weight=DEFAULT_ARTIFACT_SPAN_LOSS_WEIGHT,
):
    if epochs <= 0:
        return False, history_epoch_offset

    X, Y = build_training_example(case)
    X, Y = X.to(device), Y.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    completed_steps = 0

    print(
        f"\n--- Final Polish | full_context_bytes={len(case['sample_bytes'])} "
        f"| epochs={epochs} | lr={lr:.2e} ---"
    )

    for polish_step in range(1, epochs + 1):
        completed_steps = polish_step
        model.train()
        optimizer.zero_grad()
        logits = forward_json_retrieval(model, X, case)
        logits_target, targets = collect_answer_targets(logits, Y)
        loss = compute_weighted_answer_loss(
            logits_target,
            targets,
            case=case,
            museum_span_loss_weight=museum_span_loss_weight,
            artifact_span_loss_weight=artifact_span_loss_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        del logits, logits_target, targets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        is_log_step = (
            polish_step == 1
            or polish_step % eval_interval == 0
            or polish_step == epochs
        )
        if not is_log_step:
            continue

        with torch.no_grad():
            eval_logits = forward_json_retrieval(model, X, case)
            teacher_forced = evaluate_teacher_forced(eval_logits, Y, case)
            eval_logits_target, eval_targets = collect_answer_targets(eval_logits, Y)
            train_preds = predict_byte_tokens(eval_logits_target)
            train_acc = (train_preds == eval_targets).float().mean().item()

        absolute_epoch = history_epoch_offset + polish_step
        history.append(
            {
                "epoch": absolute_epoch,
                "stage": "polish",
                "context_bytes": len(case["sample_bytes"]),
                "loss": float(loss.item()),
                "train_token_acc": float(train_acc),
                "teacher_forced_exact_match": teacher_forced["exact_byte_match"],
                "teacher_forced_sequence_accuracy": teacher_forced["sequence_accuracy"],
                "teacher_forced_prefix_match_length": teacher_forced["prefix_match_length"],
                "first_mismatch_index": teacher_forced["first_mismatch_index"],
                "lr": float(lr),
                "scheduled_sampling_ratio": 0.0,
                "scheduled_sampling_tokens": 0,
            }
        )
        print(
            f"Polish {polish_step:4d} | Loss: {loss.item():.4f} "
            f"| Train Token Acc: {train_acc*100:5.1f}% "
            f"| Seq Acc: {teacher_forced['sequence_accuracy']*100:5.1f}% "
            f"| Prefix Match: {teacher_forced['prefix_match_length']:3d} "
            f"| Teacher Forced Exact Match: {teacher_forced['exact_byte_match']}"
        )

        del eval_logits, eval_logits_target, eval_targets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if teacher_forced["exact_byte_match"]:
            return True, history_epoch_offset + completed_steps

    return False, history_epoch_offset + completed_steps


def run_generation_polish(
    model,
    case,
    device,
    eval_interval,
    epochs,
    lr,
    history,
    history_epoch_offset,
    max_self_feed_ratio,
    rollout_loss_weight,
    teacher_forced_loss_weight,
    polish_cases=None,
    generation_polish_batch_size=DEFAULT_GENERATION_POLISH_BATCH_SIZE,
    monitor_cases=None,
    generation_polish_seed=0,
    museum_span_loss_weight=DEFAULT_MUSEUM_SPAN_LOSS_WEIGHT,
    artifact_span_loss_weight=DEFAULT_ARTIFACT_SPAN_LOSS_WEIGHT,
):
    if epochs <= 0:
        return False, history_epoch_offset

    polish_cases = list(polish_cases) if polish_cases is not None else [case]
    if not polish_cases:
        raise ValueError("polish_cases must contain at least one case.")
    monitor_cases = list(monitor_cases) if monitor_cases is not None else [case]
    if not monitor_cases:
        raise ValueError("monitor_cases must contain at least one case.")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    completed_steps = 0
    max_self_feed_ratio = float(max(0.0, min(max_self_feed_ratio, 1.0)))
    rollout_loss_weight = float(max(0.0, rollout_loss_weight))
    teacher_forced_loss_weight = float(max(0.0, teacher_forced_loss_weight))
    generation_polish_batch_size = max(1, int(generation_polish_batch_size))
    generation_exact_achieved = False
    repair_self_feed_ratio = min(max_self_feed_ratio, 0.2)
    repair_lr = lr * 0.2
    polish_sampler = random.Random(generation_polish_seed)
    monitor_mode = "reference_case" if len(monitor_cases) == 1 else f"case_pool({len(monitor_cases)})"

    print(
        f"\n--- Generation Polish | polish_cases={len(polish_cases)} | monitor={monitor_mode} "
        f"| epochs={epochs} | lr={lr:.2e} | max_self_feed_ratio={max_self_feed_ratio:.2f} "
        f"| batch_size={generation_polish_batch_size} "
        f"| rollout_loss_weight={rollout_loss_weight:.2f} "
        f"| teacher_forced_loss_weight={teacher_forced_loss_weight:.2f} ---"
    )

    for polish_step in range(1, epochs + 1):
        completed_steps = polish_step
        model.train()
        optimizer.zero_grad()
        current_self_feed_ratio = max_self_feed_ratio
        if epochs > 1:
            current_self_feed_ratio = max_self_feed_ratio * ((polish_step - 1) / (epochs - 1))
        if generation_exact_achieved:
            current_self_feed_ratio = repair_self_feed_ratio

        current_teacher_forced_weight = teacher_forced_loss_weight if generation_exact_achieved else 0.0

        sampled_cases = select_case_batch(
            polish_cases,
            sampler=polish_sampler,
            batch_size=generation_polish_batch_size,
        )
        sampled_token_count = 0
        rollout_loss_value = 0.0
        teacher_forced_loss_value = 0.0
        loss_value = 0.0

        for sampled_case in sampled_cases:
            prompt_tokens, answer_tokens = build_generation_prompt(sampled_case)
            answer_target = torch.tensor([answer_tokens], dtype=torch.long, device=device)
            full_X, full_Y = build_training_example(sampled_case)
            full_X, full_Y = full_X.to(device), full_Y.to(device)

            rollout_logits, case_sampled_token_count = rollout_generation_logits(
                model=model,
                case=sampled_case,
                prompt_tokens=prompt_tokens,
                answer_tokens=answer_tokens,
                device=device,
                self_feed_ratio=current_self_feed_ratio,
                rng=polish_sampler,
            )
            rollout_loss = compute_weighted_answer_loss(
                rollout_logits.reshape(-1, rollout_logits.shape[-1]),
                answer_target.reshape(-1),
                case=sampled_case,
                museum_span_loss_weight=museum_span_loss_weight,
                artifact_span_loss_weight=artifact_span_loss_weight,
            )
            teacher_forced_loss = torch.tensor(0.0, device=device)
            teacher_forced_logits = None
            teacher_forced_logits_target = None
            teacher_forced_targets = None
            if current_teacher_forced_weight > 0.0:
                teacher_forced_logits = forward_json_retrieval(model, full_X, sampled_case)
                teacher_forced_logits_target, teacher_forced_targets = collect_answer_targets(
                    teacher_forced_logits,
                    full_Y,
                )
                teacher_forced_loss = compute_weighted_answer_loss(
                    teacher_forced_logits_target,
                    teacher_forced_targets,
                    case=sampled_case,
                    museum_span_loss_weight=museum_span_loss_weight,
                    artifact_span_loss_weight=artifact_span_loss_weight,
                )

            case_loss = (
                rollout_loss_weight * rollout_loss
                + current_teacher_forced_weight * teacher_forced_loss
            )
            (case_loss / len(sampled_cases)).backward()
            sampled_token_count += case_sampled_token_count
            rollout_loss_value += float(rollout_loss.item())
            teacher_forced_loss_value += float(teacher_forced_loss.item())
            loss_value += float(case_loss.item())

            del (
                rollout_logits,
                teacher_forced_logits,
                teacher_forced_logits_target,
                teacher_forced_targets,
                full_X,
                full_Y,
                answer_target,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        is_log_step = (
            polish_step == 1
            or polish_step % eval_interval == 0
            or polish_step == epochs
        )
        if not is_log_step:
            continue

        with torch.no_grad():
            if len(monitor_cases) == 1:
                monitor_case = monitor_cases[0]
                monitor_X, monitor_Y = build_training_example(monitor_case)
                monitor_X, monitor_Y = monitor_X.to(device), monitor_Y.to(device)
                full_logits = forward_json_retrieval(model, monitor_X, monitor_case)
                teacher_forced = evaluate_teacher_forced(full_logits, monitor_Y, monitor_case)
                generation_eval = evaluate_generation(model, monitor_case, device)
                monitor_generation_exact = generation_eval["exact_byte_match"]
                monitor_teacher_forced_exact = teacher_forced["exact_byte_match"]
                monitor_generation_seq_acc = generation_eval["sequence_accuracy"]
                monitor_teacher_forced_seq_acc = teacher_forced["sequence_accuracy"]
                monitor_generation_prefix = generation_eval["prefix_match_length"]
                monitor_teacher_forced_prefix = teacher_forced["prefix_match_length"]
            else:
                pool_eval = evaluate_case_pool(model, monitor_cases, device)
                teacher_forced = {
                    "exact_byte_match": pool_eval["teacher_forced_exact_match_rate"] >= 1.0,
                    "sequence_accuracy": pool_eval["teacher_forced_mean_sequence_accuracy"],
                    "prefix_match_length": pool_eval["teacher_forced_mean_prefix_match_length"],
                    "first_mismatch_index": None,
                }
                generation_eval = {
                    "exact_byte_match": pool_eval["generation_exact_match_rate"] >= 1.0,
                    "sequence_accuracy": pool_eval["generation_mean_sequence_accuracy"],
                    "prefix_match_length": pool_eval["generation_mean_prefix_match_length"],
                    "first_mismatch_index": None,
                }
                monitor_generation_exact = generation_eval["exact_byte_match"]
                monitor_teacher_forced_exact = teacher_forced["exact_byte_match"]
                monitor_generation_seq_acc = generation_eval["sequence_accuracy"]
                monitor_teacher_forced_seq_acc = teacher_forced["sequence_accuracy"]
                monitor_generation_prefix = generation_eval["prefix_match_length"]
                monitor_teacher_forced_prefix = teacher_forced["prefix_match_length"]

        absolute_epoch = history_epoch_offset + polish_step
        history.append(
            {
                "epoch": absolute_epoch,
                "stage": "generation_polish_pool" if len(polish_cases) > 1 else "generation_polish",
                "context_bytes": None if len(polish_cases) > 1 else len(case["sample_bytes"]),
                "loss": loss_value / len(sampled_cases),
                "train_token_acc": monitor_generation_seq_acc,
                "teacher_forced_exact_match": monitor_teacher_forced_exact,
                "teacher_forced_sequence_accuracy": monitor_teacher_forced_seq_acc,
                "teacher_forced_prefix_match_length": monitor_teacher_forced_prefix,
                "first_mismatch_index": generation_eval["first_mismatch_index"],
                "lr": float(optimizer.param_groups[0]["lr"]),
                "scheduled_sampling_ratio": float(current_self_feed_ratio),
                "scheduled_sampling_tokens": int(sampled_token_count),
                "generation_exact_match": monitor_generation_exact,
                "generation_sequence_accuracy": monitor_generation_seq_acc,
                "generation_prefix_match_length": monitor_generation_prefix,
                "generation_polish_batch_size": len(sampled_cases),
                "generation_polish_monitor_cases": len(monitor_cases),
                "rollout_loss": rollout_loss_value / len(sampled_cases),
                "teacher_forced_loss": teacher_forced_loss_value / len(sampled_cases),
            }
        )
        print(
            f"Gen Polish {polish_step:4d} | Loss: {loss_value / len(sampled_cases):.4f} "
            f"| Monitor TF Exact: {monitor_teacher_forced_exact} "
            f"| Monitor Gen Seq Acc: {monitor_generation_seq_acc*100:5.1f}% "
            f"| Monitor Gen Prefix: {monitor_generation_prefix:5.1f} "
            f"| Monitor Gen Exact: {monitor_generation_exact} "
            f"| Self Feed: {current_self_feed_ratio:.2f}"
        )

        if len(monitor_cases) == 1:
            del full_logits, monitor_X, monitor_Y
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if monitor_generation_exact and not generation_exact_achieved:
            generation_exact_achieved = True
            for param_group in optimizer.param_groups:
                param_group["lr"] = repair_lr

        if monitor_generation_exact and monitor_teacher_forced_exact:
            return True, history_epoch_offset + completed_steps

    return False, history_epoch_offset + completed_steps


def save_json_retrieval_reports(case, config, history, teacher_forced_eval, generation_eval, search_results, reports_dir):
    reports_dir = ensure_reports_dir(reports_dir)
    payload = {
        "input_file": str(case["input_file"]),
        "metadata_file": str(case["metadata_file"]),
        "question": case["metadata"]["question"],
        "expected_answer_text": case["metadata"]["expected_answer_text"],
        "insert_position_byte_index": case["metadata"]["insert_position_byte_index"],
        "needle_position_pct": case["metadata"]["needle_position_pct"],
        "config": config,
        "history": history,
        "teacher_forced_evaluation": teacher_forced_eval,
        "generation_evaluation": generation_eval,
        "search_results": search_results,
    }
    write_json(reports_dir / "json_retrieval_report.json", payload)

    lines = [
        "# JSON Retrieval Report",
        "",
        f"- Question: `{case['metadata']['question']}`",
        f"- Input File: `{case['input_file'].name}`",
        f"- Metadata File: `{case['metadata_file'].name}`",
        f"- Sequence Bytes: `{len(case['sample_bytes'])}`",
        f"- Expected Answer Bytes: `{len(case['expected_answer_bytes'])}`",
        f"- Insert Position: `{case['metadata']['insert_position_byte_index']}`",
        f"- Curriculum: `{config['curriculum_labels']}`",
        f"- Search Trials: `{len(search_results)}`",
        "",
        "## Expected Answer",
        case["metadata"]["expected_answer_text"],
        "",
        "## Teacher-Forced Evaluation",
        f"- Exact Byte Match: `{teacher_forced_eval['exact_byte_match']}`",
        f"- Sequence Accuracy: `{teacher_forced_eval['sequence_accuracy']*100:.2f}%`",
        f"- Prefix Match Length: `{teacher_forced_eval['prefix_match_length']}`",
        f"- First Mismatch Index: `{teacher_forced_eval['first_mismatch_index']}`",
        f"- First Mismatch Expected Byte: `{teacher_forced_eval['first_mismatch_expected_byte']}`",
        f"- First Mismatch Predicted Byte: `{teacher_forced_eval['first_mismatch_predicted_byte']}`",
        "",
        "## Teacher-Forced Entity Spans",
        f"- Museum Exact Match: `{teacher_forced_eval['entity_span_metrics'].get('museum', {}).get('exact_match')}`",
        f"- Museum Sequence Accuracy: `{format_optional_percent(teacher_forced_eval['entity_span_metrics'].get('museum', {}).get('sequence_accuracy'))}`",
        f"- Museum Prefix Match Length: `{teacher_forced_eval['entity_span_metrics'].get('museum', {}).get('prefix_match_length', 'N/A')}`",
        f"- Artifact Exact Match: `{teacher_forced_eval['entity_span_metrics'].get('artifact', {}).get('exact_match')}`",
        f"- Artifact Sequence Accuracy: `{format_optional_percent(teacher_forced_eval['entity_span_metrics'].get('artifact', {}).get('sequence_accuracy'))}`",
        f"- Artifact Prefix Match Length: `{teacher_forced_eval['entity_span_metrics'].get('artifact', {}).get('prefix_match_length', 'N/A')}`",
        "",
        "## Generation Evaluation",
        f"- Exact Byte Match: `{generation_eval['exact_byte_match']}`",
        f"- Exact Text Match: `{generation_eval['exact_text_match']}`",
        f"- Sequence Accuracy: `{generation_eval['sequence_accuracy']*100:.2f}%`",
        f"- Prefix Match Length: `{generation_eval['prefix_match_length']}`",
        f"- First Mismatch Index: `{generation_eval['first_mismatch_index']}`",
        f"- First Mismatch Expected Byte: `{generation_eval['first_mismatch_expected_byte']}`",
        f"- First Mismatch Predicted Byte: `{generation_eval['first_mismatch_predicted_byte']}`",
        "",
        "## Generation Entity Spans",
        f"- Museum Exact Match: `{generation_eval['entity_span_metrics'].get('museum', {}).get('exact_match')}`",
        f"- Museum Sequence Accuracy: `{format_optional_percent(generation_eval['entity_span_metrics'].get('museum', {}).get('sequence_accuracy'))}`",
        f"- Museum Prefix Match Length: `{generation_eval['entity_span_metrics'].get('museum', {}).get('prefix_match_length', 'N/A')}`",
        f"- Artifact Exact Match: `{generation_eval['entity_span_metrics'].get('artifact', {}).get('exact_match')}`",
        f"- Artifact Sequence Accuracy: `{format_optional_percent(generation_eval['entity_span_metrics'].get('artifact', {}).get('sequence_accuracy'))}`",
        f"- Artifact Prefix Match Length: `{generation_eval['entity_span_metrics'].get('artifact', {}).get('prefix_match_length', 'N/A')}`",
        "",
        "## Predicted Answer",
        generation_eval["predicted_text"],
        "",
        "## Training Config",
        f"- Device: `{config['device']}`",
        f"- Model Type: `{config['model_type']}`",
        f"- Epochs: `{config['epochs']}`",
        f"- Eval Interval: `{config['eval_interval']}`",
        f"- Dim: `{config['dim']}`",
        f"- K: `{config['K']}`",
        f"- kr: `{config['kr']}`",
        f"- Chunk Size: `{config['chunk_size']}`",
        f"- Learning Rate: `{config['lr']}`",
        f"- Warmup Ratio: `{config['warmup_ratio']}`",
        f"- Local Context Mode: `{config['local_context_mode']}`",
        f"- Local Context Size: `{config['local_context_size']}`",
        f"- Scheduled Sampling Max Ratio: `{config['scheduled_sampling_max_ratio']}`",
        f"- Target Case Sampling Ratio: `{config['target_case_sampling_ratio']}`",
        f"- Training Mode: `{config['training_mode']}`",
        f"- Train Dataset Size: `{config['train_dataset_size']}`",
        f"- Train Dataset Seed: `{config['train_dataset_seed']}`",
        f"- Final Polish Epochs: `{config['final_polish_epochs']}`",
        f"- Final Polish LR: `{config['final_polish_lr']}`",
        f"- Final Generation Polish Epochs: `{config['final_generation_polish_epochs']}`",
        f"- Final Generation Polish LR: `{config['final_generation_polish_lr']}`",
        f"- Generation Polish Max Self Feed Ratio: `{config['generation_polish_max_self_feed_ratio']}`",
        f"- Generation Polish Rollout Loss Weight: `{config['generation_polish_rollout_loss_weight']}`",
        f"- Generation Polish Teacher Forced Loss Weight: `{config['generation_polish_teacher_forced_loss_weight']}`",
        f"- Generation Polish Batch Size: `{config['generation_polish_batch_size']}`",
        f"- Generation Polish Monitor Case Count: `{config['generation_polish_monitor_case_count']}`",
        f"- Museum Span Loss Weight: `{config['museum_span_loss_weight']}`",
        f"- Artifact Span Loss Weight: `{config['artifact_span_loss_weight']}`",
        f"- Entity Span Loss Min Context Bytes: `{config['entity_span_loss_min_context_bytes']}`",
        f"- Museum Hint Injection Weight: `{config['museum_hint_injection_weight']}`",
        f"- Artifact Hint Injection Weight: `{config['artifact_hint_injection_weight']}`",
        f"- Entity Hint Injection Min Context Bytes: `{config['entity_hint_injection_min_context_bytes']}`",
        f"- Entity Hint Uses Gold Labels During Training: `{config['entity_hint_use_gold_labels_during_training']}`",
        f"- Slot Decoder Loss Weight: `{config['slot_decoder_loss_weight']}`",
        f"- Slot Decoder Logit Bias: `{config['slot_decoder_logit_bias']}`",
        f"- Slot Decoder Min Context Bytes: `{config['slot_decoder_min_context_bytes']}`",
        f"- Evidence Window Count: `{config['evidence_window_count']}`",
        f"- Evidence Loss Weight: `{config['evidence_loss_weight']}`",
        f"- Evidence Hint Weight: `{config['evidence_hint_weight']}`",
        f"- Evidence Min Context Bytes: `{config['evidence_min_context_bytes']}`",
    ]
    if search_results:
        lines.extend(
            [
                "",
                "## Search Summary",
            ]
        )
        for rank, result in enumerate(search_results, start=1):
            lines.append(
                f"- Trial {rank}: kr={result['kr']}, chunk_size={result['chunk_size']}, lr={result['lr']}, "
                f"warmup_ratio={result['warmup_ratio']}, "
                f"scheduled_sampling_max_ratio={result['scheduled_sampling_max_ratio']}, "
                f"gen_seq_acc={result['generation_sequence_accuracy']*100:.2f}%, "
                f"gen_prefix={result['generation_prefix_match_length']}, "
                f"teacher_seq_acc={result['teacher_forced_sequence_accuracy']*100:.2f}%"
            )
    write_markdown(reports_dir / "json_retrieval_report.md", lines)


def save_json_retrieval_generalization_reports(
    config,
    history,
    validation_eval,
    test_eval,
    search_results,
    validation_tail_error_analysis,
    test_tail_error_analysis,
    reports_dir,
):
    reports_dir = ensure_reports_dir(reports_dir)
    payload = {
        "config": config,
        "history": history,
        "validation_pool_evaluation": validation_eval,
        "test_pool_evaluation": test_eval,
        "validation_tail_error_analysis": validation_tail_error_analysis,
        "test_tail_error_analysis": test_tail_error_analysis,
        "search_results": search_results,
    }
    write_json(reports_dir / "json_retrieval_generalization_report.json", payload)

    lines = [
        "# JSON Retrieval Generalization Report",
        "",
        "## Validation Pool",
        f"- Cases: `{validation_eval['num_cases']}`",
        f"- Generation Exact Match Rate: `{validation_eval['generation_exact_match_rate']*100:.2f}%`",
        f"- Generation Mean Sequence Accuracy: `{validation_eval['generation_mean_sequence_accuracy']*100:.2f}%`",
        f"- Generation Mean Prefix Match Length: `{validation_eval['generation_mean_prefix_match_length']:.2f}`",
        f"- Teacher-Forced Exact Match Rate: `{validation_eval['teacher_forced_exact_match_rate']*100:.2f}%`",
        f"- Teacher-Forced Mean Sequence Accuracy: `{validation_eval['teacher_forced_mean_sequence_accuracy']*100:.2f}%`",
        f"- Teacher-Forced Mean Prefix Match Length: `{validation_eval['teacher_forced_mean_prefix_match_length']:.2f}`",
        "",
        "## Validation Entity Span Analysis",
        f"- Teacher-Forced Museum Exact Match Rate: `{format_optional_percent(validation_eval.get('teacher_forced_museum_exact_match_rate'))}`",
        f"- Teacher-Forced Museum Mean Sequence Accuracy: `{format_optional_percent(validation_eval.get('teacher_forced_museum_mean_sequence_accuracy'))}`",
        f"- Teacher-Forced Museum Mean Prefix Match Length: `{format_optional_float(validation_eval.get('teacher_forced_museum_mean_prefix_match_length'))}`",
        f"- Teacher-Forced Artifact Exact Match Rate: `{format_optional_percent(validation_eval.get('teacher_forced_artifact_exact_match_rate'))}`",
        f"- Teacher-Forced Artifact Mean Sequence Accuracy: `{format_optional_percent(validation_eval.get('teacher_forced_artifact_mean_sequence_accuracy'))}`",
        f"- Teacher-Forced Artifact Mean Prefix Match Length: `{format_optional_float(validation_eval.get('teacher_forced_artifact_mean_prefix_match_length'))}`",
        f"- Generation Museum Exact Match Rate: `{format_optional_percent(validation_eval.get('generation_museum_exact_match_rate'))}`",
        f"- Generation Museum Mean Sequence Accuracy: `{format_optional_percent(validation_eval.get('generation_museum_mean_sequence_accuracy'))}`",
        f"- Generation Museum Mean Prefix Match Length: `{format_optional_float(validation_eval.get('generation_museum_mean_prefix_match_length'))}`",
        f"- Generation Artifact Exact Match Rate: `{format_optional_percent(validation_eval.get('generation_artifact_exact_match_rate'))}`",
        f"- Generation Artifact Mean Sequence Accuracy: `{format_optional_percent(validation_eval.get('generation_artifact_mean_sequence_accuracy'))}`",
        f"- Generation Artifact Mean Prefix Match Length: `{format_optional_float(validation_eval.get('generation_artifact_mean_prefix_match_length'))}`",
        "",
        "## Validation Entity Auxiliary",
        f"- Museum Auxiliary Accuracy: `{format_optional_percent(validation_eval.get('entity_aux_museum_accuracy'))}`",
        f"- Artifact Auxiliary Accuracy: `{format_optional_percent(validation_eval.get('entity_aux_artifact_accuracy'))}`",
        "",
        "## Validation Slot Decoder",
        f"- Full Answer Accuracy: `{format_optional_percent(validation_eval.get('slot_decoder_full_answer_accuracy'))}`",
        f"- Museum Accuracy: `{format_optional_percent(validation_eval.get('slot_decoder_museum_accuracy'))}`",
        f"- Artifact Accuracy: `{format_optional_percent(validation_eval.get('slot_decoder_artifact_accuracy'))}`",
        f"- Artist Accuracy: `{format_optional_percent(validation_eval.get('slot_decoder_artist_accuracy'))}`",
        f"- Dynasty Accuracy: `{format_optional_percent(validation_eval.get('slot_decoder_dynasty_accuracy'))}`",
        "",
        "## Validation Evidence Decoder",
        f"- Window Accuracy: `{format_optional_percent(validation_eval.get('evidence_window_accuracy'))}`",
        f"- Mean Window Distance: `{format_optional_float(validation_eval.get('evidence_window_mean_distance'))}`",
        "",
        "## Validation Extract-Then-Compose",
        f"- Exact Match Rate: `{format_optional_percent(validation_eval.get('extract_then_compose_exact_match_rate'))}`",
        f"- Mean Sequence Accuracy: `{format_optional_percent(validation_eval.get('extract_then_compose_mean_sequence_accuracy'))}`",
        f"- Mean Prefix Match Length: `{format_optional_float(validation_eval.get('extract_then_compose_mean_prefix_match_length'))}`",
        f"- Museum Accuracy: `{format_optional_percent(validation_eval.get('extract_then_compose_museum_accuracy'))}`",
        f"- Artifact Accuracy: `{format_optional_percent(validation_eval.get('extract_then_compose_artifact_accuracy'))}`",
        f"- Artist Accuracy: `{format_optional_percent(validation_eval.get('extract_then_compose_artist_accuracy'))}`",
        f"- Dynasty Accuracy: `{format_optional_percent(validation_eval.get('extract_then_compose_dynasty_accuracy'))}`",
        "",
        "## Test Pool",
        f"- Cases: `{test_eval['num_cases']}`",
        f"- Generation Exact Match Rate: `{test_eval['generation_exact_match_rate']*100:.2f}%`",
        f"- Generation Mean Sequence Accuracy: `{test_eval['generation_mean_sequence_accuracy']*100:.2f}%`",
        f"- Generation Mean Prefix Match Length: `{test_eval['generation_mean_prefix_match_length']:.2f}`",
        f"- Teacher-Forced Exact Match Rate: `{test_eval['teacher_forced_exact_match_rate']*100:.2f}%`",
        f"- Teacher-Forced Mean Sequence Accuracy: `{test_eval['teacher_forced_mean_sequence_accuracy']*100:.2f}%`",
        f"- Teacher-Forced Mean Prefix Match Length: `{test_eval['teacher_forced_mean_prefix_match_length']:.2f}`",
        "",
        "## Test Entity Span Analysis",
        f"- Teacher-Forced Museum Exact Match Rate: `{format_optional_percent(test_eval.get('teacher_forced_museum_exact_match_rate'))}`",
        f"- Teacher-Forced Museum Mean Sequence Accuracy: `{format_optional_percent(test_eval.get('teacher_forced_museum_mean_sequence_accuracy'))}`",
        f"- Teacher-Forced Museum Mean Prefix Match Length: `{format_optional_float(test_eval.get('teacher_forced_museum_mean_prefix_match_length'))}`",
        f"- Teacher-Forced Artifact Exact Match Rate: `{format_optional_percent(test_eval.get('teacher_forced_artifact_exact_match_rate'))}`",
        f"- Teacher-Forced Artifact Mean Sequence Accuracy: `{format_optional_percent(test_eval.get('teacher_forced_artifact_mean_sequence_accuracy'))}`",
        f"- Teacher-Forced Artifact Mean Prefix Match Length: `{format_optional_float(test_eval.get('teacher_forced_artifact_mean_prefix_match_length'))}`",
        f"- Generation Museum Exact Match Rate: `{format_optional_percent(test_eval.get('generation_museum_exact_match_rate'))}`",
        f"- Generation Museum Mean Sequence Accuracy: `{format_optional_percent(test_eval.get('generation_museum_mean_sequence_accuracy'))}`",
        f"- Generation Museum Mean Prefix Match Length: `{format_optional_float(test_eval.get('generation_museum_mean_prefix_match_length'))}`",
        f"- Generation Artifact Exact Match Rate: `{format_optional_percent(test_eval.get('generation_artifact_exact_match_rate'))}`",
        f"- Generation Artifact Mean Sequence Accuracy: `{format_optional_percent(test_eval.get('generation_artifact_mean_sequence_accuracy'))}`",
        f"- Generation Artifact Mean Prefix Match Length: `{format_optional_float(test_eval.get('generation_artifact_mean_prefix_match_length'))}`",
        "",
        "## Test Entity Auxiliary",
        f"- Museum Auxiliary Accuracy: `{format_optional_percent(test_eval.get('entity_aux_museum_accuracy'))}`",
        f"- Artifact Auxiliary Accuracy: `{format_optional_percent(test_eval.get('entity_aux_artifact_accuracy'))}`",
        "",
        "## Test Slot Decoder",
        f"- Full Answer Accuracy: `{format_optional_percent(test_eval.get('slot_decoder_full_answer_accuracy'))}`",
        f"- Museum Accuracy: `{format_optional_percent(test_eval.get('slot_decoder_museum_accuracy'))}`",
        f"- Artifact Accuracy: `{format_optional_percent(test_eval.get('slot_decoder_artifact_accuracy'))}`",
        f"- Artist Accuracy: `{format_optional_percent(test_eval.get('slot_decoder_artist_accuracy'))}`",
        f"- Dynasty Accuracy: `{format_optional_percent(test_eval.get('slot_decoder_dynasty_accuracy'))}`",
        "",
        "## Test Evidence Decoder",
        f"- Window Accuracy: `{format_optional_percent(test_eval.get('evidence_window_accuracy'))}`",
        f"- Mean Window Distance: `{format_optional_float(test_eval.get('evidence_window_mean_distance'))}`",
        "",
        "## Test Extract-Then-Compose",
        f"- Exact Match Rate: `{format_optional_percent(test_eval.get('extract_then_compose_exact_match_rate'))}`",
        f"- Mean Sequence Accuracy: `{format_optional_percent(test_eval.get('extract_then_compose_mean_sequence_accuracy'))}`",
        f"- Mean Prefix Match Length: `{format_optional_float(test_eval.get('extract_then_compose_mean_prefix_match_length'))}`",
        f"- Museum Accuracy: `{format_optional_percent(test_eval.get('extract_then_compose_museum_accuracy'))}`",
        f"- Artifact Accuracy: `{format_optional_percent(test_eval.get('extract_then_compose_artifact_accuracy'))}`",
        f"- Artist Accuracy: `{format_optional_percent(test_eval.get('extract_then_compose_artist_accuracy'))}`",
        f"- Dynasty Accuracy: `{format_optional_percent(test_eval.get('extract_then_compose_dynasty_accuracy'))}`",
        "",
        "## Validation Tail Error Analysis",
        f"- Tail Token Count: `{validation_tail_error_analysis['tail_token_count']}`",
        f"- Teacher-Forced Mean First Mismatch Index: `{validation_tail_error_analysis['teacher_forced']['mean_first_mismatch_index']}`",
        f"- Teacher-Forced Tail Mean Sequence Accuracy: `{validation_tail_error_analysis['teacher_forced']['mean_tail_sequence_accuracy']*100:.2f}%`",
        f"- Teacher-Forced Tail Exact Match Rate: `{validation_tail_error_analysis['teacher_forced']['tail_exact_match_rate']*100:.2f}%`",
        f"- Teacher-Forced Late Tail Failure Rate: `{validation_tail_error_analysis['teacher_forced']['late_tail_failure_rate']*100:.2f}%`",
        f"- Generation Mean First Mismatch Index: `{validation_tail_error_analysis['generation']['mean_first_mismatch_index']}`",
        f"- Generation Tail Mean Sequence Accuracy: `{validation_tail_error_analysis['generation']['mean_tail_sequence_accuracy']*100:.2f}%`",
        f"- Generation Tail Exact Match Rate: `{validation_tail_error_analysis['generation']['tail_exact_match_rate']*100:.2f}%`",
        f"- Generation Late Tail Failure Rate: `{validation_tail_error_analysis['generation']['late_tail_failure_rate']*100:.2f}%`",
        "",
        "## Test Tail Error Analysis",
        f"- Tail Token Count: `{test_tail_error_analysis['tail_token_count']}`",
        f"- Teacher-Forced Mean First Mismatch Index: `{test_tail_error_analysis['teacher_forced']['mean_first_mismatch_index']}`",
        f"- Teacher-Forced Tail Mean Sequence Accuracy: `{test_tail_error_analysis['teacher_forced']['mean_tail_sequence_accuracy']*100:.2f}%`",
        f"- Teacher-Forced Tail Exact Match Rate: `{test_tail_error_analysis['teacher_forced']['tail_exact_match_rate']*100:.2f}%`",
        f"- Teacher-Forced Late Tail Failure Rate: `{test_tail_error_analysis['teacher_forced']['late_tail_failure_rate']*100:.2f}%`",
        f"- Generation Mean First Mismatch Index: `{test_tail_error_analysis['generation']['mean_first_mismatch_index']}`",
        f"- Generation Tail Mean Sequence Accuracy: `{test_tail_error_analysis['generation']['mean_tail_sequence_accuracy']*100:.2f}%`",
        f"- Generation Tail Exact Match Rate: `{test_tail_error_analysis['generation']['tail_exact_match_rate']*100:.2f}%`",
        f"- Generation Late Tail Failure Rate: `{test_tail_error_analysis['generation']['late_tail_failure_rate']*100:.2f}%`",
        "",
        "## Training Config",
        f"- Device: `{config['device']}`",
        f"- Model Type: `{config['model_type']}`",
        f"- Epochs: `{config['epochs']}`",
        f"- Dim: `{config['dim']}`",
        f"- K: `{config['K']}`",
        f"- kr: `{config['kr']}`",
        f"- Chunk Size: `{config['chunk_size']}`",
        f"- Learning Rate: `{config['lr']}`",
        f"- Warmup Ratio: `{config['warmup_ratio']}`",
        f"- Local Context Mode: `{config['local_context_mode']}`",
        f"- Local Context Size: `{config['local_context_size']}`",
        f"- Train Dataset Size: `{config['train_dataset_size']}`",
        f"- Train Dataset Seed: `{config['train_dataset_seed']}`",
        f"- Pool Split Mode: `{config['pool_split_mode']}`",
        f"- Pair Split Seed: `{config['pair_split_seed']}`",
        f"- Generalization Score Mode: `{config['generalization_score_mode']}`",
        f"- Train Pair Count: `{config['train_pair_count']}`",
        f"- Validation Dataset Size: `{config['validation_dataset_size']}`",
        f"- Validation Dataset Seed: `{config['validation_dataset_seed']}`",
        f"- Validation Pair Count: `{config['validation_pair_count']}`",
        f"- Test Dataset Size: `{config['test_dataset_size']}`",
        f"- Test Dataset Seed: `{config['test_dataset_seed']}`",
        f"- Test Pair Count: `{config['test_pair_count']}`",
        f"- Scheduled Sampling Max Ratio: `{config['scheduled_sampling_max_ratio']}`",
        f"- Final Generation Polish Epochs: `{config['final_generation_polish_epochs']}`",
        f"- Final Generation Polish LR: `{config['final_generation_polish_lr']}`",
        f"- Generation Polish Batch Size: `{config['generation_polish_batch_size']}`",
        f"- Generation Polish Monitor Case Count: `{config['generation_polish_monitor_case_count']}`",
        f"- Museum Span Loss Weight: `{config['museum_span_loss_weight']}`",
        f"- Artifact Span Loss Weight: `{config['artifact_span_loss_weight']}`",
        f"- Entity Span Loss Min Context Bytes: `{config['entity_span_loss_min_context_bytes']}`",
        f"- Museum Auxiliary Loss Weight: `{config['museum_aux_loss_weight']}`",
        f"- Artifact Auxiliary Loss Weight: `{config['artifact_aux_loss_weight']}`",
        f"- Entity Auxiliary Loss Min Context Bytes: `{config['entity_aux_loss_min_context_bytes']}`",
        f"- Museum Hint Injection Weight: `{config['museum_hint_injection_weight']}`",
        f"- Artifact Hint Injection Weight: `{config['artifact_hint_injection_weight']}`",
        f"- Entity Hint Injection Min Context Bytes: `{config['entity_hint_injection_min_context_bytes']}`",
        f"- Entity Hint Uses Gold Labels During Training: `{config['entity_hint_use_gold_labels_during_training']}`",
        f"- Slot Decoder Loss Weight: `{config['slot_decoder_loss_weight']}`",
        f"- Slot Decoder Logit Bias: `{config['slot_decoder_logit_bias']}`",
        f"- Slot Decoder Min Context Bytes: `{config['slot_decoder_min_context_bytes']}`",
        f"- Evidence Window Count: `{config['evidence_window_count']}`",
        f"- Evidence Loss Weight: `{config['evidence_loss_weight']}`",
        f"- Evidence Hint Weight: `{config['evidence_hint_weight']}`",
        f"- Evidence Min Context Bytes: `{config['evidence_min_context_bytes']}`",
    ]
    if search_results:
        lines.extend(
            [
                "",
                "## Search Summary",
            ]
        )
        for rank, result in enumerate(search_results, start=1):
            lines.append(
                f"- Trial {rank}: epochs={result['epochs']}, kr={result['kr']}, chunk_size={result['chunk_size']}, lr={result['lr']}, "
                f"warmup_ratio={result['warmup_ratio']}, "
                f"scheduled_sampling_max_ratio={result['scheduled_sampling_max_ratio']}, "
                f"val_gen_exact={result['validation_generation_exact_match_rate']*100:.2f}%, "
                f"val_gen_seq={result['validation_generation_mean_sequence_accuracy']*100:.2f}%, "
                f"val_tf_exact={result['validation_teacher_forced_exact_match_rate']*100:.2f}%, "
                f"val_tf_seq={result['validation_teacher_forced_mean_sequence_accuracy']*100:.2f}%, "
                f"test_gen_exact={result['test_generation_exact_match_rate']*100:.2f}%, "
                f"test_gen_seq={result['test_generation_mean_sequence_accuracy']*100:.2f}%, "
                f"test_tf_exact={result['test_teacher_forced_exact_match_rate']*100:.2f}%, "
                f"test_tf_seq={result['test_teacher_forced_mean_sequence_accuracy']*100:.2f}%"
            )
        lines.extend(["", "## Validation Close Misses"])
        for mode_label, mode_key in (("Teacher-Forced", "teacher_forced"), ("Generation", "generation")):
            lines.append(f"- {mode_label}:")
            for case_summary in validation_tail_error_analysis[mode_key]["close_miss_cases"]:
                lines.append(
                    f"  question={case_summary['question']} | museum={case_summary['museum']} | artifact={case_summary['artifact']} "
                    f"| prefix={case_summary['prefix_match_length']} | tail_seq_acc={case_summary['tail_sequence_accuracy']*100:.2f}% "
                    f"| museum_span_seq={format_optional_percent(case_summary.get('museum_span_sequence_accuracy'))} "
                    f"| artifact_span_seq={format_optional_percent(case_summary.get('artifact_span_sequence_accuracy'))} "
                    f"| first_mismatch={case_summary['first_mismatch_index']}"
                )
        lines.extend(["", "## Test Close Misses"])
        for mode_label, mode_key in (("Teacher-Forced", "teacher_forced"), ("Generation", "generation")):
            lines.append(f"- {mode_label}:")
            for case_summary in test_tail_error_analysis[mode_key]["close_miss_cases"]:
                lines.append(
                    f"  question={case_summary['question']} | museum={case_summary['museum']} | artifact={case_summary['artifact']} "
                    f"| prefix={case_summary['prefix_match_length']} | tail_seq_acc={case_summary['tail_sequence_accuracy']*100:.2f}% "
                    f"| museum_span_seq={format_optional_percent(case_summary.get('museum_span_sequence_accuracy'))} "
                    f"| artifact_span_seq={format_optional_percent(case_summary.get('artifact_span_sequence_accuracy'))} "
                    f"| first_mismatch={case_summary['first_mismatch_index']}"
                )
    write_markdown(reports_dir / "json_retrieval_generalization_report.md", lines)


def train_single_configuration(
    case,
    device,
    epochs,
    eval_interval,
    dim,
    K,
    kr,
    chunk_size,
    lr,
    warmup_ratio,
    scheduled_sampling_max_ratio=DEFAULT_SCHEDULED_SAMPLING_MAX_RATIO,
    train_dataset_size=DEFAULT_TRAIN_DATASET_SIZE,
    train_dataset_seed=DEFAULT_TRAIN_DATASET_SEED,
    training_cases_override=None,
    target_case_sampling_ratio=DEFAULT_TARGET_CASE_SAMPLING_RATIO,
    final_polish_epochs=DEFAULT_FINAL_POLISH_EPOCHS,
    final_polish_lr=None,
    final_generation_polish_epochs=DEFAULT_FINAL_GENERATION_POLISH_EPOCHS,
    final_generation_polish_lr=None,
    generation_polish_max_self_feed_ratio=DEFAULT_GENERATION_POLISH_MAX_SELF_FEED_RATIO,
    generation_polish_rollout_loss_weight=DEFAULT_GENERATION_POLISH_ROLLOUT_LOSS_WEIGHT,
    generation_polish_teacher_forced_loss_weight=DEFAULT_GENERATION_POLISH_TEACHER_FORCED_LOSS_WEIGHT,
    generation_polish_batch_size=DEFAULT_GENERATION_POLISH_BATCH_SIZE,
    generation_polish_monitor_case_count=DEFAULT_GENERATION_POLISH_MONITOR_CASE_COUNT,
    museum_span_loss_weight=DEFAULT_MUSEUM_SPAN_LOSS_WEIGHT,
    artifact_span_loss_weight=DEFAULT_ARTIFACT_SPAN_LOSS_WEIGHT,
    entity_span_loss_min_context_bytes=DEFAULT_ENTITY_SPAN_LOSS_MIN_CONTEXT_BYTES,
    museum_aux_loss_weight=DEFAULT_MUSEUM_AUX_LOSS_WEIGHT,
    artifact_aux_loss_weight=DEFAULT_ARTIFACT_AUX_LOSS_WEIGHT,
    entity_aux_loss_min_context_bytes=DEFAULT_ENTITY_AUX_LOSS_MIN_CONTEXT_BYTES,
    museum_hint_injection_weight=DEFAULT_MUSEUM_HINT_INJECTION_WEIGHT,
    artifact_hint_injection_weight=DEFAULT_ARTIFACT_HINT_INJECTION_WEIGHT,
    entity_hint_injection_min_context_bytes=DEFAULT_ENTITY_HINT_INJECTION_MIN_CONTEXT_BYTES,
    entity_hint_use_gold_labels_during_training=DEFAULT_ENTITY_HINT_USE_GOLD_LABELS_DURING_TRAINING,
    slot_decoder_loss_weight=DEFAULT_SLOT_DECODER_LOSS_WEIGHT,
    slot_decoder_logit_bias=DEFAULT_SLOT_DECODER_LOGIT_BIAS,
    slot_decoder_min_context_bytes=DEFAULT_SLOT_DECODER_MIN_CONTEXT_BYTES,
    evidence_window_count=DEFAULT_EVIDENCE_WINDOW_COUNT,
    evidence_loss_weight=DEFAULT_EVIDENCE_LOSS_WEIGHT,
    evidence_hint_weight=DEFAULT_EVIDENCE_HINT_WEIGHT,
    evidence_min_context_bytes=DEFAULT_EVIDENCE_MIN_CONTEXT_BYTES,
    model_type=DEFAULT_MODEL_TYPE,
    local_context_size=DEFAULT_LOCAL_CONTEXT_SIZE,
    local_context_mode=DEFAULT_LOCAL_CONTEXT_MODE,
    return_model=False,
):
    if not isinstance(device, torch.device):
        device = torch.device(device)
    curriculum_plan = build_curriculum_plan(case, epochs)
    if training_cases_override is None:
        training_cases = build_random_training_case_pool(
            reference_case=case,
            dataset_size=train_dataset_size,
            seed=train_dataset_seed,
        )
    else:
        training_cases = list(training_cases_override)
        if not training_cases:
            raise ValueError("training_cases_override must contain at least one case.")
    training_case_sampler = random.Random(train_dataset_seed)

    model = build_retrieval_model(
        model_type=model_type,
        vocab_size=VOCAB_SIZE,
        dim=dim,
        K=K,
        kr=kr,
        chunk_size=chunk_size,
        local_context_size=local_context_size,
        local_context_mode=local_context_mode,
    ).to(device)
    if (
        museum_aux_loss_weight > 0.0
        or artifact_aux_loss_weight > 0.0
        or museum_hint_injection_weight > 0.0
        or artifact_hint_injection_weight > 0.0
    ):
        attach_entity_auxiliary_heads(model, device)
    if slot_decoder_loss_weight > 0.0 or slot_decoder_logit_bias > 0.0:
        attach_slot_decoder_heads(model, device)
    if evidence_loss_weight > 0.0 or evidence_hint_weight > 0.0:
        attach_evidence_heads(model, device)
    model.museum_hint_injection_weight = float(museum_hint_injection_weight)
    model.artifact_hint_injection_weight = float(artifact_hint_injection_weight)
    model.entity_hint_injection_min_context_bytes = int(entity_hint_injection_min_context_bytes)
    model.slot_decoder_loss_weight = float(slot_decoder_loss_weight)
    model.slot_decoder_logit_bias = float(slot_decoder_logit_bias)
    model.slot_decoder_min_context_bytes = int(slot_decoder_min_context_bytes)
    model.evidence_window_count = int(evidence_window_count)
    model.evidence_loss_weight = float(evidence_loss_weight)
    model.evidence_hint_weight = float(evidence_hint_weight)
    model.evidence_min_context_bytes = int(evidence_min_context_bytes)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_steps = sum(stage["epochs"] for stage in curriculum_plan)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=build_lr_schedule_lambda(total_steps, warmup_steps),
    )
    scheduled_sampling_ratio = build_scheduled_sampling_ratio_schedule(
        total_steps=total_steps,
        max_ratio=scheduled_sampling_max_ratio,
    )
    target_case_sampling_ratio = float(max(0.0, min(target_case_sampling_ratio, 1.0)))
    effective_final_polish_lr = float(final_polish_lr if final_polish_lr is not None else lr * 0.2)
    effective_final_generation_polish_lr = float(
        final_generation_polish_lr if final_generation_polish_lr is not None else lr * 0.1
    )

    history = []
    best_teacher_forced_exact_match = False
    training_mode = "random_case_pool"
    if target_case_sampling_ratio >= 1.0:
        training_mode = "target_case_only"
    elif target_case_sampling_ratio > 0.0:
        training_mode = f"mixed_case_pool(target_ratio={target_case_sampling_ratio:.2f})"

    global_epoch = 0
    for stage_idx, stage in enumerate(curriculum_plan, start=1):
        use_entity_span_loss = stage["context_bytes"] >= int(entity_span_loss_min_context_bytes)
        use_entity_aux_loss = stage["context_bytes"] >= int(entity_aux_loss_min_context_bytes)
        stage_museum_span_loss_weight = museum_span_loss_weight if use_entity_span_loss else 1.0
        stage_artifact_span_loss_weight = artifact_span_loss_weight if use_entity_span_loss else 1.0
        stage_museum_aux_loss_weight = museum_aux_loss_weight if use_entity_aux_loss else 0.0
        stage_artifact_aux_loss_weight = artifact_aux_loss_weight if use_entity_aux_loss else 0.0
        stage_preview_case = case if target_case_sampling_ratio >= 1.0 else training_cases[0]
        preview_X, _ = build_training_example(stage_preview_case, context_bytes=stage["context_bytes"])
        print(
            f"\n--- Curriculum Stage {stage_idx}/{len(curriculum_plan)} | "
            f"context={stage['name']} | approx_train_seq_len={preview_X.shape[1]} "
            f"| stage_epochs={stage['epochs']} | train_cases={len(training_cases)} ---"
        )

        for stage_epoch in range(1, stage["epochs"] + 1):
            global_epoch += 1
            train_case = select_training_case(
                training_cases=training_cases,
                reference_case=case,
                sampler=training_case_sampler,
                target_case_sampling_ratio=target_case_sampling_ratio,
            )
            X, Y = build_training_example(train_case, context_bytes=stage["context_bytes"])
            X, Y = X.to(device), Y.to(device)
            model.train()
            optimizer.zero_grad()
            current_sampling_ratio = scheduled_sampling_ratio(global_epoch - 1)
            sampled_token_count = 0
            teacher_forced_logits = None
            aux_metrics = {"museum_loss": 0.0, "artifact_loss": 0.0}
            slot_decoder_metrics = {
                f"{slot_name}_loss": 0.0
                for slot_name in ANSWER_SLOT_NAMES
            }
            evidence_metrics = {"window_loss": 0.0, "window_exact_match": 0.0}

            if current_sampling_ratio > 0.0:
                with torch.no_grad():
                    teacher_forced_logits = forward_json_retrieval(
                        model,
                        X,
                        train_case,
                        context_bytes=stage["context_bytes"],
                        use_gold_entity_hints=entity_hint_use_gold_labels_during_training,
                    )
                    teacher_forced_logits_target, _ = collect_answer_targets(teacher_forced_logits, Y)
                    teacher_forced_preds = predict_byte_tokens(teacher_forced_logits_target)
                    train_X, sampled_token_count = build_scheduled_sampling_inputs(
                        X,
                        Y,
                        teacher_forced_preds,
                        current_sampling_ratio,
                    )
            else:
                train_X = X

            if (
                stage_museum_aux_loss_weight > 0.0
                or stage_artifact_aux_loss_weight > 0.0
                or entity_hinting_is_active(model, context_bytes=stage["context_bytes"])
                or slot_decoder_is_active(model, context_bytes=stage["context_bytes"])
                or evidence_supervision_is_active(model, context_bytes=stage["context_bytes"])
            ):
                logits, hidden_states, _ = forward_json_retrieval(
                    model,
                    train_X,
                    train_case,
                    context_bytes=stage["context_bytes"],
                    return_hidden=True,
                    use_gold_entity_hints=entity_hint_use_gold_labels_during_training,
                )
            else:
                logits = model(train_X)
                hidden_states = None
            logits_target, targets = collect_answer_targets(logits, Y)
            loss = compute_weighted_answer_loss(
                logits_target,
                targets,
                case=train_case,
                museum_span_loss_weight=stage_museum_span_loss_weight,
                artifact_span_loss_weight=stage_artifact_span_loss_weight,
            )
            if hidden_states is not None:
                aux_loss, aux_metrics = compute_entity_auxiliary_loss(
                    model=model,
                    hidden_states=hidden_states,
                    case=train_case,
                    device=device,
                    context_bytes=stage["context_bytes"],
                    museum_aux_loss_weight=stage_museum_aux_loss_weight,
                    artifact_aux_loss_weight=stage_artifact_aux_loss_weight,
                )
                loss = loss + aux_loss
                if slot_decoder_is_active(model, context_bytes=stage["context_bytes"]):
                    slot_decoder_loss, slot_decoder_metrics = compute_slot_decoder_loss(
                        model=model,
                        hidden_states=hidden_states,
                        case=train_case,
                        device=device,
                        context_bytes=stage["context_bytes"],
                    )
                    loss = loss + float(slot_decoder_loss_weight) * slot_decoder_loss
                if evidence_supervision_is_active(model, context_bytes=stage["context_bytes"]):
                    evidence_loss, evidence_metrics = compute_evidence_loss(
                        model=model,
                        hidden_states=hidden_states,
                        case=train_case,
                        device=device,
                        context_bytes=stage["context_bytes"],
                    )
                    loss = loss + float(evidence_loss_weight) * evidence_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                train_preds = predict_byte_tokens(logits_target)
                train_acc = (train_preds == targets).float().mean().item()

            is_log_step = (
                global_epoch == 1
                or stage_epoch % eval_interval == 0
                or stage_epoch == stage["epochs"]
            )
            if is_log_step:
                if teacher_forced_logits is None:
                    teacher_forced_logits = logits
                teacher_forced = evaluate_teacher_forced(teacher_forced_logits, Y, train_case)
                best_teacher_forced_exact_match = (
                    best_teacher_forced_exact_match or teacher_forced["exact_byte_match"]
                )
                history.append(
                    {
                        "epoch": global_epoch,
                        "stage": stage["name"],
                        "context_bytes": stage["context_bytes"],
                        "loss": float(loss.item()),
                        "train_token_acc": float(train_acc),
                        "teacher_forced_exact_match": teacher_forced["exact_byte_match"],
                        "teacher_forced_sequence_accuracy": teacher_forced["sequence_accuracy"],
                        "teacher_forced_prefix_match_length": teacher_forced["prefix_match_length"],
                        "first_mismatch_index": teacher_forced["first_mismatch_index"],
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "scheduled_sampling_ratio": float(current_sampling_ratio),
                        "scheduled_sampling_tokens": sampled_token_count,
                        "museum_aux_loss": aux_metrics["museum_loss"],
                        "artifact_aux_loss": aux_metrics["artifact_loss"],
                        "slot_decoder_loss": (
                            sum(slot_decoder_metrics.values()) / max(1, len(slot_decoder_metrics))
                        ),
                        "evidence_loss": evidence_metrics["window_loss"],
                        "evidence_window_exact_match": evidence_metrics["window_exact_match"],
                    }
                )
                print(
                    f"Epoch {global_epoch:4d} | Stage: {stage['name']:>3} | Loss: {loss.item():.4f} "
                    f"| Train Token Acc: {train_acc*100:5.1f}% "
                    f"| Seq Acc: {teacher_forced['sequence_accuracy']*100:5.1f}% "
                    f"| Prefix Match: {teacher_forced['prefix_match_length']:3d} "
                    f"| LR: {optimizer.param_groups[0]['lr']:.2e} "
                    f"| SS Ratio: {current_sampling_ratio:.2f} "
                    f"| SS Tokens: {sampled_token_count:2d} "
                    f"| Aux(M/A): {aux_metrics['museum_loss']:.3f}/{aux_metrics['artifact_loss']:.3f} "
                    f"| Slot Loss: {sum(slot_decoder_metrics.values()) / max(1, len(slot_decoder_metrics)):.3f} "
                    f"| Evidence: {evidence_metrics['window_loss']:.3f}/{evidence_metrics['window_exact_match']:.0f} "
                    f"| Teacher Forced Exact Match: {teacher_forced['exact_byte_match']}"
                )

            del logits, logits_target, targets, loss, train_X, teacher_forced_logits, hidden_states
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    polish_exact_match = False
    if final_polish_epochs > 0:
        polish_exact_match, global_epoch = run_target_case_polish(
            model=model,
            case=case,
            device=device,
            eval_interval=eval_interval,
            epochs=final_polish_epochs,
            lr=effective_final_polish_lr,
            history=history,
            history_epoch_offset=global_epoch,
            museum_span_loss_weight=museum_span_loss_weight,
            artifact_span_loss_weight=artifact_span_loss_weight,
        )
        best_teacher_forced_exact_match = best_teacher_forced_exact_match or polish_exact_match

    generation_exact_match = False
    if final_generation_polish_epochs > 0:
        if training_cases_override is None:
            generation_polish_cases = [case]
        else:
            generation_polish_cases = list(training_cases)
        monitor_case_count = max(1, min(int(generation_polish_monitor_case_count), len(generation_polish_cases)))
        if monitor_case_count >= len(generation_polish_cases):
            generation_polish_monitor_cases = list(generation_polish_cases)
        else:
            monitor_sampler = random.Random(train_dataset_seed + 17)
            generation_polish_monitor_cases = monitor_sampler.sample(
                generation_polish_cases,
                k=monitor_case_count,
            )
        generation_exact_match, global_epoch = run_generation_polish(
            model=model,
            case=case,
            device=device,
            eval_interval=eval_interval,
            epochs=final_generation_polish_epochs,
            lr=effective_final_generation_polish_lr,
            history=history,
            history_epoch_offset=global_epoch,
            max_self_feed_ratio=generation_polish_max_self_feed_ratio,
            rollout_loss_weight=generation_polish_rollout_loss_weight,
            teacher_forced_loss_weight=generation_polish_teacher_forced_loss_weight,
            polish_cases=generation_polish_cases,
            generation_polish_batch_size=generation_polish_batch_size,
            monitor_cases=generation_polish_monitor_cases,
            generation_polish_seed=train_dataset_seed + 1009,
            museum_span_loss_weight=museum_span_loss_weight,
            artifact_span_loss_weight=artifact_span_loss_weight,
        )

    full_X, full_Y = build_training_example(case)
    full_X, full_Y = full_X.to(device), full_Y.to(device)
    with torch.no_grad():
        full_logits = forward_json_retrieval(model, full_X, case)
    final_teacher_forced = evaluate_teacher_forced(full_logits, full_Y, case)
    final_generation = evaluate_generation(model, case, device)
    config = {
        "device": str(device),
        "model_type": model_type,
        "epochs": epochs,
        "eval_interval": eval_interval,
        "dim": dim,
        "K": K,
        "kr": kr,
        "chunk_size": chunk_size,
        "lr": lr,
        "warmup_ratio": warmup_ratio,
        "scheduled_sampling_max_ratio": scheduled_sampling_max_ratio,
        "target_case_sampling_ratio": target_case_sampling_ratio,
        "train_dataset_size": len(training_cases),
        "train_dataset_seed": train_dataset_seed,
        "training_mode": training_mode,
        "curriculum_labels": " -> ".join(stage["name"] for stage in curriculum_plan),
        "final_polish_epochs": int(final_polish_epochs),
        "final_polish_lr": effective_final_polish_lr if final_polish_epochs > 0 else None,
        "final_generation_polish_epochs": int(final_generation_polish_epochs),
        "final_generation_polish_lr": (
            effective_final_generation_polish_lr if final_generation_polish_epochs > 0 else None
        ),
        "generation_polish_max_self_feed_ratio": generation_polish_max_self_feed_ratio,
        "generation_polish_rollout_loss_weight": generation_polish_rollout_loss_weight,
        "generation_polish_teacher_forced_loss_weight": generation_polish_teacher_forced_loss_weight,
        "generation_polish_batch_size": int(generation_polish_batch_size),
        "generation_polish_monitor_case_count": int(generation_polish_monitor_case_count),
        "museum_span_loss_weight": float(museum_span_loss_weight),
        "artifact_span_loss_weight": float(artifact_span_loss_weight),
        "entity_span_loss_min_context_bytes": int(entity_span_loss_min_context_bytes),
        "museum_aux_loss_weight": float(museum_aux_loss_weight),
        "artifact_aux_loss_weight": float(artifact_aux_loss_weight),
        "entity_aux_loss_min_context_bytes": int(entity_aux_loss_min_context_bytes),
        "museum_hint_injection_weight": float(museum_hint_injection_weight),
        "artifact_hint_injection_weight": float(artifact_hint_injection_weight),
        "entity_hint_injection_min_context_bytes": int(entity_hint_injection_min_context_bytes),
        "entity_hint_use_gold_labels_during_training": bool(entity_hint_use_gold_labels_during_training),
        "slot_decoder_loss_weight": float(slot_decoder_loss_weight),
        "slot_decoder_logit_bias": float(slot_decoder_logit_bias),
        "slot_decoder_min_context_bytes": int(slot_decoder_min_context_bytes),
        "evidence_window_count": int(evidence_window_count),
        "evidence_loss_weight": float(evidence_loss_weight),
        "evidence_hint_weight": float(evidence_hint_weight),
        "evidence_min_context_bytes": int(evidence_min_context_bytes),
        "local_context_size": int(local_context_size),
        "local_context_mode": local_context_mode,
        "best_teacher_forced_exact_match": (
            best_teacher_forced_exact_match or final_teacher_forced["exact_byte_match"]
        ),
        "best_generation_exact_match": generation_exact_match or final_generation["exact_byte_match"],
    }

    result = {
        "config": config,
        "history": history,
        "teacher_forced_evaluation": final_teacher_forced,
        "evaluation": final_generation,
    }
    if return_model:
        result["model"] = model
    return result


def run_json_retrieval_test(
    input_path="tests/fixtures/test_input.json",
    metadata_path="tests/fixtures/test_metadata.json",
    reports_dir=None,
    epochs=1000,
    eval_interval=10,
    dim=128,
    K=128,
    kr_grid=None,
    chunk_size_grid=None,
    lr_grid=None,
    warmup_ratio=DEFAULT_WARMUP_RATIO,
    warmup_ratio_grid=None,
    scheduled_sampling_max_ratio=DEFAULT_SCHEDULED_SAMPLING_MAX_RATIO,
    scheduled_sampling_max_ratio_grid=None,
    train_dataset_size=DEFAULT_TRAIN_DATASET_SIZE,
    train_dataset_seed=DEFAULT_TRAIN_DATASET_SEED,
    target_case_sampling_ratio=DEFAULT_TARGET_CASE_SAMPLING_RATIO,
    final_polish_epochs=DEFAULT_FINAL_POLISH_EPOCHS,
    final_polish_lr=None,
    final_generation_polish_epochs=DEFAULT_FINAL_GENERATION_POLISH_EPOCHS,
    final_generation_polish_lr=None,
    generation_polish_max_self_feed_ratio=DEFAULT_GENERATION_POLISH_MAX_SELF_FEED_RATIO,
    generation_polish_rollout_loss_weight=DEFAULT_GENERATION_POLISH_ROLLOUT_LOSS_WEIGHT,
    generation_polish_teacher_forced_loss_weight=DEFAULT_GENERATION_POLISH_TEACHER_FORCED_LOSS_WEIGHT,
    generation_polish_batch_size=DEFAULT_GENERATION_POLISH_BATCH_SIZE,
    generation_polish_monitor_case_count=DEFAULT_GENERATION_POLISH_MONITOR_CASE_COUNT,
    museum_span_loss_weight=DEFAULT_MUSEUM_SPAN_LOSS_WEIGHT,
    artifact_span_loss_weight=DEFAULT_ARTIFACT_SPAN_LOSS_WEIGHT,
    entity_span_loss_min_context_bytes=DEFAULT_ENTITY_SPAN_LOSS_MIN_CONTEXT_BYTES,
    museum_aux_loss_weight=DEFAULT_MUSEUM_AUX_LOSS_WEIGHT,
    artifact_aux_loss_weight=DEFAULT_ARTIFACT_AUX_LOSS_WEIGHT,
    entity_aux_loss_min_context_bytes=DEFAULT_ENTITY_AUX_LOSS_MIN_CONTEXT_BYTES,
    museum_hint_injection_weight=DEFAULT_MUSEUM_HINT_INJECTION_WEIGHT,
    artifact_hint_injection_weight=DEFAULT_ARTIFACT_HINT_INJECTION_WEIGHT,
    entity_hint_injection_min_context_bytes=DEFAULT_ENTITY_HINT_INJECTION_MIN_CONTEXT_BYTES,
    entity_hint_use_gold_labels_during_training=DEFAULT_ENTITY_HINT_USE_GOLD_LABELS_DURING_TRAINING,
    slot_decoder_loss_weight=DEFAULT_SLOT_DECODER_LOSS_WEIGHT,
    slot_decoder_logit_bias=DEFAULT_SLOT_DECODER_LOGIT_BIAS,
    slot_decoder_min_context_bytes=DEFAULT_SLOT_DECODER_MIN_CONTEXT_BYTES,
    evidence_window_count=DEFAULT_EVIDENCE_WINDOW_COUNT,
    evidence_loss_weight=DEFAULT_EVIDENCE_LOSS_WEIGHT,
    evidence_hint_weight=DEFAULT_EVIDENCE_HINT_WEIGHT,
    evidence_min_context_bytes=DEFAULT_EVIDENCE_MIN_CONTEXT_BYTES,
    model_type=DEFAULT_MODEL_TYPE,
    local_context_size=DEFAULT_LOCAL_CONTEXT_SIZE,
    local_context_mode=DEFAULT_LOCAL_CONTEXT_MODE,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    case = load_json_retrieval_case(input_path=input_path, metadata_path=metadata_path)
    curriculum_plan = build_curriculum_plan(case, epochs)

    kr_grid = kr_grid or DEFAULT_KR_GRID
    chunk_size_grid = chunk_size_grid or DEFAULT_CHUNK_SIZE_GRID
    lr_grid = lr_grid or DEFAULT_LR_GRID
    warmup_ratio_grid = resolve_search_grid(
        explicit_grid=warmup_ratio_grid,
        scalar_value=warmup_ratio,
        default_grid=DEFAULT_WARMUP_RATIO_GRID,
        default_scalar_value=DEFAULT_WARMUP_RATIO,
    )
    scheduled_sampling_max_ratio_grid = resolve_search_grid(
        explicit_grid=scheduled_sampling_max_ratio_grid,
        scalar_value=scheduled_sampling_max_ratio,
        default_grid=DEFAULT_SCHEDULED_SAMPLING_MAX_RATIO_GRID,
        default_scalar_value=DEFAULT_SCHEDULED_SAMPLING_MAX_RATIO,
    )
    search_space = [
        (kr, chunk_size, lr, warmup_ratio_value, scheduled_sampling_max_ratio_value)
        for kr in kr_grid
        for chunk_size in chunk_size_grid
        for lr in lr_grid
        for warmup_ratio_value in warmup_ratio_grid
        for scheduled_sampling_max_ratio_value in scheduled_sampling_max_ratio_grid
    ]

    print("\n--- Running JSON File Retrieval Test ---")
    print(f"Using device: {device}")
    print(
        f"Config | full_seq_len={len(case['sample_bytes']) + 1 + len(case['question_bytes']) + 1 + len(case['expected_answer_bytes']) - 1} "
        f"| answer_len={len(case['expected_answer_bytes'])} | dim={dim} | K={K} | epochs={epochs} | model_type={model_type}"
    )
    print(
        f"Training Data | mode=random_case_pool | dataset_size={train_dataset_size} "
        f"| dataset_seed={train_dataset_seed} | target_case_sampling_ratio={target_case_sampling_ratio}"
    )
    print(
        f"Local Context | mode={local_context_mode} | size={local_context_size}"
    )
    print(
        f"Entity Span Loss | museum={museum_span_loss_weight} | artifact={artifact_span_loss_weight} "
        f"| min_context_bytes={entity_span_loss_min_context_bytes}"
    )
    print(
        f"Entity Auxiliary Loss | museum={museum_aux_loss_weight} | artifact={artifact_aux_loss_weight} "
        f"| min_context_bytes={entity_aux_loss_min_context_bytes}"
    )
    print(
        f"Entity Hint Injection | museum={museum_hint_injection_weight} | artifact={artifact_hint_injection_weight} "
        f"| min_context_bytes={entity_hint_injection_min_context_bytes}"
    )
    print(
        f"Entity Hint Training | use_gold_labels={entity_hint_use_gold_labels_during_training}"
    )
    print(
        f"Slot Decoder | loss_weight={slot_decoder_loss_weight} | logit_bias={slot_decoder_logit_bias} "
        f"| min_context_bytes={slot_decoder_min_context_bytes}"
    )
    print(
        f"Evidence Decoder | windows={evidence_window_count} | loss_weight={evidence_loss_weight} "
        f"| hint_weight={evidence_hint_weight} | min_context_bytes={evidence_min_context_bytes}"
    )
    print(
        "Curriculum | "
        + " -> ".join(f"{stage['name']}({stage['epochs']})" for stage in curriculum_plan)
    )
    print(
        "Grid Search | "
        f"kr={kr_grid} | chunk_size={chunk_size_grid} | lr={lr_grid} "
        f"| warmup_ratio={warmup_ratio_grid} "
        f"| scheduled_sampling_max_ratio={scheduled_sampling_max_ratio_grid}"
    )
    if final_polish_epochs > 0:
        polish_lr_display = final_polish_lr if final_polish_lr is not None else "auto"
        print(
            f"Final Polish | epochs={final_polish_epochs} | lr={polish_lr_display}"
        )
    if final_generation_polish_epochs > 0:
        gen_polish_lr_display = (
            final_generation_polish_lr if final_generation_polish_lr is not None else "auto"
        )
        print(
            f"Generation Polish | epochs={final_generation_polish_epochs} "
            f"| lr={gen_polish_lr_display} | max_self_feed_ratio={generation_polish_max_self_feed_ratio} "
            f"| rollout_loss_weight={generation_polish_rollout_loss_weight} "
            f"| teacher_forced_loss_weight={generation_polish_teacher_forced_loss_weight}"
        )

    best_result = None
    search_results = []
    for trial_idx, (
        kr,
        chunk_size,
        lr,
        warmup_ratio_value,
        scheduled_sampling_max_ratio_value,
    ) in enumerate(search_space, start=1):
        print(
            f"\n=== Search Trial {trial_idx}/{len(search_space)} | "
            f"kr={kr} | chunk_size={chunk_size} | lr={lr:.0e} | warmup_ratio={warmup_ratio_value} "
            f"| scheduled_sampling_max_ratio={scheduled_sampling_max_ratio_value} ==="
        )
        result = train_single_configuration(
            case=case,
            device=device,
            epochs=epochs,
            eval_interval=eval_interval,
            dim=dim,
            K=K,
            kr=kr,
            chunk_size=chunk_size,
            lr=lr,
            warmup_ratio=warmup_ratio_value,
            scheduled_sampling_max_ratio=scheduled_sampling_max_ratio_value,
            train_dataset_size=train_dataset_size,
            train_dataset_seed=train_dataset_seed,
            target_case_sampling_ratio=target_case_sampling_ratio,
            final_polish_epochs=final_polish_epochs,
            final_polish_lr=final_polish_lr,
            final_generation_polish_epochs=final_generation_polish_epochs,
            final_generation_polish_lr=final_generation_polish_lr,
            generation_polish_max_self_feed_ratio=generation_polish_max_self_feed_ratio,
            generation_polish_rollout_loss_weight=generation_polish_rollout_loss_weight,
            generation_polish_teacher_forced_loss_weight=generation_polish_teacher_forced_loss_weight,
            generation_polish_batch_size=generation_polish_batch_size,
            generation_polish_monitor_case_count=generation_polish_monitor_case_count,
            museum_span_loss_weight=museum_span_loss_weight,
            artifact_span_loss_weight=artifact_span_loss_weight,
            entity_span_loss_min_context_bytes=entity_span_loss_min_context_bytes,
            museum_aux_loss_weight=museum_aux_loss_weight,
            artifact_aux_loss_weight=artifact_aux_loss_weight,
            entity_aux_loss_min_context_bytes=entity_aux_loss_min_context_bytes,
            museum_hint_injection_weight=museum_hint_injection_weight,
            artifact_hint_injection_weight=artifact_hint_injection_weight,
            entity_hint_injection_min_context_bytes=entity_hint_injection_min_context_bytes,
            entity_hint_use_gold_labels_during_training=entity_hint_use_gold_labels_during_training,
            slot_decoder_loss_weight=slot_decoder_loss_weight,
            slot_decoder_logit_bias=slot_decoder_logit_bias,
            slot_decoder_min_context_bytes=slot_decoder_min_context_bytes,
            evidence_window_count=evidence_window_count,
            evidence_loss_weight=evidence_loss_weight,
            evidence_hint_weight=evidence_hint_weight,
            evidence_min_context_bytes=evidence_min_context_bytes,
            model_type=model_type,
            local_context_size=local_context_size,
            local_context_mode=local_context_mode,
        )
        search_results.append(summarize_search_result(result))
        if best_result is None or score_search_result(result) > score_search_result(best_result):
            best_result = result

    search_results.sort(
        key=lambda item: (
            int(item["generation_exact_byte_match"]),
            int(item["teacher_forced_exact_byte_match"]),
            item["generation_prefix_match_length"],
            item["generation_sequence_accuracy"],
            item["teacher_forced_prefix_match_length"],
            item["teacher_forced_sequence_accuracy"],
        ),
        reverse=True,
    )

    if reports_dir is not None:
        save_json_retrieval_reports(
            case,
            best_result["config"],
            best_result["history"],
            best_result["teacher_forced_evaluation"],
            best_result["evaluation"],
            search_results,
            Path(reports_dir),
        )

    print("\n=== JSON Retrieval Result ===")
    print(
        f"Best Config | kr={best_result['config']['kr']} | chunk_size={best_result['config']['chunk_size']} "
        f"| lr={best_result['config']['lr']:.0e} | warmup_ratio={best_result['config']['warmup_ratio']} "
        f"| scheduled_sampling_max_ratio={best_result['config']['scheduled_sampling_max_ratio']}"
    )
    print(
        f"Teacher Forced Prefix Match Length: {best_result['teacher_forced_evaluation']['prefix_match_length']}"
    )
    print(
        f"Teacher Forced Sequence Accuracy: "
        f"{best_result['teacher_forced_evaluation']['sequence_accuracy']*100:.2f}%"
    )
    print(
        f"Teacher Forced Exact Match: {best_result['teacher_forced_evaluation']['exact_byte_match']}"
    )
    print(f"Exact Byte Match: {best_result['evaluation']['exact_byte_match']}")
    print(f"Exact Text Match: {best_result['evaluation']['exact_text_match']}")
    print(f"Prefix Match Length: {best_result['evaluation']['prefix_match_length']}")
    print(f"Sequence Accuracy: {best_result['evaluation']['sequence_accuracy']*100:.2f}%")
    print(f"First Mismatch Index: {best_result['evaluation']['first_mismatch_index']}")
    print(f"Predicted Answer: {best_result['evaluation']['predicted_text']}")

    return {
        "config": best_result["config"],
        "history": best_result["history"],
        "teacher_forced_evaluation": best_result["teacher_forced_evaluation"],
        "evaluation": best_result["evaluation"],
        "search_results": search_results,
    }


def run_json_retrieval_generalization_test(
    input_path="tests/fixtures/test_input.json",
    metadata_path="tests/fixtures/test_metadata.json",
    reports_dir=None,
    epochs=1000,
    epochs_grid=None,
    eval_interval=10,
    dim=128,
    K=128,
    kr_grid=None,
    chunk_size_grid=None,
    lr_grid=None,
    warmup_ratio=DEFAULT_WARMUP_RATIO,
    warmup_ratio_grid=None,
    scheduled_sampling_max_ratio=DEFAULT_SCHEDULED_SAMPLING_MAX_RATIO,
    scheduled_sampling_max_ratio_grid=None,
    train_dataset_size=DEFAULT_TRAIN_DATASET_SIZE,
    train_dataset_seed=DEFAULT_TRAIN_DATASET_SEED,
    validation_dataset_size=DEFAULT_VALIDATION_DATASET_SIZE,
    validation_dataset_seed=DEFAULT_VALIDATION_DATASET_SEED,
    test_dataset_size=DEFAULT_TEST_DATASET_SIZE,
    test_dataset_seed=DEFAULT_TEST_DATASET_SEED,
    pair_split_seed=DEFAULT_PAIR_SPLIT_SEED,
    train_pair_ratio=DEFAULT_TRAIN_PAIR_RATIO,
    validation_pair_ratio=DEFAULT_VALIDATION_PAIR_RATIO,
    target_case_sampling_ratio=DEFAULT_TARGET_CASE_SAMPLING_RATIO,
    final_polish_epochs=0,
    final_polish_lr=None,
    final_generation_polish_epochs=0,
    final_generation_polish_lr=None,
    generation_polish_max_self_feed_ratio=DEFAULT_GENERATION_POLISH_MAX_SELF_FEED_RATIO,
    generation_polish_rollout_loss_weight=DEFAULT_GENERATION_POLISH_ROLLOUT_LOSS_WEIGHT,
    generation_polish_teacher_forced_loss_weight=DEFAULT_GENERATION_POLISH_TEACHER_FORCED_LOSS_WEIGHT,
    generation_polish_batch_size=DEFAULT_GENERATION_POLISH_BATCH_SIZE,
    generation_polish_monitor_case_count=DEFAULT_GENERATION_POLISH_MONITOR_CASE_COUNT,
    museum_span_loss_weight=DEFAULT_MUSEUM_SPAN_LOSS_WEIGHT,
    artifact_span_loss_weight=DEFAULT_ARTIFACT_SPAN_LOSS_WEIGHT,
    entity_span_loss_min_context_bytes=DEFAULT_ENTITY_SPAN_LOSS_MIN_CONTEXT_BYTES,
    museum_aux_loss_weight=DEFAULT_MUSEUM_AUX_LOSS_WEIGHT,
    artifact_aux_loss_weight=DEFAULT_ARTIFACT_AUX_LOSS_WEIGHT,
    entity_aux_loss_min_context_bytes=DEFAULT_ENTITY_AUX_LOSS_MIN_CONTEXT_BYTES,
    museum_hint_injection_weight=DEFAULT_MUSEUM_HINT_INJECTION_WEIGHT,
    artifact_hint_injection_weight=DEFAULT_ARTIFACT_HINT_INJECTION_WEIGHT,
    entity_hint_injection_min_context_bytes=DEFAULT_ENTITY_HINT_INJECTION_MIN_CONTEXT_BYTES,
    entity_hint_use_gold_labels_during_training=DEFAULT_ENTITY_HINT_USE_GOLD_LABELS_DURING_TRAINING,
    slot_decoder_loss_weight=DEFAULT_SLOT_DECODER_LOSS_WEIGHT,
    slot_decoder_logit_bias=DEFAULT_SLOT_DECODER_LOGIT_BIAS,
    slot_decoder_min_context_bytes=DEFAULT_SLOT_DECODER_MIN_CONTEXT_BYTES,
    evidence_window_count=DEFAULT_EVIDENCE_WINDOW_COUNT,
    evidence_loss_weight=DEFAULT_EVIDENCE_LOSS_WEIGHT,
    evidence_hint_weight=DEFAULT_EVIDENCE_HINT_WEIGHT,
    evidence_min_context_bytes=DEFAULT_EVIDENCE_MIN_CONTEXT_BYTES,
    model_type=DEFAULT_MODEL_TYPE,
    generalization_score_mode=DEFAULT_GENERALIZATION_SCORE_MODE,
    local_context_size=DEFAULT_LOCAL_CONTEXT_SIZE,
    local_context_mode=DEFAULT_LOCAL_CONTEXT_MODE,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reference_case = load_json_retrieval_case(input_path=input_path, metadata_path=metadata_path)
    pair_split = split_museum_artifact_pairs(
        seed=pair_split_seed,
        train_ratio=train_pair_ratio,
        validation_ratio=validation_pair_ratio,
    )
    train_pairs = pair_split["train"]
    validation_pairs = pair_split["validation"]
    test_pairs = pair_split["test"]
    used_case_signatures = set()
    training_cases, used_case_signatures = build_disjoint_case_pool(
        reference_case=reference_case,
        dataset_size=train_dataset_size,
        seed=train_dataset_seed,
        used_signatures=used_case_signatures,
        allowed_museum_artifact_pairs=train_pairs,
    )
    validation_cases, used_case_signatures = build_disjoint_case_pool(
        reference_case=reference_case,
        dataset_size=validation_dataset_size,
        seed=validation_dataset_seed,
        used_signatures=used_case_signatures,
        allowed_museum_artifact_pairs=validation_pairs,
    )
    test_cases, used_case_signatures = build_disjoint_case_pool(
        reference_case=reference_case,
        dataset_size=test_dataset_size,
        seed=test_dataset_seed,
        used_signatures=used_case_signatures,
        allowed_museum_artifact_pairs=test_pairs,
    )

    epochs_grid = sorted({int(epoch_value) for epoch_value in (epochs_grid or [epochs])})
    kr_grid = kr_grid or DEFAULT_KR_GRID
    chunk_size_grid = chunk_size_grid or DEFAULT_CHUNK_SIZE_GRID
    lr_grid = lr_grid or DEFAULT_LR_GRID
    warmup_ratio_grid = resolve_search_grid(
        explicit_grid=warmup_ratio_grid,
        scalar_value=warmup_ratio,
        default_grid=DEFAULT_WARMUP_RATIO_GRID,
        default_scalar_value=DEFAULT_WARMUP_RATIO,
    )
    scheduled_sampling_max_ratio_grid = resolve_search_grid(
        explicit_grid=scheduled_sampling_max_ratio_grid,
        scalar_value=scheduled_sampling_max_ratio,
        default_grid=DEFAULT_SCHEDULED_SAMPLING_MAX_RATIO_GRID,
        default_scalar_value=DEFAULT_SCHEDULED_SAMPLING_MAX_RATIO,
    )
    search_space = [
        (
            epoch_value,
            kr,
            chunk_size,
            lr,
            warmup_ratio_value,
            scheduled_sampling_max_ratio_value,
        )
        for epoch_value in epochs_grid
        for kr in kr_grid
        for chunk_size in chunk_size_grid
        for lr in lr_grid
        for warmup_ratio_value in warmup_ratio_grid
        for scheduled_sampling_max_ratio_value in scheduled_sampling_max_ratio_grid
    ]

    print("\n--- Running JSON Retrieval Generalization Test ---")
    print(f"Using device: {device}")
    print(
        f"Config | full_seq_len={len(reference_case['sample_bytes']) + 1 + len(reference_case['question_bytes']) + 1 + len(reference_case['expected_answer_bytes']) - 1} "
        f"| answer_len={len(reference_case['expected_answer_bytes'])} | dim={dim} | K={K} | model_type={model_type}"
    )
    print(
        f"Case Pools | train={len(training_cases)} (seed={train_dataset_seed}) "
        f"| val={len(validation_cases)} (seed={validation_dataset_seed}) "
        f"| test={len(test_cases)} (seed={test_dataset_seed})"
    )
    print(
        f"Training Data | target_case_sampling_ratio={target_case_sampling_ratio} "
        f"| disjoint_split=museum/artifact held-out"
    )
    print(
        f"Pair Split | seed={pair_split_seed} | train_pairs={len(train_pairs)} "
        f"| val_pairs={len(validation_pairs)} | test_pairs={len(test_pairs)}"
    )
    print(
        f"Local Context | mode={local_context_mode} | size={local_context_size}"
    )
    print(
        f"Entity Span Loss | museum={museum_span_loss_weight} | artifact={artifact_span_loss_weight} "
        f"| min_context_bytes={entity_span_loss_min_context_bytes}"
    )
    print(
        f"Entity Auxiliary Loss | museum={museum_aux_loss_weight} | artifact={artifact_aux_loss_weight} "
        f"| min_context_bytes={entity_aux_loss_min_context_bytes}"
    )
    print(
        f"Entity Hint Injection | museum={museum_hint_injection_weight} | artifact={artifact_hint_injection_weight} "
        f"| min_context_bytes={entity_hint_injection_min_context_bytes}"
    )
    print(
        f"Entity Hint Training | use_gold_labels={entity_hint_use_gold_labels_during_training}"
    )
    print(
        f"Slot Decoder | loss_weight={slot_decoder_loss_weight} | logit_bias={slot_decoder_logit_bias} "
        f"| min_context_bytes={slot_decoder_min_context_bytes}"
    )
    print(
        f"Evidence Decoder | windows={evidence_window_count} | loss_weight={evidence_loss_weight} "
        f"| hint_weight={evidence_hint_weight} | min_context_bytes={evidence_min_context_bytes}"
    )
    print(f"Selection Score Mode | mode={generalization_score_mode}")
    print(
        "Grid Search | "
        f"epochs={epochs_grid} | kr={kr_grid} | chunk_size={chunk_size_grid} | lr={lr_grid} "
        f"| warmup_ratio={warmup_ratio_grid} "
        f"| scheduled_sampling_max_ratio={scheduled_sampling_max_ratio_grid}"
    )

    best_result = None
    search_results = []
    for trial_idx, (
        epoch_value,
        kr,
        chunk_size,
        lr,
        warmup_ratio_value,
        scheduled_sampling_max_ratio_value,
    ) in enumerate(search_space, start=1):
        curriculum_plan = build_curriculum_plan(reference_case, epoch_value)
        print(
            f"\n=== Generalization Trial {trial_idx}/{len(search_space)} | "
            f"epochs={epoch_value} | kr={kr} | chunk_size={chunk_size} | lr={lr:.0e} | warmup_ratio={warmup_ratio_value} "
            f"| scheduled_sampling_max_ratio={scheduled_sampling_max_ratio_value} ==="
        )
        print(
            "Curriculum | "
            + " -> ".join(f"{stage['name']}({stage['epochs']})" for stage in curriculum_plan)
        )
        train_result = train_single_configuration(
            case=reference_case,
            device=device,
            epochs=epoch_value,
            eval_interval=eval_interval,
            dim=dim,
            K=K,
            kr=kr,
            chunk_size=chunk_size,
            lr=lr,
            warmup_ratio=warmup_ratio_value,
            scheduled_sampling_max_ratio=scheduled_sampling_max_ratio_value,
            train_dataset_size=train_dataset_size,
            train_dataset_seed=train_dataset_seed,
            training_cases_override=training_cases,
            target_case_sampling_ratio=target_case_sampling_ratio,
            final_polish_epochs=final_polish_epochs,
            final_polish_lr=final_polish_lr,
            final_generation_polish_epochs=final_generation_polish_epochs,
            final_generation_polish_lr=final_generation_polish_lr,
            generation_polish_max_self_feed_ratio=generation_polish_max_self_feed_ratio,
            generation_polish_rollout_loss_weight=generation_polish_rollout_loss_weight,
            generation_polish_teacher_forced_loss_weight=generation_polish_teacher_forced_loss_weight,
            generation_polish_batch_size=generation_polish_batch_size,
            generation_polish_monitor_case_count=generation_polish_monitor_case_count,
            museum_span_loss_weight=museum_span_loss_weight,
            artifact_span_loss_weight=artifact_span_loss_weight,
            entity_span_loss_min_context_bytes=entity_span_loss_min_context_bytes,
            museum_aux_loss_weight=museum_aux_loss_weight,
            artifact_aux_loss_weight=artifact_aux_loss_weight,
            entity_aux_loss_min_context_bytes=entity_aux_loss_min_context_bytes,
            museum_hint_injection_weight=museum_hint_injection_weight,
            artifact_hint_injection_weight=artifact_hint_injection_weight,
            entity_hint_injection_min_context_bytes=entity_hint_injection_min_context_bytes,
            entity_hint_use_gold_labels_during_training=entity_hint_use_gold_labels_during_training,
            slot_decoder_loss_weight=slot_decoder_loss_weight,
            slot_decoder_logit_bias=slot_decoder_logit_bias,
            slot_decoder_min_context_bytes=slot_decoder_min_context_bytes,
            evidence_window_count=evidence_window_count,
            evidence_loss_weight=evidence_loss_weight,
            evidence_hint_weight=evidence_hint_weight,
            evidence_min_context_bytes=evidence_min_context_bytes,
            model_type=model_type,
            local_context_size=local_context_size,
            local_context_mode=local_context_mode,
            return_model=True,
        )
        trained_model = train_result.pop("model")
        validation_eval = evaluate_case_pool(trained_model, validation_cases, device)
        test_eval = evaluate_case_pool(trained_model, test_cases, device)
        config = {
            **train_result["config"],
            "validation_dataset_size": len(validation_cases),
            "validation_dataset_seed": validation_dataset_seed,
            "pool_split_mode": "museum_artifact_held_out",
            "pair_split_seed": pair_split_seed,
            "generalization_score_mode": generalization_score_mode,
            "train_pair_count": len(train_pairs),
            "test_dataset_size": len(test_cases),
            "test_dataset_seed": test_dataset_seed,
            "validation_pair_count": len(validation_pairs),
            "test_pair_count": len(test_pairs),
        }
        result = {
            "config": config,
            "history": train_result["history"],
            "validation_pool_evaluation": validation_eval,
            "test_pool_evaluation": test_eval,
        }
        search_results.append(summarize_generalization_search_result(result))
        print(
            f"Validation | gen_exact={validation_eval['generation_exact_match_rate']*100:.2f}% "
            f"| tf_exact={validation_eval['teacher_forced_exact_match_rate']*100:.2f}% "
            f"| gen_seq_acc={validation_eval['generation_mean_sequence_accuracy']*100:.2f}% "
            f"| tf_seq_acc={validation_eval['teacher_forced_mean_sequence_accuracy']*100:.2f}%"
        )
        print(
            f"Test       | gen_exact={test_eval['generation_exact_match_rate']*100:.2f}% "
            f"| tf_exact={test_eval['teacher_forced_exact_match_rate']*100:.2f}% "
            f"| gen_seq_acc={test_eval['generation_mean_sequence_accuracy']*100:.2f}% "
            f"| tf_seq_acc={test_eval['teacher_forced_mean_sequence_accuracy']*100:.2f}%"
        )
        if best_result is None or score_generalization_result(
            result,
            score_mode=generalization_score_mode,
        ) > score_generalization_result(
            best_result,
            score_mode=generalization_score_mode,
        ):
            best_result = result

        del trained_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    search_results.sort(
        key=lambda item: score_generalization_summary(
            item,
            score_mode=generalization_score_mode,
        ),
        reverse=True,
    )

    validation_tail_error_analysis = build_tail_error_analysis(best_result["validation_pool_evaluation"])
    test_tail_error_analysis = build_tail_error_analysis(best_result["test_pool_evaluation"])

    if reports_dir is not None:
        save_json_retrieval_generalization_reports(
            best_result["config"],
            best_result["history"],
            best_result["validation_pool_evaluation"],
            best_result["test_pool_evaluation"],
            search_results,
            validation_tail_error_analysis,
            test_tail_error_analysis,
            Path(reports_dir),
        )

    print("\n=== JSON Retrieval Generalization Result ===")
    print(
        f"Best Config | epochs={best_result['config']['epochs']} | kr={best_result['config']['kr']} | chunk_size={best_result['config']['chunk_size']} "
        f"| lr={best_result['config']['lr']:.0e} | warmup_ratio={best_result['config']['warmup_ratio']} "
        f"| scheduled_sampling_max_ratio={best_result['config']['scheduled_sampling_max_ratio']}"
    )
    print(f"Selection Score Mode: {generalization_score_mode}")
    print(
        f"Validation Generation Exact Match Rate: "
        f"{best_result['validation_pool_evaluation']['generation_exact_match_rate']*100:.2f}%"
    )
    print(
        f"Validation Teacher-Forced Exact Match Rate: "
        f"{best_result['validation_pool_evaluation']['teacher_forced_exact_match_rate']*100:.2f}%"
    )
    print(
        f"Test Generation Exact Match Rate: "
        f"{best_result['test_pool_evaluation']['generation_exact_match_rate']*100:.2f}%"
    )
    print(
        f"Test Teacher-Forced Exact Match Rate: "
        f"{best_result['test_pool_evaluation']['teacher_forced_exact_match_rate']*100:.2f}%"
    )

    return {
        "config": best_result["config"],
        "history": best_result["history"],
        "validation_pool_evaluation": best_result["validation_pool_evaluation"],
        "test_pool_evaluation": best_result["test_pool_evaluation"],
        "validation_tail_error_analysis": validation_tail_error_analysis,
        "test_tail_error_analysis": test_tail_error_analysis,
        "search_results": search_results,
    }


if __name__ == "__main__":
    run_json_retrieval_test(reports_dir=Path(__file__).resolve().parents[1] / "reports")
