import json
import random
from pathlib import Path
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from report_utils import ensure_reports_dir, write_json, write_markdown
from toy_task_associative_recall import DSRAModel


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


def load_json_retrieval_case(
    input_path="test_input.json",
    metadata_path="test_metadata.json",
):
    base_dir = Path(__file__).resolve().parent
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


def generate_random_json_retrieval_case(reference_case, rng, target_total_bytes=None):
    metadata = reference_case["metadata"]
    total_bytes = int(target_total_bytes or metadata.get("target_total_bytes", len(reference_case["sample_bytes"])))
    museum = rng.choice(MUSEUM_NAMES)
    artifact = rng.choice(ARTIFACT_NAMES)
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
            "question": question,
            "expected_answer_text": expected_answer_text,
            "expected_answer_bytes": list(expected_answer_bytes),
        },
        "question_bytes": question.encode("ascii"),
        "expected_answer_bytes": expected_answer_bytes,
    }


def build_random_training_case_pool(reference_case, dataset_size, seed):
    rng = random.Random(seed)
    return [
        generate_random_json_retrieval_case(reference_case, rng)
        for _ in range(max(1, dataset_size))
    ]


def build_curriculum_context(case, context_bytes):
    sample_bytes = case["sample_bytes"]
    total_bytes = len(sample_bytes)
    if context_bytes >= total_bytes:
        return sample_bytes

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

    return sample_bytes[start:end]


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


def greedy_generate_answer(model, prompt_tokens, answer_len, device):
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    raw_prompt_emb = model.embedding(prompt_tensor)
    contextual_prompt_emb = model.build_causal_context(raw_prompt_emb)

    S_prev = None
    bypass_kv = None
    S_time_prev = None
    last_out = None
    chunk_idx = 0

    with torch.no_grad():
        for start in range(0, contextual_prompt_emb.shape[1], model.chunk_size):
            chunk = contextual_prompt_emb[:, start:start + model.chunk_size, :]
            out_chunk, S_prev, bypass_kv, S_time_prev = model.dsra(
                chunk,
                S_prev=S_prev,
                bypass_kv=bypass_kv,
                S_time_prev=S_time_prev,
                chunk_idx=chunk_idx,
            )
            last_out = out_chunk[:, -1:, :]
            chunk_idx += 1

        last_out = model.norm(last_out)
        next_logits = model.out_proj(last_out)[:, -1, :]

    generated_answer = []
    raw_history = deque([raw_prompt_emb[:, idx:idx + 1, :] for idx in range(raw_prompt_emb.shape[1])], maxlen=4)

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
            step_input = torch.stack(list(raw_history), dim=0).sum(dim=0)
            out_t, S_prev, bypass_kv = model.dsra.forward_step(
                step_input,
                S_prev=S_prev,
                kv_cache=bypass_kv,
            )
            out_t = model.norm(out_t)
            next_logits = model.out_proj(out_t)[:, -1, :]

    return generated_answer


def evaluate_teacher_forced(logits, Y, case):
    logits_target, targets = collect_answer_targets(logits, Y)
    predicted_tokens = predict_byte_tokens(logits_target).tolist()
    expected_tokens = list(case["expected_answer_bytes"])
    predicted_bytes = bytes(predicted_tokens)
    expected_bytes = case["expected_answer_bytes"]
    metrics = compute_sequence_metrics(predicted_tokens, expected_tokens)

    return {
        "predicted_tokens": predicted_tokens,
        "predicted_text": predicted_bytes.decode("utf-8", errors="replace"),
        "exact_byte_match": predicted_bytes == expected_bytes,
        "token_acc": metrics["sequence_accuracy"],
        **metrics,
    }


def evaluate_generation(model, case, device, context_bytes=None):
    sample_bytes = build_curriculum_context(case, context_bytes) if context_bytes is not None else case["sample_bytes"]
    sample_tokens = list(sample_bytes)
    question_tokens = list(case["question_bytes"])
    expected_answer_tokens = list(case["expected_answer_bytes"])
    prompt_tokens = sample_tokens + [QUESTION_TOKEN_ID] + question_tokens + [ANSWER_START_TOKEN_ID]

    predicted_tokens = greedy_generate_answer(
        model=model,
        prompt_tokens=prompt_tokens,
        answer_len=len(expected_answer_tokens),
        device=device,
    )
    predicted_bytes = bytes(predicted_tokens)
    expected_bytes = case["expected_answer_bytes"]
    predicted_text = predicted_bytes.decode("utf-8", errors="replace")
    expected_text = expected_bytes.decode("utf-8", errors="replace")
    metrics = compute_sequence_metrics(predicted_tokens, expected_answer_tokens)

    return {
        "predicted_tokens": predicted_tokens,
        "predicted_text": predicted_text,
        "expected_text": expected_text,
        "exact_byte_match": predicted_bytes == expected_bytes,
        "exact_text_match": predicted_text == expected_text,
        **metrics,
    }


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
        "## Generation Evaluation",
        f"- Exact Byte Match: `{generation_eval['exact_byte_match']}`",
        f"- Exact Text Match: `{generation_eval['exact_text_match']}`",
        f"- Sequence Accuracy: `{generation_eval['sequence_accuracy']*100:.2f}%`",
        f"- Prefix Match Length: `{generation_eval['prefix_match_length']}`",
        f"- First Mismatch Index: `{generation_eval['first_mismatch_index']}`",
        f"- First Mismatch Expected Byte: `{generation_eval['first_mismatch_expected_byte']}`",
        f"- First Mismatch Predicted Byte: `{generation_eval['first_mismatch_predicted_byte']}`",
        "",
        "## Predicted Answer",
        generation_eval["predicted_text"],
        "",
        "## Training Config",
        f"- Device: `{config['device']}`",
        f"- Epochs: `{config['epochs']}`",
        f"- Eval Interval: `{config['eval_interval']}`",
        f"- Dim: `{config['dim']}`",
        f"- K: `{config['K']}`",
        f"- kr: `{config['kr']}`",
        f"- Chunk Size: `{config['chunk_size']}`",
        f"- Learning Rate: `{config['lr']}`",
        f"- Warmup Ratio: `{config['warmup_ratio']}`",
        f"- Scheduled Sampling Max Ratio: `{config['scheduled_sampling_max_ratio']}`",
        f"- Training Mode: `{config['training_mode']}`",
        f"- Train Dataset Size: `{config['train_dataset_size']}`",
        f"- Train Dataset Seed: `{config['train_dataset_seed']}`",
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
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    curriculum_plan = build_curriculum_plan(case, epochs)
    training_cases = build_random_training_case_pool(
        reference_case=case,
        dataset_size=train_dataset_size,
        seed=train_dataset_seed,
    )
    training_case_sampler = random.Random(train_dataset_seed)

    model = DSRAModel(
        vocab_size=VOCAB_SIZE,
        dim=dim,
        K=K,
        kr=kr,
        chunk_size=chunk_size,
        pe_mode="none",
        use_orthogonal_update=True,
        use_bypass=True,
    ).to(device)
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
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

    history = []
    best_exact_match = False

    global_epoch = 0
    for stage_idx, stage in enumerate(curriculum_plan, start=1):
        stage_preview_case = training_cases[0]
        preview_X, _ = build_training_example(stage_preview_case, context_bytes=stage["context_bytes"])
        print(
            f"\n--- Curriculum Stage {stage_idx}/{len(curriculum_plan)} | "
            f"context={stage['name']} | approx_train_seq_len={preview_X.shape[1]} "
            f"| stage_epochs={stage['epochs']} | train_cases={len(training_cases)} ---"
        )

        for stage_epoch in range(1, stage["epochs"] + 1):
            global_epoch += 1
            train_case = training_case_sampler.choice(training_cases)
            X, Y = build_training_example(train_case, context_bytes=stage["context_bytes"])
            X, Y = X.to(device), Y.to(device)
            model.train()
            optimizer.zero_grad()
            current_sampling_ratio = scheduled_sampling_ratio(global_epoch - 1)
            sampled_token_count = 0
            teacher_forced_logits = None

            if current_sampling_ratio > 0.0:
                with torch.no_grad():
                    teacher_forced_logits = model(X)
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

            logits = model(train_X)
            logits_target, targets = collect_answer_targets(logits, Y)
            loss = criterion(logits_target, targets)
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
                best_exact_match = best_exact_match or teacher_forced["exact_byte_match"]
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
                    f"| Teacher Forced Exact Match: {teacher_forced['exact_byte_match']}"
                )

            del logits, logits_target, targets, loss, train_X, teacher_forced_logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    full_X, full_Y = build_training_example(case)
    full_X, full_Y = full_X.to(device), full_Y.to(device)
    with torch.no_grad():
        full_logits = model(full_X)
    final_teacher_forced = evaluate_teacher_forced(full_logits, full_Y, case)
    final_generation = evaluate_generation(model, case, device)
    config = {
        "device": str(device),
        "epochs": epochs,
        "eval_interval": eval_interval,
        "dim": dim,
        "K": K,
        "kr": kr,
        "chunk_size": chunk_size,
        "lr": lr,
        "warmup_ratio": warmup_ratio,
        "scheduled_sampling_max_ratio": scheduled_sampling_max_ratio,
        "train_dataset_size": train_dataset_size,
        "train_dataset_seed": train_dataset_seed,
        "training_mode": "random_case_pool",
        "curriculum_labels": " -> ".join(stage["name"] for stage in curriculum_plan),
        "best_exact_match": best_exact_match or final_generation["exact_byte_match"],
    }

    return {
        "config": config,
        "history": history,
        "teacher_forced_evaluation": final_teacher_forced,
        "evaluation": final_generation,
    }


def run_json_retrieval_test(
    input_path="test_input.json",
    metadata_path="test_metadata.json",
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
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    case = load_json_retrieval_case(input_path=input_path, metadata_path=metadata_path)
    curriculum_plan = build_curriculum_plan(case, epochs)

    kr_grid = kr_grid or DEFAULT_KR_GRID
    chunk_size_grid = chunk_size_grid or DEFAULT_CHUNK_SIZE_GRID
    lr_grid = lr_grid or DEFAULT_LR_GRID
    warmup_ratio_grid = warmup_ratio_grid or DEFAULT_WARMUP_RATIO_GRID
    scheduled_sampling_max_ratio_grid = (
        scheduled_sampling_max_ratio_grid or DEFAULT_SCHEDULED_SAMPLING_MAX_RATIO_GRID
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
        f"| answer_len={len(case['expected_answer_bytes'])} | dim={dim} | K={K} | epochs={epochs}"
    )
    print(
        f"Training Data | mode=random_case_pool | dataset_size={train_dataset_size} "
        f"| dataset_seed={train_dataset_seed}"
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


if __name__ == "__main__":
    run_json_retrieval_test(reports_dir=Path(__file__).resolve().parent / "reports")
