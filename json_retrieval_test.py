import json
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


def summarize_search_result(result):
    return {
        "kr": result["config"]["kr"],
        "chunk_size": result["config"]["chunk_size"],
        "lr": result["config"]["lr"],
        "warmup_ratio": result["config"]["warmup_ratio"],
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
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    curriculum_plan = build_curriculum_plan(case, epochs)

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
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

    history = []
    best_exact_match = False

    global_epoch = 0
    for stage_idx, stage in enumerate(curriculum_plan, start=1):
        X, Y = build_training_example(case, context_bytes=stage["context_bytes"])
        X, Y = X.to(device), Y.to(device)
        print(
            f"\n--- Curriculum Stage {stage_idx}/{len(curriculum_plan)} | "
            f"context={stage['name']} | train_seq_len={X.shape[1]} | stage_epochs={stage['epochs']} ---"
        )

        for stage_epoch in range(1, stage["epochs"] + 1):
            global_epoch += 1
            model.train()
            optimizer.zero_grad()
            logits = model(X)
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
                teacher_forced = evaluate_teacher_forced(logits, Y, case)
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
                    }
                )
                print(
                    f"Epoch {global_epoch:4d} | Stage: {stage['name']:>3} | Loss: {loss.item():.4f} "
                    f"| Train Token Acc: {train_acc*100:5.1f}% "
                    f"| Seq Acc: {teacher_forced['sequence_accuracy']*100:5.1f}% "
                    f"| Prefix Match: {teacher_forced['prefix_match_length']:3d} "
                    f"| LR: {optimizer.param_groups[0]['lr']:.2e} "
                    f"| Teacher Forced Exact Match: {teacher_forced['exact_byte_match']}"
                )
                if teacher_forced["exact_byte_match"] and stage["context_bytes"] == len(case["sample_bytes"]):
                    print("JSON retrieval task solved successfully!")
                    break

            del logits, logits_target, targets, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if best_exact_match and stage["context_bytes"] == len(case["sample_bytes"]):
            break

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
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    case = load_json_retrieval_case(input_path=input_path, metadata_path=metadata_path)
    curriculum_plan = build_curriculum_plan(case, epochs)

    kr_grid = kr_grid or DEFAULT_KR_GRID
    chunk_size_grid = chunk_size_grid or DEFAULT_CHUNK_SIZE_GRID
    lr_grid = lr_grid or DEFAULT_LR_GRID
    warmup_ratio_grid = warmup_ratio_grid or DEFAULT_WARMUP_RATIO_GRID
    search_space = [
        (kr, chunk_size, lr, warmup_ratio_value)
        for kr in kr_grid
        for chunk_size in chunk_size_grid
        for lr in lr_grid
        for warmup_ratio_value in warmup_ratio_grid
    ]

    print("\n--- Running JSON File Retrieval Test ---")
    print(f"Using device: {device}")
    print(
        f"Config | full_seq_len={len(case['sample_bytes']) + 1 + len(case['question_bytes']) + 1 + len(case['expected_answer_bytes']) - 1} "
        f"| answer_len={len(case['expected_answer_bytes'])} | dim={dim} | K={K} | epochs={epochs}"
    )
    print(
        "Curriculum | "
        + " -> ".join(f"{stage['name']}({stage['epochs']})" for stage in curriculum_plan)
    )
    print(
        "Grid Search | "
        f"kr={kr_grid} | chunk_size={chunk_size_grid} | lr={lr_grid} | warmup_ratio={warmup_ratio_grid}"
    )

    best_result = None
    search_results = []
    for trial_idx, (kr, chunk_size, lr, warmup_ratio_value) in enumerate(search_space, start=1):
        print(
            f"\n=== Search Trial {trial_idx}/{len(search_space)} | "
            f"kr={kr} | chunk_size={chunk_size} | lr={lr:.0e} | warmup_ratio={warmup_ratio_value} ==="
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
        f"| lr={best_result['config']['lr']:.0e} | warmup_ratio={best_result['config']['warmup_ratio']}"
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
