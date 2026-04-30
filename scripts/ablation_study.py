import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.dsra.report_utils import build_ablation_markdown, ensure_reports_dir, write_json, write_markdown
from scripts.toy_task_associative_recall import (
    MHDSRA2Model,
    build_fixed_associative_mapping,
    generate_associative_recall_data,
)


DEFAULT_SEED = 42
CURRICULUM_STAGES = [
    {"name": "warmup-128", "seq_len": 128, "num_pairs": 2, "steps": 120},
    {"name": "warmup-256", "seq_len": 256, "num_pairs": 4, "steps": 160},
    {"name": "warmup-512", "seq_len": 512, "num_pairs": 8, "steps": 200},
    {"name": "target-1024", "seq_len": 1024, "num_pairs": 20, "steps": 260},
]
LR_GRID = [1e-3, 5e-4, 1e-4]
DATA_MODE = "fixed_mapping"
FIXED_MAPPING_NOISE_POOL = 24


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    X, Y = batch
    return X.to(device), Y.to(device)


def collect_target_logits(logits, Y):
    target_indices = (Y != 0).nonzero(as_tuple=True)
    logits_target = logits[target_indices[0], target_indices[1], :]
    targets = Y[target_indices[0], target_indices[1]]
    return logits_target, targets


def evaluate_model(model, eval_batches, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in eval_batches:
            X, Y = move_batch_to_device(batch, device)
            logits = model(X)
            logits_target, targets = collect_target_logits(logits, Y)
            loss = criterion(logits_target, targets)
            preds = logits_target.argmax(dim=-1)

            total_loss += loss.item() * targets.size(0)
            total_correct += (preds == targets).sum().item()
            total_examples += targets.size(0)

    avg_loss = total_loss / max(total_examples, 1)
    avg_acc = total_correct / max(total_examples, 1)
    return avg_loss, avg_acc


def build_mapping_for_stage(vocab_size, num_pairs):
    return build_fixed_associative_mapping(
        vocab_size=vocab_size,
        num_pairs=num_pairs,
        seed=DEFAULT_SEED + num_pairs,
        noise_pool_size=FIXED_MAPPING_NOISE_POOL,
    )


def generate_batch_from_mode(batch_size, seq_len, vocab_size, num_pairs, data_mode, mapping=None):
    if data_mode == "fixed_mapping":
        if mapping is None:
            raise ValueError("Fixed mapping mode requires a mapping.")
        return generate_associative_recall_data(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            num_pairs=num_pairs,
            fixed_pairs=mapping["pairs"],
            fixed_noise_tokens=mapping["noise_tokens"],
        )
    return generate_associative_recall_data(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_pairs=num_pairs,
    )


def build_eval_set(seed, batch_size, num_batches, seq_len, vocab_size, num_pairs, data_mode):
    local_rng_state = random.getstate()
    torch_rng_state = torch.random.get_rng_state()
    set_seed(seed)
    mapping = build_mapping_for_stage(vocab_size, num_pairs) if data_mode == "fixed_mapping" else None

    eval_batches = [
        generate_batch_from_mode(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            num_pairs=num_pairs,
            data_mode=data_mode,
            mapping=mapping,
        )
        for _ in range(num_batches)
    ]

    random.setstate(local_rng_state)
    torch.random.set_rng_state(torch_rng_state)
    return eval_batches, mapping


def train_with_curriculum(
    model,
    device,
    lr,
    criterion,
    train_batch_size,
    vocab_size,
    eval_batches,
    warmup_steps=40,
    data_mode="fixed_mapping",
    print_prefix="",
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min((step + 1) / max(warmup_steps, 1), 1.0),
    )

    global_step = 0
    best_eval_acc = 0.0
    best_eval_loss = float("inf")
    history = []

    for stage_idx, stage in enumerate(CURRICULUM_STAGES, start=1):
        stage_mapping = build_mapping_for_stage(vocab_size, stage["num_pairs"]) if data_mode == "fixed_mapping" else None
        print(
            f"{print_prefix}Stage {stage_idx}/{len(CURRICULUM_STAGES)} "
            f"| {stage['name']} | seq_len={stage['seq_len']} | num_pairs={stage['num_pairs']} | steps={stage['steps']}"
        )
        model.train()

        for step in range(stage["steps"]):
            X, Y = generate_batch_from_mode(
                batch_size=train_batch_size,
                seq_len=stage["seq_len"],
                vocab_size=vocab_size,
                num_pairs=stage["num_pairs"],
                data_mode=data_mode,
                mapping=stage_mapping,
            )
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            logits_target, targets = collect_target_logits(logits, Y)
            loss = criterion(logits_target, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % 40 == 0 or (stage_idx == 1 and step == 0):
                eval_loss, eval_acc = evaluate_model(model, eval_batches, criterion, device)
                best_eval_acc = max(best_eval_acc, eval_acc)
                best_eval_loss = min(best_eval_loss, eval_loss)
                history.append(
                    {
                        "step": global_step,
                        "stage": stage["name"],
                        "eval_loss": eval_loss,
                        "eval_acc": eval_acc,
                    }
                )
                print(
                    f"{print_prefix}Step {global_step:4d} | "
                    f"train_loss={loss.item():.4f} | eval_loss={eval_loss:.4f} | eval_acc={eval_acc*100:6.2f}%"
                )

    final_eval_loss, final_eval_acc = evaluate_model(model, eval_batches, criterion, device)
    return {
        "best_eval_acc": best_eval_acc,
        "best_eval_loss": best_eval_loss,
        "final_eval_acc": final_eval_acc,
        "final_eval_loss": final_eval_loss,
        "history": history,
    }


def build_model(config, vocab_size, dim, K, kr, chunk_size, device):
    """Build an MHDSRA2 model for one ablation configuration.

    中文说明:
    - 调用方 / Called by: `run_ablation`
    - 调用对象 / Calls: `MHDSRA2Model`
    - 作用 / Purpose: 将消融研究从旧 DSRA 机制迁移到 MHDSRA2 的 local/retrieval/window 配置
    - 变量 / Variables:
      `config` 包含 `use_local/use_retrieval/local_window`, 其余参数为模型结构与设备
    - 接入 / Integration: 新增 MHDSRA2 消融项时扩展 `config` 字段即可
    - 错误处理 / Error handling: 非法配置由 MHDSRA2 底层抛出
    - 关键词 / Keywords:
      ablation|mhdsra2|build_model|local|retrieval|window|toy_task|config|migration|消融
    """
    return MHDSRA2Model(
        vocab_size=vocab_size,
        dim=dim,
        K=K,
        kr=kr,
        chunk_size=chunk_size,
        use_local=config["use_local"],
        use_retrieval=config["use_retrieval"],
        local_window=config.get("local_window"),
    ).to(device)


def run_ablation(
    name,
    config,
    device,
    eval_batches,
    vocab_size=100,
    dim=64,
    K=64,
    kr=8,
    chunk_size=128,
    train_batch_size=32,
    data_mode="fixed_mapping",
):
    print(f"\n--- Starting Ablation Study: {name} ---")
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    best_run = None

    for lr in LR_GRID:
        set_seed(DEFAULT_SEED)
        model = build_model(config, vocab_size, dim, K, kr, chunk_size, device)
        run_result = train_with_curriculum(
            model=model,
            device=device,
            lr=lr,
            criterion=criterion,
            train_batch_size=train_batch_size,
            vocab_size=vocab_size,
            eval_batches=eval_batches,
            warmup_steps=40,
            data_mode=data_mode,
            print_prefix=f"[{name} | lr={lr:.0e}] ",
        )
        run_result["lr"] = lr

        if best_run is None or run_result["final_eval_acc"] > best_run["final_eval_acc"]:
            best_run = run_result

    print(
        f"[{name}] Best Run | lr={best_run['lr']:.0e} | "
        f"final_eval_acc={best_run['final_eval_acc']*100:.2f}% | "
        f"best_eval_acc={best_run['best_eval_acc']*100:.2f}% | "
        f"final_eval_loss={best_run['final_eval_loss']:.4f}"
    )
    return best_run


def save_ablation_reports(results, reports_dir):
    reports_dir = ensure_reports_dir(reports_dir)
    write_markdown(reports_dir / "ablation_summary.md", build_ablation_markdown(results))
    write_json(reports_dir / "ablation_summary.json", results)


def main(reports_dir=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    set_seed(DEFAULT_SEED)

    vocab_size = 100
    dim = 64
    chunk_size = 128
    K = 64
    kr = 8
    train_batch_size = 64
    eval_batch_size = 128
    eval_num_batches = 16
    eval_seed = DEFAULT_SEED + 1
    target_stage = CURRICULUM_STAGES[-1]

    eval_batches, eval_mapping = build_eval_set(
        seed=eval_seed,
        batch_size=eval_batch_size,
        num_batches=eval_num_batches,
        seq_len=target_stage["seq_len"],
        vocab_size=vocab_size,
        num_pairs=target_stage["num_pairs"],
        data_mode=DATA_MODE,
    )

    print(
        f"Fixed validation set | mode={DATA_MODE} | batches={eval_num_batches} | batch_size={eval_batch_size} | "
        f"seq_len={target_stage['seq_len']} | num_pairs={target_stage['num_pairs']}"
    )
    if eval_mapping is not None:
        print(
            f"Validation mapping | pairs={len(eval_mapping['pairs'])} | noise_pool={len(eval_mapping['noise_tokens'])}"
        )

    baseline_config = {
        "use_local": True,
        "use_retrieval": True,
        "local_window": chunk_size,
    }
    baseline_result = run_ablation(
        name="Full MHDSRA2 (local+slot)",
        config=baseline_config,
        device=device,
        eval_batches=eval_batches,
        vocab_size=vocab_size,
        dim=dim,
        K=K,
        kr=kr,
        chunk_size=chunk_size,
        train_batch_size=train_batch_size,
        data_mode=DATA_MODE,
    )

    results = {"Full MHDSRA2 (local+slot)": baseline_result}
    baseline_threshold = 0.9

    if baseline_result["final_eval_acc"] < baseline_threshold:
        print(
            "\nBaseline has not learned the task reliably yet; running the minimal MHDSRA2 "
            "local/window ablations and skipping wider variants."
        )
        ablations = [
            ("MHDSRA2 Slot Only (no local)", {"use_local": False, "use_retrieval": True, "local_window": chunk_size}),
            ("MHDSRA2 Narrow Local Window", {"use_local": True, "use_retrieval": True, "local_window": max(1, chunk_size // 4)}),
        ]
    else:
        ablations = [
            ("MHDSRA2 Slot Only (no local)", {"use_local": False, "use_retrieval": True, "local_window": chunk_size}),
            ("MHDSRA2 Narrow Local Window", {"use_local": True, "use_retrieval": True, "local_window": max(1, chunk_size // 4)}),
            ("MHDSRA2 Wide Local Window", {"use_local": True, "use_retrieval": True, "local_window": chunk_size * 2}),
            ("MHDSRA2 Retrieval Flag Off", {"use_local": True, "use_retrieval": False, "local_window": chunk_size}),
        ]

    for name, config in ablations:
        results[name] = run_ablation(
            name=name,
            config=config,
            device=device,
            eval_batches=eval_batches,
            vocab_size=vocab_size,
            dim=dim,
            K=K,
            kr=kr,
            chunk_size=chunk_size,
            train_batch_size=train_batch_size,
            data_mode=DATA_MODE,
        )

    print("\n=== Ablation Study Results (Fixed Validation Set) ===")
    for name, result in results.items():
        print(
            f"{name} | lr={result['lr']:.0e} | "
            f"final_eval_acc={result['final_eval_acc']*100:.2f}% | "
            f"best_eval_acc={result['best_eval_acc']*100:.2f}% | "
            f"final_eval_loss={result['final_eval_loss']:.4f}"
        )

    if reports_dir is not None:
        save_ablation_reports(results, Path(reports_dir))

    return results


if __name__ == '__main__':
    main()
