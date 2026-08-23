"""长程检索卖点恢复攻关脚本 (Target: 80%+ Accuracy on CUDA:0).

中文说明:
- 调用方 / Called by: python scripts/breakthrough_niah_80plus.py
- 作用: 在 GPU cuda:0 上满载运行多任务对比引导端到端长程检索训练，攻克 80%+ 准确率。
- 保证严格防作弊: eval 阶段关闭全部 aux loss，纯前向端到端测试。
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dsra.report_utils import save_figure, write_json, write_markdown  # noqa: E402
from scripts.needle_in_haystack_test import (  # noqa: E402
    build_niah_model,
    compute_query_evidence_alignment_loss,
    compute_retrieval_projection_contrastive_loss,
    evaluate_niah_depths,
    extract_query_positions_and_targets,
    generate_haystack_with_needle,
    seed_all,
)


def run_breakthrough_training(
    seq_len: int = 16384,
    epochs: int = 120,
    batch_size: int = 8,
    dim: int = 128,
    device_name: str = "cuda:0",
    seed: int = 20260506,
    lr: float = 1e-3,
    target_acc: float = 0.80,
):
    """运行长程检索 80%+ 突破训练."""
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"=== Starting NIAH Breakthrough Training on {device} ===")
    print(f"Config: seq_len={seq_len}, epochs={epochs}, batch_size={batch_size}, dim={dim}, seed={seed}")

    seed_all(seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()

    # 关键机制开启: 邻居召回 + 门控偏置 + Top8截断 + 状态不detach
    override = {
        "use_retrieval": True,
        "retrieval_neighbor_span": 1,
        "retrieval_neighbor_direction": "right",
        "retrieval_query_pooling": "mean",
        "retrieval_attention_topk": 8,
        "retrieval_quality_gate_bias": 2.0,
        "detach_state": False,
    }

    model = build_niah_model(
        device=device,
        vocab_size=100,
        dim=dim,
        num_layers=2,
        K=64,
        kr=8,
        chunk_size=1024,
        use_retrieval=True,
        mhdsra2_config_override=override,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(ignore_index=0)

    depths_cycle = [0.1, 0.3, 0.5, 0.7, 0.9]
    history = []
    best_eval_acc = 0.0
    best_step = 0
    t0 = time.time()

    for step in range(epochs):
        model.train()
        depth = depths_cycle[step % len(depths_cycle)]
        X, Y, needle_positions = generate_haystack_with_needle(batch_size, seq_len, 100, depth)
        qpos, targets = extract_query_positions_and_targets(X, Y, device)
        needles = targets

        opt.zero_grad()

        # 训练前向: 收集主预测、hidden 与检索 aux
        retrieval_evidence_positions = needle_positions + 1
        logits, hidden_query, retrieval_aux = model.forward_selected_logits(
            X,
            qpos,
            return_hidden=True,
            return_aux=True,
            return_retrieval_projection_aux=True,
            train_retrieval_evidence_positions=retrieval_evidence_positions,
        )

        loss_main = crit(logits, targets)
        loss = loss_main

        # 辅助 1: Needle 局部位置监督
        needle_val_pos = (needle_positions + 1).to(device)
        logits_needle = model.forward_selected_logits(X, needle_val_pos)
        loss_needle = crit(logits_needle, needles)
        loss = loss + 0.5 * loss_needle

        # 辅助 2: Query-Key 投影对比损失 (直接打通冷启动)
        loss_contrast, contrast_metrics = compute_retrieval_projection_contrastive_loss(
            retrieval_aux,
            retrieval_evidence_positions,
            device=device,
            temperature=0.1,
        )
        if contrast_metrics.get("available", False):
            loss = loss + 1.0 * loss_contrast

        # 辅助 3: 隐藏状态与目标嵌入余弦对齐
        loss_align, _ = compute_query_evidence_alignment_loss(
            hidden_query,
            needles,
            model.embedding,
            detach_evidence=True,
        )
        loss = loss + 0.5 * loss_align

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        preds = logits.argmax(dim=-1)
        step_acc = (preds == targets).float().mean().item()

        # 评估 (每 10 步或最后一步)
        if (step + 1) % 10 == 0 or step == epochs - 1:
            eval_res = evaluate_niah_depths(
                model,
                seq_len,
                device,
                vocab_size=100,
                batch_size=2,
                criterion=crit,
                batches_per_depth=4,
            )
            eval_acc = eval_res["mean_accuracy"]
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                best_step = step + 1

            elapsed = time.time() - t0
            print(
                f"[Step {step+1:3d}/{epochs}] loss={loss.item():.4f} (main={loss_main.item():.4f}, "
                f"needle={loss_needle.item():.4f}, contrast={loss_contrast.item():.4f}) | "
                f"train_acc={step_acc*100:5.1f}% | eval_acc={eval_acc*100:5.1f}% "
                f"(best={best_eval_acc*100:5.1f}% @ step {best_step}) | time={elapsed:.1f}s"
            )
            history.append({
                "step": step + 1,
                "loss": float(loss.item()),
                "loss_main": float(loss_main.item()),
                "loss_needle": float(loss_needle.item()),
                "loss_contrast": float(loss_contrast.item()),
                "train_acc": float(step_acc),
                "eval_acc": float(eval_acc),
            })

            if eval_acc >= target_acc:
                print(
                    f">>> GOAL ACHIEVED! Reached {eval_acc*100:.1f}% accuracy "
                    f"(target: {target_acc*100:.1f}%) at step {step+1}!"
                )
                break

    # 保存报告与图表
    out_dir = PROJECT_ROOT / "docs" / "reports" / "verify_technical_report" / "breakthrough_80plus"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "target_accuracy": target_acc,
        "best_eval_accuracy": best_eval_acc,
        "best_step": best_step,
        "success": best_eval_acc >= target_acc,
        "total_time_sec": time.time() - t0,
        "config": {
            "seq_len": seq_len,
            "epochs": epochs,
            "batch_size": batch_size,
            "dim": dim,
            "device": str(device),
        },
        "history": history,
    }
    write_json(out_dir / "breakthrough_summary.json", summary)

    md_lines = [
        "# 长程检索 80%+ 攻关验证报告",
        "",
        f"- 目标准确率: `{target_acc*100:.1f}%`",
        f"- 实测最佳准确率: `**{best_eval_acc*100:.1f}%**` (第 {best_step} 轮)",
        f"- 攻关判定: `**{'PASSED (成功达成)' if summary['success'] else 'FAILED'}**`",
        f"- 硬件与耗时: `{device}` | 总耗时 `{summary['total_time_sec']:.1f}s`",
        "",
        "## 训练与评估曲线",
        "",
        "| 步数 (Step) | 总 Loss | 主预测 Loss | 对比检索 Loss | 训练 Acc | 纯端到端 Eval Acc |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|",
        *[
            f"| {h['step']} | {h['loss']:.4f} | {h['loss_main']:.4f} | {h['loss_contrast']:.4f} | "
            f"{h['train_acc']*100:.1f}% | **{h['eval_acc']*100:.1f}%** |"
            for h in history
        ],
    ]
    write_markdown(out_dir / "breakthrough_summary.md", md_lines)

    # 绘制高质量曲线图
    fig, ax1 = plt.subplots(figsize=(9, 5))
    steps = [h["step"] for h in history]
    eval_accs = [h["eval_acc"] * 100 for h in history]
    losses = [h["loss_main"] for h in history]

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss (Main CE)", color="#e53e3e")
    ax1.plot(steps, losses, color="#e53e3e", marker="o", label="Main CE Loss")
    ax1.tick_params(axis="y", labelcolor="#e53e3e")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Eval Accuracy (%)", color="#2b6cb0")
    ax2.plot(steps, eval_accs, color="#2b6cb0", marker="s", lw=2, label="Eval Accuracy (%)")
    ax2.axhline(target_acc * 100, color="g", ls="--", label=f"Target ({target_acc*100:.0f}%)")
    ax2.tick_params(axis="y", labelcolor="#2b6cb0")
    ax2.set_ylim(0, 105)

    plt.title(f"MHDSRA2 NIAH 80%+ Breakthrough Curve (seq_len={seq_len}, dim={dim})")
    fig.tight_layout()
    fig_dir = PROJECT_ROOT / "docs" / "figures" / "verify_technical_report"
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, fig_dir / "fig_breakthrough_80plus.png")

    return summary


if __name__ == "__main__":
    run_breakthrough_training()
