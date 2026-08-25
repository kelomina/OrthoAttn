# -*- coding: utf-8 -*-
"""RULER-NIAH 权威基准评测脚本: MHDSRA2 门控偏置 A/B 对照.

中文说明:
- 调用方 / Called by: CLI(`python scripts/benchmark_ruler_niah.py ...`)与低优先级后台包装器
- 调用对象 / Calls: `src.dsra.domain.ruler_niah`(数据),
  `src.dsra.dsra_model.MultiLayerMHDSRA2Model`(模型),
  `scripts.benchmark_mqar.get_cosine_warmup_scheduler`(调度器)
- 作用 / Purpose:
  1) 在 NVIDIA/RULER 规范的 S-NIAH 任务上从零训练 MHDSRA2;
  2) 支持 `--retrieval-quality-gate-bias` 单变量注入, 用于验证 MQAR 线发现的
     "负质量门控偏置解除 retrieval 支路垄断" 是否迁移到权威长程检索任务;
  3) 周期性评估序列级精确匹配(EM)并记录三路门控均值轨迹(slot/local/retrieval);
  4) 结果(JSON, 含完整 history 与门控轨迹)写入指定路径。
- 设备纪律 / Device discipline: 默认 cuda:0; CUDA 不可用时直接报错, 绝不静默回退 CPU;
  启动时打印参数驻留取证。
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_mqar import get_cosine_warmup_scheduler, seed_all  # noqa: E402
from src.dsra.domain.ruler_niah import VOCAB, RulerNiahConfig, generate_ruler_niah_batch  # noqa: E402
from src.dsra.dsra_model import MultiLayerMHDSRA2Model  # noqa: E402


def attach_gate_recorder(model: torch.nn.Module) -> dict:
    """在每层 fuse_gate 输出挂钩子, 累计 sigmoid 后三路均值(近似归一化门控)."""
    store = {"sum": torch.zeros(3), "count": 0}

    def _hook(_mod, _inputs, output):
        g = torch.sigmoid(output.float()).detach()
        store["sum"] += g.mean(dim=(0, 1, 2)).cpu()
        store["count"] += 1

    handles = [layer.fuse_gate.register_forward_hook(_hook) for layer in model.layers]
    store["handles"] = handles
    return store


def probe_gate_means(store: dict) -> list:
    """读取自上次清零以来的三路门控平均权重 [slot, local, retrieval]."""
    if store["count"] == 0:
        return [0.0, 0.0, 0.0]
    total = store["sum"] / store["count"]
    norm = total.sum().clamp_min(1e-6)
    return (total / norm).tolist()


def reset_gate_store(store: dict) -> None:
    store["sum"].zero_()
    store["count"] = 0


def score_prediction(pred_ids: list, answers: list) -> tuple:
    """对单个样本的监督位预测打分 / Score one sample's supervised-position predictions.

    中文说明:
    - 调用方 / Called by: `evaluate_exact_match`, 单元测试
    - 作用 / Purpose: 只比对答案数字位的预测(排除末尾句号监督位), 返回
      (序列级 EM, 首位数字正确率)。首位数字需真正检索上下文, 其余位存在
      相邻复制捷径, 因此两项指标分开上报。
    - 参数 / Parameters: `pred_ids` 是监督位置的 argmax token id 列表(含句号);
      `answers` 是真值数字字符串列表。
    - 返回 / Returns: (em: bool, first_digit_ok: bool)
    """
    n_digits = sum(len(a) for a in answers)
    digit_preds = pred_ids[:n_digits]
    expected: list = []
    for a in answers:
        expected.extend(VOCAB[ch] for ch in a)
    em = len(digit_preds) == n_digits and digit_preds == expected
    first_ok = bool(digit_preds) and digit_preds[0] == expected[0]
    return bool(em), bool(first_ok)


@torch.no_grad()
def evaluate_exact_match(
    model: torch.nn.Module,
    cfg: RulerNiahConfig,
    device: torch.device,
    batches: int,
) -> tuple:
    """序列级 EM 与首位检索准确率评估 / Evaluate EM and first-digit retrieval accuracy.

    中文说明:
    - 对每个样本取监督位置(Y!=0)处的 argmax 预测, 经 `score_prediction` 打分;
    - 返回 / Returns: (em 准确率, first_digit_acc 准确率)
    """
    model.eval()
    em_flags = []
    first_flags = []
    for _ in range(batches):
        X, Y, metas = generate_ruler_niah_batch(cfg)
        X, Y = X.to(device), Y.to(device)
        logits = model(X)
        pos = Y != 0
        for b in range(X.shape[0]):
            pred_ids = logits[b][pos[b]].argmax(dim=-1).tolist()
            em, first_ok = score_prediction(pred_ids, [str(a) for a in metas[b]["answers"]])
            em_flags.append(em)
            first_flags.append(first_ok)
    model.train()
    em_acc = sum(em_flags) / max(1, len(em_flags))
    first_acc = sum(first_flags) / max(1, len(first_flags))
    return em_acc, first_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="RULER-NIAH benchmark for MHDSRA2")
    parser.add_argument("--variant", type=str, default="sniah1")
    parser.add_argument("--num-haystack", type=int, default=192,
                        help="噪声海草句条数(约 18 token/条), 控制上下文长度")
    parser.add_argument("--num-needle-k", type=int, default=1)
    parser.add_argument("--num-needle-q", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--slots", type=int, default=64)
    parser.add_argument("--read-topk", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=4,
                        help="评估批数; 每批固定 8 样本, EM 粒度 = 1/(8*eval-batches)"
                        " (如默认 4 批 -> 3.125%%)。与 --batch-size(训练批)独立。")
    parser.add_argument("--retrieval-quality-gate-bias", type=float, default=0.0,
                        help="A/B 单变量: 负值抑制 retrieval 支路门控")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="计算设备; 请求 cuda 而 CUDA 不可用时报错退出, 不回退 CPU")
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"--device {args.device} 需要 CUDA 但当前不可用; 禁止静默回退 CPU")
    device = torch.device(args.device)
    seed_all(args.seed)

    cfg = RulerNiahConfig(
        variant=args.variant,
        num_haystack=args.num_haystack,
        num_needle_k=args.num_needle_k,
        num_needle_q=args.num_needle_q,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )
    eval_cfg = RulerNiahConfig(
        variant=args.variant,
        num_haystack=args.num_haystack,
        num_needle_k=args.num_needle_k,
        num_needle_q=args.num_needle_q,
        batch_size=8,
        device=device,
        seed=args.seed + 777,
    )

    override = {
        "use_retrieval": True,
        "retrieval_neighbor_span": 1,
        "retrieval_neighbor_direction": "right",
        "retrieval_query_pooling": "max_token",
        "retrieval_attention_topk": 16,
        "retrieval_quality_gate_bias": float(args.retrieval_quality_gate_bias),
        "detach_state": False,
    }
    model = MultiLayerMHDSRA2Model(
        vocab_size=len(VOCAB),
        dim=args.dim,
        num_layers=args.num_layers,
        K=args.slots,
        kr=args.read_topk,
        chunk_size=args.chunk_size,
        use_retrieval=True,
        mhdsra2_config_override=override,
    ).to(device)
    print(
        f"[forensics] requested={args.device} param_device="
        f"{next(model.parameters()).device} vocab={len(VOCAB)}",
        flush=True,
    )

    gate_store = attach_gate_recorder(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.98))
    sched = get_cosine_warmup_scheduler(opt, warmup_steps=args.warmup_steps, total_steps=args.epochs)
    crit = torch.nn.CrossEntropyLoss(ignore_index=0)

    records = []
    best_em = 0.0
    best_step = 0
    for step in range(args.epochs):
        model.train()
        # 逐训练步重播种, 避免冻结在单批样本上死记硬背(教训: benchmark_mqar 亦如此)
        step_cfg = dataclasses.replace(cfg, seed=args.seed + step)
        X, Y, _ = generate_ruler_niah_batch(step_cfg)
        X, Y = X.to(device), Y.to(device)
        opt.zero_grad()
        logits = model(X)
        loss = crit(logits.view(-1, logits.shape[-1]), Y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if (step + 1) % args.eval_interval == 0 or step == args.epochs - 1:
            reset_gate_store(gate_store)
            em, first_acc = evaluate_exact_match(model, eval_cfg, device, args.eval_batches)
            gates = probe_gate_means(gate_store)
            best_em, best_step = max(best_em, em), (step + 1) if em > best_em else best_step
            rec = {
                "step": step + 1,
                "train_loss": float(loss.item()),
                "eval_em": float(em),
                "first_digit_acc": float(first_acc),
                "gate_means_slot_local_retrieval": gates,
            }
            records.append(rec)
            print(
                f"Step {step+1}/{args.epochs} loss={loss.item():.4f} em={em*100:.1f}% "
                f"first={first_acc*100:.1f}% (best_em={best_em*100:.1f}%@{best_step}) "
                f"gates(slr)={[round(g,2) for g in gates]}",
                flush=True,
            )
    final = records[-1] if records else {"eval_em": 0.0}
    result = {
        "config": vars(args) | {"vocab_size": len(VOCAB)},
        "best_eval_em": float(best_em),
        "best_step": int(best_step),
        "final_eval_em": float(final["eval_em"]),
        "records": records,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    # 保存模型权重以便事后复评(教训: 此前无 checkpoint 导致评测 bug 无法追溯修复)
    ckpt_path = out.with_suffix(".pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args) | {"vocab_size": len(VOCAB)},
            "best_eval_em": float(best_em),
        },
        ckpt_path,
    )
    print(f"DONE bias={args.retrieval_quality_gate_bias} best_em={best_em:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
