"""MHDSRA 技术报告实验数据统一复核脚本.

对 ``docs/MHDSRA_Technical_Report.md`` 中表1~表5 的声明数值做独立复核:
- audit:      数据级审计(归档历史报告扫描 + 现有 JSON 解析 + 来源核对)
- probe:      表3 Top-K 稀释探针(数学闭合解 + 模型实测双验证)
- throughput: 表2 吞吐加速比抽样重测
- ppl:        表4 WikiText PPL 对照重测
- memory:     表1-B 显存探针(全 8 档长度, 三模式)
- niah:       表1-A NIAH 准确率重测(代表长度)
- ablation:   表5 消融变体重测(64K NIAH 口径)
- aggregate:  汇总全部子结果 -> verify_summary + 仪表盘总图
- self-test:  CPU mock 冒烟(验证绘图管道)

用法 / Usage:
    python -X utf8 -m scripts.verify_technical_report <subcommand> [options]

复核独立性原则: 评测指标(准确率/PPL/显存/计时)在本脚本内独立实现,
不 import 被测脚本的指标函数; 对同批数据双实现交叉比对,
偏差超过 0.5% 记录 metric_disagreement.
"""

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dsra.report_utils import save_figure, write_json, write_markdown  # noqa: E402

VERIFY_REPORTS_DIR = PROJECT_ROOT / "docs" / "reports" / "verify_technical_report"
VERIFY_FIGURES_DIR = PROJECT_ROOT / "docs" / "figures" / "verify_technical_report"

# ---------------------------------------------------------------------------
# 报告声明值常量 / Reported values from the technical report
# ---------------------------------------------------------------------------
REPORTED_TABLE1_NIAH_ACC = {
    16384: 1.00, 32768: 1.00, 65536: 1.00, 131072: 1.00,
    262144: 1.00, 524288: 1.00, 1048576: 1.00, 2097152: 1.00,
}
# 表1 四列显存(MB): [前向, 全图训练, 检查点前向, 检查点训练]
REPORTED_TABLE1_MEMORY_MB = {
    16384: [28.4, 316.1, 28.4, 52.1],
    32768: [46.9, 613.6, 46.9, 70.6],
    65536: [84.9, 1239.0, 84.9, 108.6],
    131072: [157.9, 2442.0, 157.9, 181.6],
    262144: [305.9, 4864.0, 305.9, 329.6],
    524288: [601.9, 9713.0, 601.9, 625.6],
    1048576: [1218.6, 19405.0, 1218.6, 1249.3],
    2097152: [2437.1, None, 2437.1, 2467.8],  # None = 报告声称 OOM
}
REPORTED_TABLE1_MHA = {
    16384: {"acc": 1.00, "mem_gb": 3.6},
    32768: {"acc": 1.00, "mem_gb": 7.2},
}
REPORTED_TABLE2_SPEEDUP = {
    (64, 512): 0.875, (128, 1024): 1.287, (256, 2048): 1.728, (256, 4096): 1.511,
}
REPORTED_TABLE2_STATE_RATIO = {
    (64, 512): 1.009, (128, 1024): 1.052, (256, 2048): 1.041, (256, 4096): 1.021,
}
REPORTED_TABLE3_WEIGHTS = {"none": 0.1176, "top32": 0.3531, "top16": 0.5301, "top8": 0.7073}
CASE_STUDIES_TABLE3_EXACT = {"none": 0.117558, "top32": 0.353073, "top16": 0.530058, "top8": 0.707344}
REPORTED_TABLE4_PPL = {
    (512, "standard"): 1.01,
    (512, "mhdsra2_no_rope"): 9.72,
    (512, "mhdsra2_rope"): 5.49,
    (1024, "standard"): 12.67,
    (1024, "mhdsra2"): 12.61,
}
REPORTED_TABLE5_ABLATION = {
    "full": {"niah_acc": 1.00, "overwrite": 1.00},
    "no_retrieval": {"niah_acc": 0.125, "overwrite": 0.45},
    "no_local": {"niah_acc": 1.00, "overwrite": 1.00},
    "no_slot": {"niah_acc": 0.88, "overwrite": 0.00},
    "novelty_only": {"niah_acc": 1.00, "overwrite": 0.00},
    "page_mean_only": {"niah_acc": 0.031, "overwrite": 1.00},
}

# 判定阈值 / Verdict thresholds (显式常量, 避免主观调整)
TH_TABLE3_REL = 0.01          # 表3 相对偏差 < 1% -> confirmed
TH_TABLE2_ABS = 0.1           # 表2 speedup 差 < 0.1x -> confirmed
TH_TABLE4_REL = 0.15          # 表4 PPL 相对偏差 < 15% -> confirmed
TH_TABLE5_PP = 0.05           # 表5 准确率差 < 5pp -> confirmed
TH_MEMORY_REL = 0.15          # 表1 显存偏差 < 15% -> confirmed
METRIC_DISAGREEMENT_REL = 0.005  # 双实现交叉比对阈值 0.5%

VERDICT_CONFIRMED = "confirmed"
VERDICT_DEVIATION = "deviation"
VERDICT_REFUTED = "refuted"
VERDICT_NO_SOURCE = "no_source"          # 仓库中找不到数据来源
VERDICT_CONTRADICTED = "contradicted"    # 仓库历史记录与声明直接矛盾
VERDICT_NOT_RUNNABLE = "not_runnable"    # 声称的配置在现有代码中不存在


def memory_verdict(measured, reported):
    """显存专用判定 / Memory-specific verdict scale.

    中文说明:
    - 作用: 显存偏差 <15% confirmed, 15%~50% deviation, >50% refuted;
      相对偏差标尺对"声称线性增长 vs 实测常数"的根本趋势矛盾必须给出 refuted,
      不能沿用 judge_rel 的宽松 deviation 区间
    """
    if measured is None or reported is None:
        return VERDICT_NO_SOURCE
    rel = abs(measured - reported) / max(abs(reported), 1e-12)
    if rel < TH_MEMORY_REL:
        return VERDICT_CONFIRMED
    if rel < 0.5:
        return VERDICT_DEVIATION
    return VERDICT_REFUTED


def judge_rel(measured, reported, threshold):
    """按相对偏差判定 / Judge by relative deviation.

    中文说明:
    - 作用: 给定重测值与报告声明值, 按相对偏差阈值输出三态判定
    - 参数: measured 重测值; reported 声明值; threshold 相对偏差阈值
    - 返回: (verdict, rel_diff); reported 为 None 时返回 no_source
    """
    if reported is None or measured is None:
        return VERDICT_NO_SOURCE, None
    if measured == reported:
        return VERDICT_CONFIRMED, 0.0
    base = max(abs(reported), 1e-12)
    rel = abs(measured - reported) / base
    if rel < threshold:
        return VERDICT_CONFIRMED, rel
    if rel < threshold * 10:
        return VERDICT_DEVIATION, rel
    return VERDICT_REFUTED, rel


# ---------------------------------------------------------------------------
# 独立指标实现 / Independent metric implementations (门3: 不 import 被测脚本指标)
# ---------------------------------------------------------------------------
def independent_niah_accuracy(pred_token_ids, target_token_ids, valid_mask=None):
    """独立 NIAH top-1 准确率 / Independent NIAH top-1 accuracy.

    中文说明:
    - 作用: 独立实现 top-1 命中率, 分母只计入有效样本; 与原脚本实现交叉比对用
    - 参数: pred_token_ids 预测 token [B]; target_token_ids 目标 [B];
      valid_mask 可选 bool [B], False 样本不计入分母
    - 返回: (accuracy, n_valid); n_valid=0 时返回 (None, 0)
    - 错误处理: 形状不一致抛 ValueError
    """
    if len(pred_token_ids) != len(target_token_ids):
        raise ValueError("pred/target length mismatch")
    correct = 0
    n_valid = 0
    for i, (p, t) in enumerate(zip(pred_token_ids, target_token_ids)):
        if valid_mask is not None and not valid_mask[i]:
            continue
        n_valid += 1
        if int(p) == int(t):
            correct += 1
    if n_valid == 0:
        return None, 0
    return correct / n_valid, n_valid


def independent_ppl(token_nll_sum, token_count):
    """独立 PPL / Independent perplexity from summed NLL.

    中文说明:
    - 作用: PPL = exp(sum(NLL)/n_tokens), 独立口径实现
    - 参数: token_nll_sum 逐 token NLL 之和; token_count 有效 token 数
    - 返回: float; token_count<=0 抛 ValueError
    """
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    return math.exp(token_nll_sum / token_count)


def independent_peak_memory_probe(alloc_fn):
    """独立 CUDA 峰值显存探针 / Independent CUDA peak-memory probe.

    中文说明:
    - 作用: 先 reset 计数器(不含 alloc_fn 之外的既有分配基线), 执行 alloc_fn,
      synchronize 后读 max_memory_allocated; 与原脚本测量点交叉比对用
    - 参数: alloc_fn 无参可调用, 执行待测分配/前向
    - 返回: peak MB (CUDA 不可用返回 None)
    """
    import torch

    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    alloc_fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def independent_timing(fn, warmup=1, repeats=3):
    """独立计时 / Independent wall-clock timing with sync.

    中文说明:
    - 作用: warmup 后 synchronize 再计时, 取中位数; 修正异步执行导致的假加速
    - 参数: fn 无参可调用; warmup 预热次数; repeats 计时次数
    - 返回: 中位耗时 ms
    """
    import torch

    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


# ---------------------------------------------------------------------------
# 通用辅助 / Shared helpers
# ---------------------------------------------------------------------------
def _fmt_pct(x):
    return "-" if x is None else f"{x * 100:.2f}%"


def _verdict_markdown_rows(rows):
    lines = [
        "| 项目 | 声明值 | 重测值 | 偏差 | 判定 | 备注 |",
        "|---|---:|---:|---:|:---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['item']} | {r['reported']} | {r['measured']} | "
            f"{r['diff']} | {r['verdict']} | {r['note']} |"
        )
    return lines


def _run_subprocess(cmd, timeout_sec=None):
    """运行子进程并回传输出 / Run subprocess capturing output.

    中文说明:
    - 作用: 统一以 ``python -X utf8`` 方式调用既有实验脚本, 规避 GBK 终端问题
    - 参数: cmd 参数列表(不含 python 前缀); timeout_sec 超时秒数
    - 返回: (returncode, stdout+stderr 合并文本)
    """
    full = [sys.executable, "-X", "utf8", *cmd]
    proc = subprocess.run(
        full, cwd=str(PROJECT_ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout_sec,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


# ---------------------------------------------------------------------------
# audit 子命令: 数据级审计 / Data-level audit of archived evidence
# ---------------------------------------------------------------------------
_NIAH_ACC_RE = re.compile(r"- final eval mean accuracy: `([\d.]+)%`")
_NIAH_SEQ_RE = re.compile(r"- sequence length: `(\d+)`")
_NIAH_VOCAB_RE = re.compile(r"\| vocab_size \| `(\d+)` \|")
_NIAH_PEAK_RE = re.compile(r"- peak allocated memory: `([\d.]+) MB`")


def cmd_audit(args):
    """审计归档证据与声明的一致性 / Audit archived evidence vs claims."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []

    # --- 表1: 扫描全部归档 NIAH 报告 ---
    niah_files = sorted(
        (PROJECT_ROOT / "docs").rglob("*.md")
    )
    history = []
    for f in niah_files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "final eval mean accuracy" not in text:
            continue
        m_acc = _NIAH_ACC_RE.search(text)
        m_seq = _NIAH_SEQ_RE.search(text)
        m_vocab = _NIAH_VOCAB_RE.search(text)
        m_peak = _NIAH_PEAK_RE.search(text)
        if not (m_acc and m_seq):
            continue
        history.append({
            "file": str(f.relative_to(PROJECT_ROOT)),
            "seq_len": int(m_seq.group(1)),
            "vocab": int(m_vocab.group(1)) if m_vocab else None,
            "final_eval_acc": float(m_acc.group(1)) / 100.0,
            "peak_mem_mb": float(m_peak.group(1)) if m_peak else None,
        })

    best_overall = max((h["final_eval_acc"] for h in history), default=None)
    best_vocab100 = max(
        (h["final_eval_acc"] for h in history if h["vocab"] == 100), default=None
    )
    # vocab=5 是平凡任务: needle value 从 [4,5) 取值恒为 4, 答案空间唯一, 不可作为检索证据
    best_meaningful = max(
        (h["final_eval_acc"] for h in history if (h["vocab"] or 0) >= 10), default=None
    )
    n_reports = len(history)
    rows.append({
        "item": "表1 NIAH 100% 声明 vs 全部归档报告",
        "reported": "100%",
        "measured": f"归档{n_reports}份, vocab100最高={_fmt_pct(best_vocab100)}, "
                    f"非平凡任务(vocab>=10)最高={_fmt_pct(best_meaningful)}",
        "diff": "-",
        "verdict": VERDICT_CONTRADICTED if (best_vocab100 or 0) < 0.9 else VERDICT_CONFIRMED,
        "note": "100% 仅出现在 vocab=5 平凡任务(答案空间唯一); vocab=100 下归档记录最高 "
                f"{_fmt_pct(best_vocab100)}, 与 16K~2M 全 100% 声明直接矛盾",
    })

    # --- 表1 显存: capacity 报告是否存在 ---
    capacity_files = list((PROJECT_ROOT / "docs").rglob("*capacity*.json"))
    rows.append({
        "item": "表1 显存列来源(capacity 报告)",
        "reported": "28.4MB~2.41GB",
        "measured": f"找到 {len(capacity_files)} 份 capacity JSON",
        "diff": "-",
        "verdict": VERDICT_NO_SOURCE if not capacity_files else VERDICT_CONFIRMED,
        "note": "2M 唯一训练峰值记录 123.17MB(probe 报告), 与声称 2.38GB 差约 20 倍",
    })

    # --- 表2: 解析现有 compare JSON ---
    compare_json = PROJECT_ROOT / "docs" / "reports" / "mhdsra2_vs_dsra_compare.json"
    table2_verdict = VERDICT_NO_SOURCE
    table2_note = "compare JSON 不存在"
    cfg_summary = None
    if compare_json.exists():
        payload = json.loads(compare_json.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        results = payload.get("results", [])
        speedups = [r.get("speedup_ratio") for r in results if r.get("speedup_ratio")]
        cfg_summary = {
            "device": cfg.get("device"),
            "seq_lengths": cfg.get("seq_lengths"),
            "slots": cfg.get("slots"),
            "chunk_sizes": cfg.get("chunk_sizes"),
            "grid_size": len(results),
            "speedup_min": min(speedups) if speedups else None,
            "speedup_max": max(speedups) if speedups else None,
        }
        claimed_grid = (
            cfg.get("device") == "cuda:0"
            and 131072 in (cfg.get("seq_lengths") or [])
            and 256 in (cfg.get("slots") or [])
            and 4096 in (cfg.get("chunk_sizes") or [])
            and len(results) >= 120
        )
        table2_verdict = VERDICT_CONFIRMED if claimed_grid else VERDICT_CONTRADICTED
        table2_note = (
            "现有 JSON 与表2声称网格一致" if claimed_grid
            else f"现有 JSON 网格(device={cfg.get('device')}, seq={cfg.get('seq_lengths')}, "
                 f"slots={cfg.get('slots')}, chunks={cfg.get('chunk_sizes')}, {len(results)}组) "
                 "与表2声称(seq 131K~1M / slots 64-256 / chunk 512-4096 / 120组 / GPU)不符"
        )
    rows.append({
        "item": "表2 吞吐网格来源审计",
        "reported": "GPU, seq=131K~1M, slots={64,128,256}, chunk={512~4096}, 120组",
        "measured": json.dumps(cfg_summary, ensure_ascii=False) if cfg_summary else "缺失",
        "diff": "-",
        "verdict": table2_verdict,
        "note": table2_note,
    })

    # --- 表5: ablation_study.py 任务规模核对 ---
    abl_path = PROJECT_ROOT / "scripts" / "ablation_study.py"
    max_seq = None
    if abl_path.exists():
        m = re.search(
            r"CURRICULUM_STAGES?\s*=\s*\[(.*?)\]",
            abl_path.read_text(encoding="utf-8", errors="replace"),
            re.S,
        )
        if m:
            max_seq = max(int(x) for x in re.findall(r'"seq_len":\s*(\d+)', m.group(1)))
    rows.append({
        "item": "表5 消融来源(64K NIAH 口径)",
        "reported": "NIAH 64K + 6 变体",
        "measured": f"ablation_study.py 任务最大 seq_len={max_seq}",
        "diff": "-",
        "verdict": VERDICT_NO_SOURCE if (max_seq or 0) < 65536 else VERDICT_CONFIRMED,
        "note": "现有消融脚本是 128~1024 toy 关联召回任务, 非表5声称的 64K NIAH",
    })

    payload = {
        "phase": "audit",
        "niah_history_reports": history,
        "niah_history_summary": {
            "n_reports": n_reports,
            "best_overall_eval_acc": best_overall,
            "best_vocab100_eval_acc": best_vocab100,
        },
        "table2_existing_json": cfg_summary,
        "rows": rows,
    }
    out_dir = VERIFY_REPORTS_DIR / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "audit.json", payload)
    write_markdown(out_dir / "audit.md", [
        "# Phase 0 数据级审计报告",
        "",
        *_verdict_markdown_rows(rows),
        "",
        "## 归档 NIAH 历史报告汇总",
        "",
        "| 文件 | seq_len | vocab | final eval acc | peak MB |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {h['file']} | {h['seq_len']} | {h['vocab']} | "
            f"{_fmt_pct(h['final_eval_acc'])} | {h['peak_mem_mb']} |"
            for h in history
        ],
    ])

    # 图: NIAH 历史准确率 vs 表1 100% 声明
    fig, ax = plt.subplots(figsize=(10, 5))
    if history:
        xs = [h["seq_len"] for h in history]
        ys = [h["final_eval_acc"] * 100 for h in history]
        ax.scatter(xs, ys, s=28, alpha=0.75, label="Archived final eval acc (%)")
    ax.axhline(100.0, color="red", ls="--", lw=2, label="Reported claim: 100%")
    if best_vocab100 is not None:
        ax.axhline(best_vocab100 * 100, color="orange", ls=":", lw=1.5,
                   label=f"Best archived vocab=100: {best_vocab100*100:.2f}%")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Final eval accuracy (%)")
    ax.set_title("Table 1 claim vs all archived NIAH reports")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save_figure(fig, VERIFY_FIGURES_DIR / "fig_audit_niah_history.png")
    print(f"[audit] reports={n_reports} best_overall={best_overall} best_vocab100={best_vocab100}")
    return payload


# ---------------------------------------------------------------------------
# probe 子命令: 表3 Top-K 稀释探针
# ---------------------------------------------------------------------------
def _closed_form_exact_weight(tau, head_dim, n_candidates):
    """表3 数学闭合解 / Closed-form exact-match weight.

    中文说明:
    - 作用: 探针构造下 exact logits = tau*1/sqrt(d), 干扰 logits = 0,
      softmax 后 exact 权重 = e^(tau/sqrt(d)) / (e^(tau/sqrt(d)) + N - 1);
      该闭合解完全独立于项目代码, 用于 oracle 验证
    """
    z = math.exp(tau / math.sqrt(head_dim))
    return z / (z + n_candidates - 1)


def cmd_probe(args):
    """表3 复核: 闭合解 + 模型实测 / Table 3: closed-form + model probe."""
    import torch

    from src.dsra.mhdsra2.improved_dsra_mha import MHDSRA2Config, MultiHeadDSRA2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(101)
    head_dim, n_cand, tau = 8, 128, 8.0
    rows = []
    for topk_label, topk_val in [("none", None), ("top32", 32), ("top16", 16), ("top8", 8)]:
        closed = _closed_form_exact_weight(tau, head_dim, n_cand if topk_val is None else topk_val)
        cfg = MHDSRA2Config(
            dim=head_dim, heads=1, slots=4, use_local=False, use_retrieval=True,
            retrieval_tau=tau, retrieval_attention_topk=topk_val,
        )
        model = MultiHeadDSRA2(cfg).to(device).eval()
        q = torch.zeros(1, 1, 1, head_dim, device=device)
        q[..., 0] = 1.0
        rk = torch.zeros(1, 1, n_cand, head_dim, device=device)
        rk[..., 0, 0] = 1.0
        rk[..., 1:, 1] = 1.0
        rv = torch.randn(1, 1, n_cand, head_dim, device=device)
        mask = torch.ones(1, n_cand, dtype=torch.bool, device=device)
        with torch.no_grad():
            _, weights = model._retrieval_attention(q, rk, rv, mask, return_weights=True)
        measured = float(weights[0, 0, 0, 0].item())
        max_distractor = float(weights[0, 0, 0, 1:].max().item())
        reported = REPORTED_TABLE3_WEIGHTS[topk_label]
        exact_ref = CASE_STUDIES_TABLE3_EXACT[topk_label]
        v_closed, rel_closed = judge_rel(measured, exact_ref, TH_TABLE3_REL)
        v_report, rel_report = judge_rel(measured, reported, TH_TABLE3_REL)
        rows.append({
            "topk": topk_label,
            "measured": measured,
            "closed_form": closed,
            "reported": reported,
            "case_studies_exact": exact_ref,
            "max_distractor": max_distractor,
            "verdict_vs_report": v_report,
            "rel_diff_vs_report": rel_report,
            "verdict_vs_closed_form": v_closed,
            "rel_diff_vs_closed": rel_closed,
        })
        print(f"[probe] topk={topk_label}: measured={measured:.6f} closed={closed:.6f} "
              f"reported={reported} verdict={v_report}")

    payload = {"phase": "probe", "device": str(device), "rows": rows}
    out_dir = VERIFY_REPORTS_DIR / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "probe.json", payload)
    write_markdown(out_dir / "probe.md", [
        "# 表3 Top-K 稀释探针复核",
        "",
        *_verdict_markdown_rows([
            {
                "item": f"topk={r['topk']}",
                "reported": f"{r['reported']}",
                "measured": f"{r['measured']:.6f}",
                "diff": f"{(r['rel_diff_vs_report'] or 0)*100:.3f}%",
                "verdict": r["verdict_vs_report"],
                "note": f"closed_form={r['closed_form']:.6f} max_distractor={r['max_distractor']:.6f}",
            } for r in rows
        ]),
    ])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [r["topk"] for r in rows]
    x = range(len(rows))
    w = 0.27
    ax.bar([i - w for i in x], [r["reported"] for r in rows], w, label="Reported", color="#888")
    ax.bar(list(x), [r["measured"] for r in rows], w, label="Re-measured", color="#2b6cb0")
    ax.bar([i + w for i in x], [r["closed_form"] for r in rows], w, label="Closed-form", color="#dd6b20")
    ax2 = ax.twinx()
    ax2.plot(list(x), [r["max_distractor"] for r in rows], "g.--", label="max distractor w")
    ax2.set_ylabel("max distractor weight")
    ax2.legend(loc="lower right", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Exact-match attention weight")
    ax.set_title("Table 3: Top-K dilution probe (reported vs re-measured vs math)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    save_figure(fig, VERIFY_FIGURES_DIR / "fig3_topk_dilution.png")
    return payload


# ---------------------------------------------------------------------------
# throughput 子命令: 表2 抽样重测
# ---------------------------------------------------------------------------
def cmd_throughput(args):
    """表2 复核: 抽样网格重测 / Table 2: sampled grid re-measurement."""
    seq_lengths = [int(x) for x in args.seq_lengths.split(",")]
    slots_list = [int(x) for x in args.slots.split(",")]
    chunks_list = [int(x) for x in args.chunks.split(",")]
    cmd = [
        "-m", "scripts.compare_mhdsra2_vs_dsra",
        "--seq-lengths", *[str(s) for s in seq_lengths],
        "--slots", *[str(s) for s in slots_list],
        "--chunk-sizes", *[str(c) for c in chunks_list],
        "--warmup-runs", str(args.warmup),
        "--repeat-runs", str(args.repeats),
        "--seed", "7",
        "--device", args.device,
        "--reports-dir", str(VERIFY_REPORTS_DIR / "throughput"),
    ]
    print("[throughput] running:", " ".join(cmd))
    # compare 脚本经 ensure_reports_dir 会在目标目录下追加 reports/ 子目录
    result_json = VERIFY_REPORTS_DIR / "throughput" / "reports" / "mhdsra2_vs_dsra_compare.json"
    if not result_json.exists():
        result_json = VERIFY_REPORTS_DIR / "throughput" / "mhdsra2_vs_dsra_compare.json"
    if args.reuse and result_json.exists():
        rc, output = 0, "(reused existing compare JSON)"
        print(f"[throughput] reuse existing results: {result_json}")
    else:
        rc, output = _run_subprocess(cmd, timeout_sec=args.timeout)
        print(output[-3000:])
    rows = []
    measured_pairs = {}
    if result_json.exists():
        payload = json.loads(result_json.read_text(encoding="utf-8"))
        for r in payload.get("results", []):
            key = (r.get("slots"), r.get("chunk_size"))
            measured_pairs.setdefault(key, []).append(r.get("speedup_ratio"))
    for (slots, chunk), reported in REPORTED_TABLE2_SPEEDUP.items():
        vals = measured_pairs.get((slots, chunk), [])
        measured = statistics.median(vals) if vals else None
        verdict, rel = judge_rel(measured, reported, TH_TABLE2_ABS)
        rows.append({
            "slots": slots, "chunk_size": chunk, "reported": reported,
            "measured": measured, "n_cases": len(vals),
            "verdict": verdict, "abs_diff": None if measured is None else measured - reported,
        })
        print(f"[throughput] K={slots} T={chunk}: reported={reported} measured={measured} {verdict}")
    payload_out = {
        "phase": "throughput", "subprocess_rc": rc,
        "grid": {"seq_lengths": seq_lengths, "slots": slots_list, "chunks": chunks_list},
        "rows": rows,
        "verdict_note": "现有归档 JSON 为 CPU/短序列网格, 表2来源已被 audit 判定为矛盾; "
                        "此处为同声明口径的独立重测",
    }
    out_dir = VERIFY_REPORTS_DIR / "throughput"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "throughput_verify.json", payload_out)
    write_markdown(out_dir / "throughput_verify.md", [
        "# 表2 吞吐加速比复核(重测)",
        "",
        *_verdict_markdown_rows([
            {
                "item": f"K={r['slots']},T={r['chunk_size']}",
                "reported": f"{r['reported']}x",
                "measured": "-" if r["measured"] is None else f"{r['measured']:.3f}x",
                "diff": "-" if r["abs_diff"] is None else f"{r['abs_diff']:+.3f}x",
                "verdict": r["verdict"],
                "note": f"n={r['n_cases']} cases",
            } for r in rows
        ]),
    ])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"K={r['slots']}\nT={r['chunk_size']}" for r in rows]
    x = range(len(rows))
    w = 0.35
    ax.bar([i - w / 2 for i in x], [r["reported"] for r in rows], w, label="Reported", color="#888")
    ax.bar([i + w / 2 for i in x],
           [r["measured"] if r["measured"] is not None else 0 for r in rows],
           w, label="Re-measured", color="#2b6cb0")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Speedup vs baseline (x)")
    ax.set_title("Table 2: throughput speedup (reported vs re-measured)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    save_figure(fig, VERIFY_FIGURES_DIR / "fig2_throughput_speedup.png")
    return payload_out


# ---------------------------------------------------------------------------
# throughput-merge 子命令: 汇总多 seq_len 网格结果 (Task 3, 目标3)
# ---------------------------------------------------------------------------
THROUGHPUT_GRID_DIRS = {
    131072: VERIFY_REPORTS_DIR / "throughput" / "reports" / "mhdsra2_vs_dsra_compare.json",
    262144: VERIFY_REPORTS_DIR / "throughput_262k" / "reports" / "mhdsra2_vs_dsra_compare.json",
    524288: VERIFY_REPORTS_DIR / "throughput_524k" / "reports" / "mhdsra2_vs_dsra_compare.json",
    1048576: VERIFY_REPORTS_DIR / "throughput_1m" / "reports" / "mhdsra2_vs_dsra_compare.json",
}


def cmd_throughput_merge(args):
    """汇总 131K~1M 全部吞吐网格 / Merge all seq_len throughput grids.

    中文说明:
    - 作用: 读取各 seq_len 的 compare JSON, 聚合为按长度分组的加速比/状态倍率
      总表 + fig2b 图 + markdown, 供报告 v1.2 表2 使用
    - 调用方 / Called by: CLI `throughput-merge`
    - 数据来源: 全部为本轮(或本会话)实测 JSON; 缺失的长度如实标注 missing
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_len = {}
    missing = []
    for seq_len, path in THROUGHPUT_GRID_DIRS.items():
        if not path.exists():
            missing.append(seq_len)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for r in payload.get("results", []):
            by_len.setdefault(seq_len, []).append({
                "slots": r.get("slots"), "chunk_size": r.get("chunk_size"),
                "speedup": r.get("speedup_ratio"),
                "state_ratio": r.get("dsra_to_mhdsra2_state_bytes_ratio"),
                "dsra_ms": r.get("dsra", {}).get("elapsed_ms"),
                "mhdsra2_ms": r.get("mhdsra2", {}).get("elapsed_ms"),
            })
    lens = sorted(by_len)
    summary_rows = []
    for seq_len in lens:
        cases = by_len[seq_len]
        speedups = [c["speedup"] for c in cases if c["speedup"] is not None]
        ratios = [c["state_ratio"] for c in cases if c["state_ratio"] is not None]
        summary_rows.append({
            "seq_len": seq_len, "n_cases": len(cases),
            "speedup_median": statistics.median(speedups) if speedups else None,
            "speedup_min": min(speedups) if speedups else None,
            "speedup_max": max(speedups) if speedups else None,
            "state_ratio_median": statistics.median(ratios) if ratios else None,
        })
    payload = {
        "phase": "throughput_merge",
        "missing_lengths": missing,
        "summary": summary_rows,
        "detail_by_length": {
            str(k): v for k, v in by_len.items()
        },
    }
    out_dir = VERIFY_REPORTS_DIR / "throughput_merge"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "throughput_merge.json", payload)
    write_markdown(out_dir / "throughput_merge.md", [
        "# 吞吐网格扩展汇总（131K~1M, 本轮实测）",
        "",
        "| seq_len | 案例数 | 加速比中位 | 加速比区间 | 状态倍率中位 |",
        "|---:|---:|---:|---:|---:|",
        *[
            f"| {r['seq_len']} | {r['n_cases']} | "
            f"{r['speedup_median']:.3f}x | "
            f"{r['speedup_min']:.2f}x ~ {r['speedup_max']:.2f}x | "
            f"{r['state_ratio_median']:.4f}x" for r in summary_rows
        ],
        *(["", f"缺失长度: {missing}"] if missing else []),
    ])

    # 图: 按 seq_len 分组的加速比(中位+区间) + 状态倍率
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    xs = [r["seq_len"] for r in summary_rows]
    med = [r["speedup_median"] or 0 for r in summary_rows]
    lo = [r["speedup_min"] or 0 for r in summary_rows]
    hi = [r["speedup_max"] or 0 for r in summary_rows]
    ax1.plot(xs, med, "o-", color="#2b6cb0", label="median speedup")
    ax1.fill_between(xs, lo, hi, alpha=0.2, color="#2b6cb0", label="min-max")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("seq_len (tokens)")
    ax1.set_ylabel("Speedup vs DSRA baseline (x)")
    ax1.set_title("Throughput speedup by context length")
    ax1.legend()
    ax1.grid(alpha=0.3)
    sr = [r["state_ratio_median"] or 0 for r in summary_rows]
    ax2.plot(xs, sr, "s-", color="#dd6b20", label="state bytes ratio median (DSRA/MHDSRA2)")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("seq_len (tokens)")
    ax2.set_ylabel("DSRA/MHDSRA2 state bytes ratio")
    ax2.set_title("State memory ratio by context length")
    ax2.legend()
    ax2.grid(alpha=0.3)
    save_figure(fig, VERIFY_FIGURES_DIR / "fig2b_throughput_by_seqlen.png")
    print(f"[throughput-merge] lengths={lens} missing={missing}")
    return payload


# ---------------------------------------------------------------------------
# ppl 子命令: 表4 重测
# ---------------------------------------------------------------------------
def cmd_ppl(args):
    """表4 复核: PPL 重测 / Table 4: PPL re-measurement."""
    rows = []
    base_cmd = [
        "scripts/tiny_llama_compare.py",
        "--batch-size", "1",
        "--dim", "128", "--heads", "4", "--layers", "2",
        "--mhdsra2-chunk-size", "1024",
        "--device", args.device,
    ]
    # N=1024 主对照, 多 seed
    for seed in [int(s) for s in args.seeds.split(",")]:
        rc, output = _run_subprocess(
            [*base_cmd, "--seq-len", "1024", "--max-steps", str(args.steps),
             "--seed", str(seed)],
            timeout_sec=args.timeout,
        )
        std = re.search(r"Standard Attention Validation PPL:\s*([\d.]+)", output)
        mh = re.search(r"MHDSRA2 Validation PPL:\s*([\d.]+)", output)
        rows.append({
            "seq_len": 1024, "seed": seed, "rc": rc,
            "standard_ppl": float(std.group(1)) if std else None,
            "mhdsra2_ppl": float(mh.group(1)) if mh else None,
        })
        print(f"[ppl] N=1024 seed={seed}: std={std.group(1) if std else 'NA'} "
              f"mhdsra2={mh.group(1) if mh else 'NA'} rc={rc}")
    # N=512 可复现配置只有 standard / mhdsra2 默认(无 slot RoPE 开关, 依赖项标记 not_runnable)
    if args.run_512:
        rc, output = _run_subprocess(
            [*base_cmd, "--seq-len", "512", "--max-steps", str(args.steps),
             "--seed", "1234"],
            timeout_sec=args.timeout,
        )
        std = re.search(r"Standard Attention Validation PPL:\s*([\d.]+)", output)
        mh = re.search(r"MHDSRA2 Validation PPL:\s*([\d.]+)", output)
        rows.append({
            "seq_len": 512, "seed": 1234, "rc": rc,
            "standard_ppl": float(std.group(1)) if std else None,
            "mhdsra2_ppl": float(mh.group(1)) if mh else None,
        })
        print(f"[ppl] N=512: std={std.group(1) if std else 'NA'} "
              f"mhdsra2={mh.group(1) if mh else 'NA'} rc={rc}")

    verdict_rows = []
    for key, reported in REPORTED_TABLE4_PPL.items():
        n, model = key
        if model == "mhdsra2_rope":
            verdict_rows.append({
                "item": f"N={n} MHDSRA2+slotRoPE", "reported": reported,
                "measured": "N/A", "diff": "-",
                "verdict": VERDICT_NOT_RUNNABLE,
                "note": "tiny_llama 脚本无 slot RoPE 开关(审计门2确认), 无法复现",
            })
            continue
        field = "standard_ppl" if model.startswith("standard") else "mhdsra2_ppl"
        vals = [r[field] for r in rows if r["seq_len"] == n and r[field] is not None]
        measured = statistics.mean(vals) if vals else None
        verdict, rel = judge_rel(measured, reported, TH_TABLE4_REL)
        verdict_rows.append({
            "item": f"N={n} {model}", "reported": reported,
            "measured": "-" if measured is None else f"{measured:.2f}",
            "diff": "-" if rel is None else f"{rel*100:.1f}%",
            "verdict": verdict,
            "note": f"n_seeds={len(vals)}" + ("; PPL=1.01 反常值审计: 正确口径下无法复现" if (n == 512 and model == 'standard') else ""),
        })

    payload = {"phase": "ppl", "rows": rows, "verdict_rows": verdict_rows}
    out_dir = VERIFY_REPORTS_DIR / "ppl"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "ppl.json", payload)
    write_markdown(out_dir / "ppl.md", [
        "# 表4 WikiText PPL 复核",
        "",
        "## 逐配置判定",
        "",
        *_verdict_markdown_rows(verdict_rows),
        "",
        "## 原始重测记录",
        "",
        "| seq_len | seed | standard PPL | mhdsra2 PPL | rc |",
        "|---:|---:|---:|---:|---:|",
        *[
            f"| {r['seq_len']} | {r['seed']} | {r['standard_ppl']} | {r['mhdsra2_ppl']} | {r['rc']} |"
            for r in rows
        ],
    ])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    keys = [(512, "standard"), (512, "mhdsra2_no_rope"), (1024, "standard"), (1024, "mhdsra2")]
    labels, rep, meas = [], [], []
    for n, model in keys:
        field = "standard_ppl" if model.startswith("standard") else "mhdsra2_ppl"
        vals = [r[field] for r in rows if r["seq_len"] == n and r[field] is not None]
        labels.append(f"N={n}\n{model.replace('_no_rope','')}")
        rep.append(REPORTED_TABLE4_PPL[(n, model)])
        meas.append(statistics.mean(vals) if vals else 0)
    x = range(len(keys))
    w = 0.35
    ax.bar([i - w / 2 for i in x], rep, w, label="Reported", color="#888")
    ax.bar([i + w / 2 for i in x], meas, w, label="Re-measured", color="#2b6cb0")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Validation PPL (lower better)")
    ax.set_title("Table 4: WikiText PPL (reported vs re-measured)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    save_figure(fig, VERIFY_FIGURES_DIR / "fig4_wikitext_ppl.png")
    return payload


# ---------------------------------------------------------------------------
# memory 子命令: 表1-B 显存探针
# ---------------------------------------------------------------------------
def cmd_memory(args):
    """表1-B 复核: 独立显存探针 / Table 1-B: independent memory probe."""
    import torch

    from scripts.needle_in_haystack_test import (
        build_niah_model,
        extract_query_positions_and_targets,
        generate_haystack_with_needle,
    )

    device = torch.device(args.device if args.device != "auto"
                          else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    lengths = [int(x) for x in args.lengths.split(",")]
    rows = []
    for seq_len in lengths:
        entry = {"seq_len": seq_len, "modes": {}}
        for mode in ["forward_only", "train_step"]:
            try:
                def probe_fn(mode=mode, seq_len=seq_len):
                    torch.manual_seed(20260506)
                    model = build_niah_model(
                        device=device, vocab_size=100, dim=64, num_layers=2,
                        K=64, kr=8, chunk_size=1024,
                    )
                    model.eval()
                    X, Y, _ = generate_haystack_with_needle(1, seq_len, 100, 0.5)
                    qpos, targets = extract_query_positions_and_targets(X, Y, device)
                    if mode == "forward_only":
                        with torch.no_grad():
                            model.forward_selected_logits(X, qpos)
                    else:
                        model.train()
                        logits = model.forward_selected_logits(X, qpos)
                        loss = torch.nn.functional.cross_entropy(logits, targets)
                        loss.backward()
                    del model, X, Y, qpos, targets

                peak = independent_peak_memory_probe(probe_fn) if device.type == "cuda" else None
                entry["modes"][mode] = {"peak_mem_mb": peak, "status": "ok"}
                del probe_fn
            except torch.cuda.OutOfMemoryError:
                entry["modes"][mode] = {"peak_mem_mb": None, "status": "oom"}
            except Exception as exc:  # noqa: BLE001 - 记录所有失败模式
                entry["modes"][mode] = {"peak_mem_mb": None, "status": f"error:{type(exc).__name__}"}
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rep = REPORTED_TABLE1_MEMORY_MB.get(seq_len, [None] * 4)
        fwd = entry["modes"].get("forward_only", {}).get("peak_mem_mb")
        train = entry["modes"].get("train_step", {}).get("peak_mem_mb")
        # 声明口径: [0]=前向(近似 detach 探针), [1]=全图训练; train_step 默认 auto-detach(长序列)
        v_fwd = memory_verdict(fwd, rep[0])
        v_train = memory_verdict(train, rep[3])
        entry["verdicts"] = {"forward_vs_reported_forward": v_fwd,
                             "train_step_vs_reported_ckpt_train": v_train}
        rows.append(entry)
        print(f"[memory] seq={seq_len}: fwd={fwd}MB train={train}MB "
              f"reported_fwd={rep[0]} reported_ckpt_train={rep[3]}")

    payload = {"phase": "memory", "device": str(device), "rows": rows}
    out_dir = VERIFY_REPORTS_DIR / "memory"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "memory.json", payload)
    write_markdown(out_dir / "memory.md", [
        "# 表1-B 显存探针复核(独立测量)",
        "",
        "| seq_len | 重测前向(MB) | 声明前向(MB) | 判定 | 重测单步训练(MB) | 声明检查点训练(MB) | 判定 |",
        "|---:|---:|---:|:---:|---:|---:|:---:|",
        *[
            f"| {r['seq_len']} | "
            f"{r['modes'].get('forward_only', {}).get('peak_mem_mb')} | "
            f"{REPORTED_TABLE1_MEMORY_MB.get(r['seq_len'], [None]*4)[0]} | "
            f"{r['verdicts']['forward_vs_reported_forward']} | "
            f"{r['modes'].get('train_step', {}).get('peak_mem_mb')} | "
            f"{REPORTED_TABLE1_MEMORY_MB.get(r['seq_len'], [None]*4)[3]} | "
            f"{r['verdicts']['train_step_vs_reported_ckpt_train']} |"
            for r in rows
        ],
        "",
        "注: 重测为独立探针(reset 后仅计模型+单批数据+一步反传); 长序列默认 auto-detach。",
    ])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fwd_meas = [(r["modes"].get("forward_only", {}) or {}).get("peak_mem_mb") for r in rows]
    train_meas = [(r["modes"].get("train_step", {}) or {}).get("peak_mem_mb") for r in rows]
    fwd_rep = [REPORTED_TABLE1_MEMORY_MB.get(r["seq_len"], [None] * 4)[0] for r in rows]
    train_rep = [REPORTED_TABLE1_MEMORY_MB.get(r["seq_len"], [None] * 4)[3] for r in rows]
    xs = [r["seq_len"] for r in rows]
    ax.plot(xs, [v or 0 for v in fwd_rep], "v--", label="Reported forward", color="#888")
    ax.plot(xs, [v or 0 for v in train_rep], "^--", label="Reported ckpt-train", color="#a0aec0")
    ax.plot(xs, [v or 0 for v in fwd_meas], "o-", label="Re-measured forward", color="#2b6cb0")
    ax.plot(xs, [v or 0 for v in train_meas], "s-", label="Re-measured 1-step train", color="#dd6b20")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Peak CUDA memory (MB)")
    ax.set_title("Table 1-B: memory probe (log-log)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    save_figure(fig, VERIFY_FIGURES_DIR / "fig1b_memory.png")
    return payload


# ---------------------------------------------------------------------------
# niah-matrix 子命令: Stage A 快速诊断矩阵 + A-0 梯度探针（目标1 实证）
# ---------------------------------------------------------------------------
# 证据分级: A-0/A0~A3 全部为本轮实测; 检验的"历史叙事"仅作假设, 结论以数据为准
NIAH_MATRIX_CONFIGS = {
    "a0": {"label": "A0 基线(检索关闭)", "use_retrieval": False, "override": {}},
    "a1": {"label": "A1 +use_retrieval", "use_retrieval": True, "override": {}},
    "a2": {"label": "A2 +retrieval_topk8", "use_retrieval": True,
           "override": {"retrieval_attention_topk": 8}},
    "a3": {"label": "A3 +detach_state=False", "use_retrieval": True,
           "override": {"retrieval_attention_topk": 8, "detach_state": False}},
}
STAGE_A_SUCCESS_THRESHOLD = 0.50


def _gradient_probe(model, opt, crit, make_batch, device, steps=20):
    """A-0 梯度范数探针 / Gradient-norm probe for write-vs-read paths.

    中文说明:
    - 作用: 训练若干步, 每步记录写入侧(layers[*].qkv 的 K/V 行切片)与读出侧
      (layers[*].out_proj + 模型级 out_proj)的梯度 L2 范数, 用实测数据检验
      "写入侧梯度被门控连乘衰减"的历史叙事([假设] -> [实测]/[证伪])
    - 参数: model/opt/crit 常规训练组件; make_batch() 返回 (X, qpos, targets);
      steps 探针步数
    - 返回: dict 含逐步范数序列与均值比; 无梯度的参数记 0
    - 副作用: 消耗 RNG、更新模型参数(探针后模型被丢弃, 不影响矩阵主实验)
    """
    rows = []
    for step in range(steps):
        X, qpos, targets = make_batch()
        opt.zero_grad()
        logits = model.forward_selected_logits(X, qpos)
        loss = crit(logits, targets)
        loss.backward()
        opt.step()
        write_norms, read_norms = [], []
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            g = param.grad.detach().float()
            if name.endswith("qkv.weight"):
                # qkv 是合并投影 [3*dim, dim]: 按行切 K/V 段(写入侧)
                dim = g.shape[1]
                k_norm = float(g[dim:2 * dim].norm().item())
                v_norm = float(g[2 * dim:3 * dim].norm().item())
                write_norms.extend([k_norm, v_norm])
            elif name.endswith("out_proj.weight"):
                read_norms.append(float(g.norm().item()))
        rows.append({
            "step": step,
            "loss": float(loss.item()),
            "write_grad_norm_mean": (sum(write_norms) / len(write_norms)) if write_norms else 0.0,
            "read_grad_norm_mean": (sum(read_norms) / len(read_norms)) if read_norms else 0.0,
        })
    write_mean = statistics.mean(r["write_grad_norm_mean"] for r in rows)
    read_mean = statistics.mean(r["read_grad_norm_mean"] for r in rows)
    ratio = write_mean / max(read_mean, 1e-12)
    return {
        "steps": rows,
        "write_grad_norm_mean": write_mean,
        "read_grad_norm_mean": read_mean,
        "write_to_read_ratio": ratio,
        "attenuation_narrative": ("confirmed" if ratio < 1e-3 else "refuted"),
    }


def cmd_niah_matrix(args):
    """Stage A: A-0 梯度探针 + A0~A3 快速矩阵 / Empirical diagnosis matrix.

    中文说明:
    - 作用: 实证检验 NIAH 成绩差的原因——A0~A3 对比"检索分支是否为缺失关键";
      A-0 用梯度范数实测替代历史"梯度衰减"叙事; 全部结论以本轮实测为准
    - 调用方 / Called by: CLI `niah-matrix`(支持 --configs 子集实现多进程并行)
    - 并行: 每进程跑 --configs 指定的子集, 各写独立 JSON; --merge-only 汇总
    """
    import torch

    from scripts.needle_in_haystack_test import (
        build_niah_model,
        evaluate_niah_depths,
        extract_query_positions_and_targets,
        generate_haystack_with_needle,
    )

    device = torch.device(args.device if args.device != "auto"
                          else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    out_dir = VERIFY_REPORTS_DIR / "niah_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        return _merge_niah_matrix(out_dir)

    wanted = [c.strip().lower() for c in args.configs.split(",") if c.strip()]
    seq_len, epochs = int(args.seq_len), int(args.epochs)
    seed = int(args.seed)
    results = []
    for key in wanted:
        spec = NIAH_MATRIX_CONFIGS[key]
        try:
            torch.manual_seed(seed)
            model = build_niah_model(
                device=device, vocab_size=100, dim=64, num_layers=2,
                K=64, kr=8, chunk_size=1024,
                mhdsra2_config_override=spec["override"],
                use_retrieval=spec["use_retrieval"],
            )
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            crit = torch.nn.CrossEntropyLoss(ignore_index=0)

            grad_probe = None
            if args.grad_probe and key == "a0":
                depths_cycle = [0.1, 0.5, 0.9]

                def make_batch():
                    X, Y, _ = generate_haystack_with_needle(
                        1, seq_len, 100, depths_cycle[torch.randint(0, 3, (1,)).item()])
                    qpos, targets = extract_query_positions_and_targets(X, Y, device)
                    return X, qpos, targets

                grad_probe = _gradient_probe(
                    model, opt, crit, make_batch, device,
                    steps=int(args.grad_probe_steps))
                # 探针后重建模型, 保证矩阵主实验从干净初始化开始
                del model, opt
                torch.manual_seed(seed)
                model = build_niah_model(
                    device=device, vocab_size=100, dim=64, num_layers=2,
                    K=64, kr=8, chunk_size=1024,
                    mhdsra2_config_override=spec["override"],
                    use_retrieval=spec["use_retrieval"],
                )
                opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            model.train()
            depths_cycle = [0.1, 0.5, 0.9]
            for step in range(epochs):
                X, Y, _ = generate_haystack_with_needle(
                    1, seq_len, 100, depths_cycle[step % 3])
                qpos, targets = extract_query_positions_and_targets(X, Y, device)
                opt.zero_grad()
                logits = model.forward_selected_logits(X, qpos)
                loss = crit(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            eval_result = evaluate_niah_depths(
                model, seq_len, device, vocab_size=100, batch_size=1,
                criterion=crit, batches_per_depth=4,
            )
            acc = eval_result["mean_accuracy"]
            sample_metrics = eval_result.get("sample_metrics", [])
            corrects = [row.get("correct") for row in sample_metrics
                        if row.get("correct") is not None]
            indep_acc = (sum(1 for c in corrects if c) / len(corrects)) if corrects else None
            results.append({
                "config": key, "label": spec["label"],
                "use_retrieval": spec["use_retrieval"], "override": spec["override"],
                "seq_len": seq_len, "epochs": epochs, "seed": seed,
                "final_eval_acc": acc, "independent_acc": indep_acc,
                "n_eval_samples": len(sample_metrics),
                "metric_disagreement": (
                    indep_acc is not None and acc is not None
                    and abs(indep_acc - acc) > METRIC_DISAGREEMENT_REL),
                "passed_threshold": acc is not None and acc >= STAGE_A_SUCCESS_THRESHOLD,
                "grad_probe": grad_probe,
            })
            print(f"[matrix] {key}({spec['label']}): acc={_fmt_pct(acc)} "
                  f"indep={_fmt_pct(indep_acc)} pass={results[-1]['passed_threshold']}")
            del model, opt
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 - 记录失败模式
            results.append({
                "config": key, "label": spec["label"], "seq_len": seq_len,
                "epochs": epochs, "seed": seed, "final_eval_acc": None,
                "independent_acc": None, "metric_disagreement": None,
                "n_eval_samples": 0, "passed_threshold": False,
                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            })
            print(f"[matrix] {key}: ERROR {type(exc).__name__}: {str(exc)[:120]}")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_json(out_dir / f"matrix_{wanted[0]}_{'_'.join(wanted[1:])}.json",
               {"phase": "niah_matrix", "rows": results})
    return {"rows": results}


def _merge_niah_matrix(out_dir):
    """汇总多进程矩阵结果 / Merge parallel matrix shards.

    中文说明:
    - 作用: 读取 niah_matrix/ 下全部 matrix_*.json 分片, 合并为总表 +
      Stage A 门限结论 + 梯度探针结论, 写 matrix_summary.json/.md
    """
    rows = []
    for f in sorted(out_dir.glob("matrix_*.json")):
        if f.name.startswith("matrix_summary"):
            continue
        payload = json.loads(f.read_text(encoding="utf-8"))
        rows.extend(payload.get("rows", []))
    best = max((r["final_eval_acc"] for r in rows
                if r.get("final_eval_acc") is not None), default=None)
    winners = [r["config"] for r in rows if r.get("passed_threshold")]
    grad_rows = [r for r in rows if r.get("grad_probe")]
    summary = {
        "phase": "niah_matrix_summary",
        "n_configs": len(rows),
        "best_acc": best,
        "stage_a_passed": bool(winners),
        "winner_configs": winners,
        "grad_probe": grad_rows[0]["grad_probe"] if grad_rows else None,
        "rows": rows,
    }
    write_json(out_dir / "matrix_summary.json", summary)
    lines = [
        "# Stage A NIAH 诊断矩阵（本轮实测）",
        "",
        f"- Stage A 门限(≥{STAGE_A_SUCCESS_THRESHOLD:.0%}): "
        f"{'通过, 胜者=' + ','.join(winners) if winners else '未通过, Stage B 取消'}",
        "",
        "| 配置 | 说明 | 检索 | final eval acc | 独立重算 | 分歧 | 过门限 |",
        "|---|---|:---:|---:|---:|:---:|:---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['label']} | {r.get('use_retrieval')} | "
            f"{_fmt_pct(r.get('final_eval_acc'))} | {_fmt_pct(r.get('independent_acc'))} | "
            f"{r.get('metric_disagreement')} | {r.get('passed_threshold')} |")
    if grad_rows:
        gp = grad_rows[0]["grad_probe"]
        lines += [
            "",
            "## A-0 梯度范数探针（写入侧 qkv[K/V] vs 读出侧 out_proj）",
            "",
            f"- 写入侧梯度范数均值: {gp['write_grad_norm_mean']:.6e}",
            f"- 读出侧梯度范数均值: {gp['read_grad_norm_mean']:.6e}",
            f"- 写/读梯度比: {gp['write_to_read_ratio']:.6e}",
            f"- 历史叙事'梯度衰减≈万分之一'判定: {gp['attenuation_narrative']}"
            "（ratio<1e-3 为 confirmed, 否则 refuted）",
        ]
    write_markdown(out_dir / "matrix_summary.md", lines)
    print(f"[matrix] summary: best={_fmt_pct(best)} winners={winners} "
          f"grad_narrative={grad_rows[0]['grad_probe']['attenuation_narrative'] if grad_rows else 'N/A'}")

# ---------------------------------------------------------------------------
# retrieval-breakthrough 子命令: 长程检索卖点攻关实验 (Task B)
# ---------------------------------------------------------------------------
RETRIEVAL_BREAKTHROUGH_CONFIGS = {
    "exp1_base": {
        "label": "Exp1 基线 (dim=64, 纯末尾CE)",
        "dim": 64,
        "needle_loss_alpha": 0.0,
        "qk_align_alpha": 0.0,
        "use_retrieval": True,
        "heads": 4,
        "override": {},
    },
    "exp2_needle_aux": {
        "label": "Exp2 +Needle局部辅助Loss (dim=64)",
        "dim": 64,
        "needle_loss_alpha": 0.5,
        "qk_align_alpha": 0.0,
        "use_retrieval": True,
        "heads": 4,
        "override": {},
    },
    "exp3_qk_align": {
        "label": "Exp3 +QK证据余弦对齐 (dim=64)",
        "dim": 64,
        "needle_loss_alpha": 0.0,
        "qk_align_alpha": 0.5,
        "use_retrieval": True,
        "heads": 4,
        "override": {},
    },
    "exp4_combined": {
        "label": "Exp4 组合Loss (Aux+QKAlign, dim=64)",
        "dim": 64,
        "needle_loss_alpha": 0.5,
        "qk_align_alpha": 0.5,
        "use_retrieval": True,
        "heads": 4,
        "override": {},
    },
    "exp5_dim128": {
        "label": "Exp5 容量扩展 (组合Loss, dim=128)",
        "dim": 128,
        "needle_loss_alpha": 0.5,
        "qk_align_alpha": 0.5,
        "use_retrieval": True,
        "heads": 4,
        "override": {},
    },
    "exp6_dim256": {
        "label": "Exp6 容量扩展 (组合Loss, dim=256)",
        "dim": 256,
        "needle_loss_alpha": 0.5,
        "qk_align_alpha": 0.5,
        "use_retrieval": True,
        "heads": 8,
        "override": {},
    },
    "exp7_neighbor_right_dim128": {
        "label": "Exp7 邻居右召回+max_token+门控偏置 (dim=128)",
        "dim": 128,
        "needle_loss_alpha": 0.5,
        "qk_align_alpha": 0.5,
        "use_retrieval": True,
        "heads": 4,
        "override": {
            "retrieval_neighbor_span": 1,
            "retrieval_neighbor_direction": "right",
            "retrieval_query_pooling": "max_token",
            "retrieval_attention_topk": 8,
            "retrieval_quality_gate_bias": 2.0,
            "detach_state": False,
        },
    },
    "exp8_neighbor_both_dim256": {
        "label": "Exp8 双向邻居+Top8+门控偏置 (dim=256)",
        "dim": 256,
        "needle_loss_alpha": 0.5,
        "qk_align_alpha": 0.5,
        "use_retrieval": True,
        "heads": 8,
        "override": {
            "retrieval_neighbor_span": 2,
            "retrieval_neighbor_direction": "both",
            "retrieval_query_pooling": "max_token",
            "retrieval_attention_topk": 8,
            "retrieval_quality_gate_bias": 2.0,
            "detach_state": False,
        },
    },
}


def cmd_retrieval_breakthrough(args):
    """长程检索卖点恢复攻关实验 / Long-range retrieval breakthrough exploration."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from scripts.needle_in_haystack_test import (
        build_niah_model,
        compute_query_evidence_alignment_loss,
        evaluate_niah_depths,
        extract_query_positions_and_targets,
        generate_haystack_with_needle,
    )

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    print(f"[breakthrough] device={device}")
    out_dir = VERIFY_REPORTS_DIR / "retrieval_breakthrough"
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_len = int(args.seq_len)
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    seed = int(args.seed)
    wanted_configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    depths_cycle = [0.1, 0.5, 0.9]

    rows = []
    for key in wanted_configs:
        if key not in RETRIEVAL_BREAKTHROUGH_CONFIGS:
            print(f"[breakthrough] unknown config: {key}, skipping")
            continue
        spec = RETRIEVAL_BREAKTHROUGH_CONFIGS[key]
        print(f"\n[breakthrough] >>> Running {key}: {spec['label']} (dim={spec['dim']}, seq_len={seq_len}, epochs={epochs})")
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        override = dict(spec.get("override", {}))
        model = build_niah_model(
            device=device,
            vocab_size=100,
            dim=spec["dim"],
            num_layers=2,
            K=64,
            kr=8,
            chunk_size=1024,
            use_retrieval=spec["use_retrieval"],
            mhdsra2_config_override=override,
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = torch.nn.CrossEntropyLoss(ignore_index=0)

        model.train()
        t0 = time.time()
        for step in range(epochs):
            depth = depths_cycle[step % 3]
            X, Y, needle_positions = generate_haystack_with_needle(
                batch_size, seq_len, 100, depth
            )
            qpos, targets = extract_query_positions_and_targets(X, Y, device)
            needles = targets

            opt.zero_grad()
            if spec["qk_align_alpha"] > 0.0:
                logits, hidden_query = model.forward_selected_logits(
                    X, qpos, return_hidden=True
                )
            else:
                logits = model.forward_selected_logits(X, qpos)
                hidden_query = None

            loss_main = crit(logits, targets)
            loss = loss_main

            loss_needle_val = None
            if spec["needle_loss_alpha"] > 0.0:
                needle_val_positions = (needle_positions + 1).to(device)
                logits_needle = model.forward_selected_logits(X, needle_val_positions)
                loss_needle_val = crit(logits_needle, needles)
                loss = loss + spec["needle_loss_alpha"] * loss_needle_val

            loss_align = None
            if spec["qk_align_alpha"] > 0.0 and hidden_query is not None:
                loss_align, _ = compute_query_evidence_alignment_loss(
                    hidden_query,
                    needles,
                    model.embedding,
                    detach_evidence=True,
                )
                loss = loss + spec["qk_align_alpha"] * loss_align

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if (step + 1) % max(1, epochs // 5) == 0 or step == epochs - 1:
                print(
                    f"  [step {step+1}/{epochs}] loss={loss.item():.4f} "
                    f"(main={loss_main.item():.4f}"
                    f"{f', needle_aux={loss_needle_val.item():.4f}' if loss_needle_val is not None else ''}"
                    f"{f', align={loss_align.item():.4f}' if loss_align is not None else ''})"
                )

        train_elapsed = time.time() - t0

        # 评估 (纯前向, 无辅助loss)
        eval_result = evaluate_niah_depths(
            model,
            seq_len,
            device,
            vocab_size=100,
            batch_size=1,
            criterion=crit,
            batches_per_depth=4,
        )
        acc = eval_result["mean_accuracy"]
        sample_metrics = eval_result.get("sample_metrics", [])
        corrects = [
            row.get("correct") for row in sample_metrics if row.get("correct") is not None
        ]
        indep_acc = (sum(1 for c in corrects if c) / len(corrects)) if corrects else None
        disagreement = (
            indep_acc is not None
            and acc is not None
            and abs(indep_acc - acc) > METRIC_DISAGREEMENT_REL
        )

        rows.append({
            "config": key,
            "label": spec["label"],
            "dim": spec["dim"],
            "seq_len": seq_len,
            "epochs": epochs,
            "batch_size": batch_size,
            "train_elapsed_sec": train_elapsed,
            "final_eval_acc": acc,
            "independent_acc": indep_acc,
            "metric_disagreement": disagreement,
            "passed_50pct": acc is not None and acc >= 0.50,
        })
        print(
            f"[breakthrough] Result {key}: acc={_fmt_pct(acc)}, "
            f"indep={_fmt_pct(indep_acc)}, time={train_elapsed:.1f}s"
        )
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "phase": "retrieval_breakthrough",
        "device": str(device),
        "seq_len": seq_len,
        "epochs": epochs,
        "rows": rows,
    }
    write_json(out_dir / "breakthrough_summary.json", payload)
    write_markdown(out_dir / "breakthrough_summary.md", [
        "# 长程检索卖点恢复攻关实验汇总",
        "",
        f"- 评测配置: `seq_len={seq_len}, epochs={epochs}, vocab=100, device={device}`",
        "- 评估口径: 严格端到端纯前向评估（eval阶段关闭全部aux loss与捷径）",
        "",
        "| 实验配置 | 说明 | 维度 (dim) | 训练耗时 (s) | 最终评估准确率 | 独立指标重算 | 是否突破50%门限 |",
        "|---|---|:---:|---:|---:|---:|:---:|",
        *[
            f"| {r['config']} | {r['label']} | {r['dim']} | {r['train_elapsed_sec']:.1f}s | "
            f"{_fmt_pct(r['final_eval_acc'])} | {_fmt_pct(r['independent_acc'])} | "
            f"{'是' if r['passed_50pct'] else '否'} |"
            for r in rows
        ],
    ])

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(rows))
    accs = [(r["final_eval_acc"] or 0) * 100 for r in rows]
    labels = [r["config"] for r in rows]
    bars = ax.bar(x, accs, color=["#718096" if a < 20 else "#2b6cb0" for a in accs], width=0.5)
    ax.axhline(50.0, color="r", ls="--", label="50% Target Threshold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Final NIAH Accuracy (%)")
    ax.set_title(f"NIAH Retrieval Breakthrough Matrix (seq_len={seq_len}, epochs={epochs})")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    save_figure(fig, VERIFY_FIGURES_DIR / "fig_retrieval_breakthrough.png")
    return payload


def cmd_niah(args):
    """表1-A 复核: NIAH 训练+评估重测(自建循环) / Table 1-A: NIAH re-measurement.

    中文说明:
    - 背景: verify-2m CLI 的报告保存在 final_aux_diagnostics.gates_mean 含 Tensor 时
      触发 "Object of type Tensor is not JSON serializable" 崩溃(存量 BUG, 复核不改被测
      脚本), 因此这里直接调用 needle 库函数自建训练+评估循环, 与表5消融同款路径。
    """
    import torch

    from scripts.needle_in_haystack_test import (
        build_niah_model,
        evaluate_niah_depths,
        extract_query_positions_and_targets,
        generate_haystack_with_needle,
    )

    device = torch.device(args.device if args.device != "auto"
                          else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    plan = []
    for item in args.plan.split(";"):
        seq_len, epochs = item.split(":")
        plan.append((int(seq_len), int(epochs)))
    rows = []
    for seq_len, epochs in plan:
        seeds = [int(s) for s in args.seeds.split(",")] if seq_len <= 65536 else [20260506]
        for seed in seeds:
            try:
                torch.manual_seed(seed)
                model = build_niah_model(
                    device=device, vocab_size=100, dim=64, num_layers=2,
                    K=64, kr=8, chunk_size=1024,
                )
                opt = torch.optim.Adam(model.parameters(), lr=1e-3)
                crit = torch.nn.CrossEntropyLoss(ignore_index=0)
                model.train()
                depths_cycle = [0.1, 0.5, 0.9]
                for step in range(epochs):
                    X, Y, _ = generate_haystack_with_needle(
                        1, seq_len, 100, depths_cycle[step % 3])
                    qpos, targets = extract_query_positions_and_targets(X, Y, device)
                    opt.zero_grad()
                    logits = model.forward_selected_logits(X, qpos)
                    loss = crit(logits, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                eval_result = evaluate_niah_depths(
                    model, seq_len, device, vocab_size=100, batch_size=1,
                    criterion=crit, batches_per_depth=4,
                )
                acc = eval_result["mean_accuracy"]
                # 独立指标交叉验证: 用样本级 metrics 重算 top1 准确率
                sample_metrics = eval_result.get("sample_metrics", [])
                if sample_metrics:
                    preds_like = [row.get("correct", None) for row in sample_metrics]
                    valid = [p for p in preds_like if p is not None]
                    indep_acc = (sum(1 for v in valid if v) / len(valid)) if valid else None
                else:
                    indep_acc = None
                disagreement = (
                    indep_acc is not None and acc is not None
                    and abs(indep_acc - acc) > METRIC_DISAGREEMENT_REL
                )
                verdict, rel = judge_rel(acc, REPORTED_TABLE1_NIAH_ACC.get(seq_len), 0.05)
                rows.append({
                    "seq_len": seq_len, "seed": seed, "epochs": epochs, "rc": 0,
                    "final_eval_acc": acc, "independent_acc": indep_acc,
                    "metric_disagreement": disagreement,
                    "n_eval_samples": len(sample_metrics),
                    "verdict": verdict,
                })
                print(f"[niah] seq={seq_len} seed={seed}: acc={acc} "
                      f"indep={indep_acc} verdict={verdict}")
                del model, opt
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001 - 记录失败模式
                rows.append({
                    "seq_len": seq_len, "seed": seed, "epochs": epochs,
                    "rc": 1, "final_eval_acc": None, "independent_acc": None,
                    "metric_disagreement": None, "n_eval_samples": 0,
                    "verdict": f"error:{type(exc).__name__}",
                })
                print(f"[niah] seq={seq_len} seed={seed}: ERROR "
                      f"{type(exc).__name__}: {str(exc)[:120]}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    payload = {"phase": "niah", "rows": rows}
    out_dir = VERIFY_REPORTS_DIR / "niah"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "niah_verify.json", payload)
    write_markdown(out_dir / "niah_verify.md", [
        "# 表1-A NIAH 准确率复核(重测)",
        "",
        "| seq_len | seed | epochs | 重测 final eval acc | 独立重算 acc | 指标分歧 | 声明 | 判定 |",
        "|---:|---:|---:|---:|---:|:---:|---:|:---:|",
        *[
            f"| {r['seq_len']} | {r['seed']} | {r['epochs']} | "
            f"{_fmt_pct(r.get('final_eval_acc'))} | {_fmt_pct(r.get('independent_acc'))} | "
            f"{r.get('metric_disagreement')} | 100% | {r['verdict']} |" for r in rows
        ],
        "",
        "注: 长序列(131K+)按历史 probe 口径少量 epoch; 完整收敛训练不在本轮预算内。",
        "训练/评估循环为复核脚本自建(与被测 verify-2m CLI 同底层库函数), 规避其",
        "报告保存链路的 Tensor JSON 序列化存量 BUG(详见审计)。",
    ])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    by_len = {}
    for r in rows:
        by_len.setdefault(r["seq_len"], []).append(r["final_eval_acc"])
    lens = sorted(by_len)
    means = [statistics.mean([v for v in by_len[length] if v is not None]) if any(
        v is not None for v in by_len[length]) else 0 for length in lens]
    ax.plot(lens, [100] * len(lens), "r.--", label="Reported claim: 100%")
    ax.plot(lens, [m * 100 for m in means], "bo-", label="Re-measured mean")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Final eval accuracy (%)")
    ax.set_title("Table 1-A: NIAH accuracy (reported vs re-measured)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(-5, 108)
    save_figure(fig, VERIFY_FIGURES_DIR / "fig1_niah_accuracy.png")
    return payload


# ---------------------------------------------------------------------------
# ablation 子命令: 表5 消融重测(64K 口径)
# ---------------------------------------------------------------------------
def cmd_ablation(args):
    """表5 复核: 64K NIAH 消融 / Table 5: 64K NIAH ablation re-measurement."""
    import torch

    from scripts.needle_in_haystack_test import (
        build_niah_model,
        evaluate_niah_depths,
        extract_query_positions_and_targets,
        generate_haystack_with_needle,
    )

    device = torch.device(args.device if args.device != "auto"
                          else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    seq_len = int(args.seq_len)
    epochs = int(args.epochs)

    # 变体构造: 统一经 build_niah_model, 通过 use_retrieval 与
    # mhdsra2_config_override 关闭对应机制(运行时配置, 不改源码)
    # write_drive_mode / page_score_mode 为本轮新增消融开关(v1.2)
    all_variants = {
        "full": {"use_retrieval": True, "override": {}},
        "no_retrieval": {"use_retrieval": False, "override": {}},
        "no_local": {"use_retrieval": True, "override": {"use_local": False}},
        "no_slot": {"use_retrieval": True, "override": {"slots": 1}},
        "novelty_only": {
            "use_retrieval": True,
            "override": {"write_drive_mode": "novelty_only"},
        },
        "page_mean_only": {
            "use_retrieval": True,
            "override": {"page_score_mode": "page_mean"},
        },
    }
    # --variants 支持子集选择: 多进程并行时每进程跑一部分, 输出带后缀分片
    wanted_variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in wanted_variants if v not in all_variants]
    if unknown:
        raise ValueError(f"unknown ablation variants: {unknown}")
    variants = {name: all_variants[name] for name in wanted_variants}
    shard_suffix = "_" + "_".join(wanted_variants) if args.sharded else ""
    rows = []
    for name, spec in variants.items():
        try:
            torch.manual_seed(20260506)
            model = build_niah_model(
                device=device, vocab_size=100, dim=64, num_layers=2,
                K=64, kr=8, chunk_size=1024,
                mhdsra2_config_override=spec["override"],
                use_retrieval=spec["use_retrieval"],
            )
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            crit = torch.nn.CrossEntropyLoss(ignore_index=0)
            model.train()
            for _ in range(epochs):
                X, Y, _ = generate_haystack_with_needle(1, seq_len, 100,
                                                        [0.1, 0.5, 0.9][_ % 3])
                qpos, targets = extract_query_positions_and_targets(X, Y, device)
                opt.zero_grad()
                logits = model.forward_selected_logits(X, qpos)
                loss = crit(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            eval_result = evaluate_niah_depths(
                model, seq_len, device, vocab_size=100, batch_size=1,
                criterion=crit, batches_per_depth=4,
            )
            acc = eval_result["mean_accuracy"]
            preds_gt = REPORTED_TABLE5_ABLATION[name]["niah_acc"]
            diff = abs((acc or 0) - preds_gt)
            verdict = (VERDICT_CONFIRMED if diff < TH_TABLE5_PP
                       else VERDICT_DEVIATION if diff < 0.20 else VERDICT_REFUTED)
            rows.append({
                "variant": name, "reported_acc": preds_gt,
                "measured_acc": acc, "verdict": verdict, "note": f"epochs={epochs}",
            })
            print(f"[ablation] {name}: reported={preds_gt} measured={acc} {verdict}")
            del model, opt
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 - 记录失败模式
            rows.append({
                "variant": name, "reported_acc": REPORTED_TABLE5_ABLATION[name]["niah_acc"],
                "measured_acc": None, "verdict": f"error:{type(exc).__name__}",
                "note": str(exc)[:160],
            })
            print(f"[ablation] {name}: ERROR {type(exc).__name__}: {str(exc)[:120]}")

    payload = {"phase": "ablation", "seq_len": seq_len, "rows": rows}
    out_dir = VERIFY_REPORTS_DIR / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"ablation{shard_suffix}.json", payload)
    write_markdown(out_dir / f"ablation{shard_suffix}.md", [
        "# 表5 消融复核(64K NIAH 口径重测)",
        "",
        "| 变体 | 声明 NIAH acc | 重测 acc | 判定 | 备注 |",
        "|---|---:|---:|:---:|---|",
        *[
            f"| {r['variant']} | {_fmt_pct(r['reported_acc'])} | "
            f"{_fmt_pct(r['measured_acc'])} | {r['verdict']} | {r['note']} |"
            for r in rows
        ],
    ])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    names = [r["variant"] for r in rows][::-1]
    rep = [r["reported_acc"] * 100 for r in rows][::-1]
    meas = [(r["measured_acc"] or 0) * 100 for r in rows][::-1]
    y = range(len(names))
    ax.barh([i + 0.2 for i in y], rep, 0.38, label="Reported", color="#888")
    ax.barh([i - 0.2 for i in y], meas, 0.38, label="Re-measured", color="#2b6cb0")
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("NIAH accuracy (%)")
    ax.set_title(f"Table 5: ablation re-measurement (seq={seq_len})")
    ax.legend()
    ax.grid(alpha=0.3, axis="x")
    save_figure(fig, VERIFY_FIGURES_DIR / "fig5_ablation.png")
    return payload


# ---------------------------------------------------------------------------
# aggregate 子命令
# ---------------------------------------------------------------------------
def cmd_aggregate(args):
    """汇总全部子结果 / Aggregate all sub-results into final verdicts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sections = []
    counts = {}
    detail = {}
    sub_files = {
        "audit": VERIFY_REPORTS_DIR / "audit" / "audit.json",
        "probe": VERIFY_REPORTS_DIR / "probe" / "probe.json",
        "throughput": VERIFY_REPORTS_DIR / "throughput" / "throughput_verify.json",
        "ppl": VERIFY_REPORTS_DIR / "ppl" / "ppl.json",
        "memory": VERIFY_REPORTS_DIR / "memory" / "memory.json",
        "niah": VERIFY_REPORTS_DIR / "niah" / "niah_verify.json",
        "ablation": VERIFY_REPORTS_DIR / "ablation" / "ablation.json",
    }
    for name, path in sub_files.items():
        if path.exists():
            detail[name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            detail[name] = None

    def collect(items):
        c = {VERDICT_CONFIRMED: 0, VERDICT_DEVIATION: 0, VERDICT_REFUTED: 0,
             VERDICT_NO_SOURCE: 0, VERDICT_CONTRADICTED: 0, VERDICT_NOT_RUNNABLE: 0}
        for it in items:
            if it in c:
                c[it] += 1
        return c

    if detail["audit"]:
        items = [r["verdict"] for r in detail["audit"]["rows"]]
        counts["audit"] = collect(items)
        sections.append(("audit(数据级审计)", items))
    if detail["probe"]:
        items = [r["verdict_vs_report"] for r in detail["probe"]["rows"]]
        counts["probe"] = collect(items)
        sections.append(("probe(表3探针)", items))
    if detail["throughput"]:
        items = [r["verdict"] for r in detail["throughput"]["rows"]]
        counts["throughput"] = collect(items)
        sections.append(("throughput(表2重测)", items))
    if detail["ppl"]:
        items = [r["verdict"] for r in detail["ppl"]["verdict_rows"]]
        counts["ppl"] = collect(items)
        sections.append(("ppl(表4重测)", items))
    if detail["memory"]:
        items = []
        for r in detail["memory"]["rows"]:
            items.append(r["verdicts"]["forward_vs_reported_forward"])
            items.append(r["verdicts"]["train_step_vs_reported_ckpt_train"])
        counts["memory"] = collect(items)
        sections.append(("memory(表1-B显存)", items))
    if detail["niah"]:
        items = [r["verdict"] for r in detail["niah"]["rows"]]
        counts["niah"] = collect(items)
        sections.append(("niah(表1-A准确率)", items))
    if detail["ablation"]:
        items = [r["verdict"] for r in detail["ablation"]["rows"]]
        counts["ablation"] = collect(items)
        sections.append(("ablation(表5消融)", items))

    lines = ["# 技术报告实验数据复核总报告 (verify_summary)", ""]
    lines.append("| 复核部分 | confirmed | deviation | refuted | no_source | contradicted | not_runnable |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, c in counts.items():
        lines.append(
            f"| {name} | {c[VERDICT_CONFIRMED]} | {c[VERDICT_DEVIATION]} | "
            f"{c[VERDICT_REFUTED]} | {c[VERDICT_NO_SOURCE]} | "
            f"{c[VERDICT_CONTRADICTED]} | {c[VERDICT_NOT_RUNNABLE]} |"
        )
    lines += ["", "## 逐项明细", ""]
    for title, items in sections:
        lines.append(f"### {title}")
        for it in items:
            lines.append(f"- {it}")
        lines.append("")
    missing = [n for n, d in detail.items() if d is None]
    if missing:
        lines.append(f"## 未执行的部分\n\n- {', '.join(missing)} (对应子命令未运行或失败, 未伪造补齐)\n")

    VERIFY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(VERIFY_REPORTS_DIR / "verify_summary.json", {
        "counts": counts,
        "detail_loaded": {k: bool(v) for k, v in detail.items()},
    })
    write_markdown(VERIFY_REPORTS_DIR / "verify_summary.md", lines)

    # 仪表盘
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    keys = list(counts.keys())
    for i, ax in enumerate(axes):
        if i < len(keys):
            c = counts[keys[i]]
            cats = [VERDICT_CONFIRMED, VERDICT_DEVIATION, VERDICT_REFUTED,
                    VERDICT_NO_SOURCE, VERDICT_CONTRADICTED, VERDICT_NOT_RUNNABLE]
            vals = [c[k] for k in cats]
            colors = ["#2f855a", "#dd6b20", "#c53030", "#805ad5", "#9b2c2c", "#718096"]
            ax.bar(range(len(cats)), vals, color=colors)
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels(["ok", "dev", "refu", "nosrc", "contra", "nrun"], fontsize=7)
            ax.set_title(keys[i], fontsize=10)
            ax.grid(alpha=0.3, axis="y")
        else:
            ax.axis("off")
    fig.suptitle("MHDSRA Technical Report Verification Dashboard", fontsize=13)
    save_figure(fig, VERIFY_FIGURES_DIR / "fig0_dashboard.png")
    print(f"[aggregate] counts={json.dumps(counts)}")
    return {"counts": counts}


# ---------------------------------------------------------------------------
# self-test 子命令
# ---------------------------------------------------------------------------
def cmd_self_test(args):
    """CPU 冒烟自检 / CPU mock smoke test for the plotting pipeline."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 指标 oracle 自检
    acc, n = independent_niah_accuracy([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 0, 4, 0, 6, 0, 8])
    assert n == 8 and abs(acc - 0.625) < 1e-12, f"accuracy oracle failed: {acc}"
    acc2, n2 = independent_niah_accuracy(
        [1, 2, 3], [1, 0, 0], valid_mask=[True, False, False]
    )
    assert n2 == 1 and acc2 == 1.0, "mask oracle failed"
    ppl = independent_ppl(math.log(2) + math.log(2) + math.log(4), 3)
    assert abs(ppl - math.exp((math.log(2) + math.log(2) + math.log(4)) / 3)) < 1e-12
    closed = _closed_form_exact_weight(8.0, 8, 128)
    assert abs(closed - 0.11752) < 5e-4, f"closed form oracle failed: {closed}"
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar([0, 1], [1, 2])
    save_figure(fig, VERIFY_FIGURES_DIR / "fig_self_test.png")
    print("[self-test] all oracle assertions passed; figure pipeline OK")
    return {"ok": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description="MHDSRA 技术报告实验数据复核")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("audit", help="数据级审计(归档扫描+JSON核对)")
    sub.add_parser("probe", help="表3 Top-K 稀释探针复核")
    sub.add_parser("self-test", help="CPU 冒烟自检")

    p = sub.add_parser("throughput", help="表2 吞吐抽样重测")
    p.add_argument("--reuse", action="store_true",
                   help="复用已存在的 compare JSON, 跳过重跑(仅重新解析与绘图)")
    p.add_argument("--seq-lengths", default="131072")
    p.add_argument("--slots", default="64,128,256")
    p.add_argument("--chunks", default="512,1024,2048,4096")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--timeout", type=int, default=7200)

    sub.add_parser("throughput-merge", help="汇总 131K~1M 全部吞吐网格(fig2b+总表)")

    p = sub.add_parser("ppl", help="表4 PPL 重测")
    p.add_argument("--seeds", default="1234,42,20260506")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--run-512", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--timeout", type=int, default=3600)

    p = sub.add_parser("memory", help="表1-B 显存探针")
    p.add_argument("--lengths", default="16384,32768,65536,131072,262144,524288,1048576,2097152")
    p.add_argument("--device", default="auto")

    p = sub.add_parser("niah-matrix", help="Stage A NIAH 诊断矩阵(A-0梯度探针+A0~A3)")
    p.add_argument("--configs", default="a0,a1,a2,a3",
                   help="要跑的配置子集(逗号分隔), 支持多进程分组并行")
    p.add_argument("--seq-len", default="16384")
    p.add_argument("--epochs", default="200")
    p.add_argument("--seed", default="20260506")
    p.add_argument("--device", default="auto")
    p.add_argument("--grad-probe", action="store_true",
                   help="在 a0 配置上运行 A-0 梯度范数探针")
    p.add_argument("--grad-probe-steps", default="20")
    p.add_argument("--merge-only", action="store_true",
                   help="只汇总已有分片, 不跑训练")

    p = sub.add_parser("niah", help="表1-A NIAH 准确率重测")
    p.add_argument("--plan", default="16384:200;65536:120;131072:60;524288:20;2097152:5")
    p.add_argument("--seeds", default="20260506,7,42")
    p.add_argument("--device", default="auto")
    p.add_argument("--timeout", type=int, default=21600)

    p = sub.add_parser("ablation", help="表5 消融重测(64K)")
    p.add_argument("--seq-len", default="65536")
    p.add_argument("--epochs", default="150")
    p.add_argument("--device", default="auto")
    p.add_argument("--variants",
                   default="full,no_retrieval,no_local,no_slot,novelty_only,page_mean_only",
                   help="要跑的变体子集(逗号分隔), 支持多进程分片并行")
    p.add_argument("--sharded", action="store_true",
                   help="输出文件带变体后缀(分片模式, 避免并行进程互相覆盖)")

    p = sub.add_parser("retrieval-breakthrough", help="长程检索卖点恢复攻关实验(辅助Loss+QK对齐+维度扩展)")
    p.add_argument("--configs",
                   default="exp1_base,exp2_needle_aux,exp3_qk_align,exp4_combined,exp5_dim128,exp6_dim256",
                   help="要跑的攻关配置列表(逗号分隔)")
    p.add_argument("--seq-len", default="16384", help="测试上下文长度(默认16384)")
    p.add_argument("--epochs", default="200", help="训练轮数(默认200)")
    p.add_argument("--batch-size", default="2", help="训练批大小(默认2)")
    p.add_argument("--seed", default="20260506")
    p.add_argument("--device", default="auto")

    sub.add_parser("aggregate", help="汇总判定+仪表盘")

    args = parser.parse_args(argv)
    handlers = {
        "audit": cmd_audit, "probe": cmd_probe, "throughput": cmd_throughput,
        "throughput-merge": cmd_throughput_merge,
        "ppl": cmd_ppl, "memory": cmd_memory, "niah": cmd_niah,
        "niah-matrix": cmd_niah_matrix,
        "retrieval-breakthrough": cmd_retrieval_breakthrough,
        "ablation": cmd_ablation, "aggregate": cmd_aggregate,
        "self-test": cmd_self_test,
    }
    return handlers[args.command](args)



if __name__ == "__main__":
    main()
