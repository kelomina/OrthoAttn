"""快速训练测试脚本 - 提升 MHDSRA2 分支质量 (Round 3)"""
import sys
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pretrain_hybrid_lm import pretrain_hybrid_model

print("=" * 70)
print("快速训练测试 R3 - 提升 MHDSRA2 分支质量")
print("=" * 70)
print("改进项:")
print("  1. read_topk: 2 -> 8 (每token读取8个slot，4x上下文视野)")
print("  2. write_topk: 2 -> 4 (每token写入4个slot，2x写入带宽)")
print("  3. tau_write_init: 4.0 -> 8.0 (锐化写入路由，减少slot混叠)")
print("  4. eta: 1.0 -> 2.0 (加快写入门饱和，加速slot稳定)")
print("  5. 移除MHDSRA2内部位置偏置 (不再偏向local注意力)")
print("  6. 保留R2的门控修复 (gate_reg_weight=0.3, LN, clamp)")
print("=" * 70)

exp_name = f"mh_quality_r3_{datetime.now().strftime('%H%M%S')}"
print(f"Experiment: {exp_name}", flush=True)

pretrain_hybrid_model(
    dim=256, n_layers=4, n_heads=4, slots=128, seq_len=512,
    batch_size=8, lr=0.001, max_steps=2000, warmup_steps=100,
    max_epochs=3, grad_accum_steps=4,
    fuse_gate_frozen_steps=100, gate_reg_weight=0.3,
    output_dir="models/hybrid_lm",
    experiment_name=exp_name,
)
print("DONE", flush=True)
