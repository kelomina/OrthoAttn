"""包装脚本：运行训练并捕获所有异常"""
import sys
import os
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from datetime import datetime

try:
    from scripts.pretrain_hybrid_lm import pretrain_hybrid_model

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
    print("Training completed successfully!", flush=True)
    
except Exception:
    crash_log = os.path.join(PROJECT_ROOT, "scripts/train_crash.log")
    with open(crash_log, "w") as f:
        traceback.print_exc(file=f)
    print("CRASHED! See scripts/train_crash.log", flush=True)
    traceback.print_exc()
