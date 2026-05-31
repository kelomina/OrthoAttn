"""检查旧检查点的结构，分析门控数据缺失原因"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
DEVICE = torch.device('cpu')  # 先在CPU上检查

checkpoint_paths = [
    "models/hybrid_lm/best_model.pt",
    "models/hybrid_lm/checkpoint_epoch5.pt",
    "models/hybrid_lm_gate_fix/best_model.pt"
]

print("="*70)
print("检查检查点结构分析")
print("="*70)

for cp_path in checkpoint_paths:
    cp_path_full = PROJECT_ROOT / cp_path
    if not cp_path_full.exists():
        print(f"\n跳过不存在: {cp_path}")
        continue
    
    print(f"\n检查点: {cp_path}")
    print("-"*70)
    try:
        checkpoint = torch.load(cp_path_full, map_location=DEVICE, weights_only=False)
        print(f"检查点包含键: {list(checkpoint.keys())}")
        if "config" in checkpoint:
            print(f"配置: {checkpoint['config']}")
        if "model_state_dict" in checkpoint:
            model_state = checkpoint["model_state_dict"]
            print(f"模型状态字典键数: {len(model_state)}")
            fuse_gate_keys = [k for k in model_state.keys() if "fuse_gates" in k]
            print(f"fuse_gates 相关键: {len(fuse_gate_keys)} 个")
            for key in fuse_gate_keys:
                print(f"  - {key}: {model_state[key].shape}")
    except Exception as e:
        print(f"错误: {e}")

print("\n" + "="*70)
print("完成")
print("="*70)
