"""验证 Phase 2 随机种子 Bug 修复"""
import sys
import os

# 确保可以导入项目
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import random

SEED = 42

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def test_seed_consistency():
    """测试随机种子一致性"""
    print("=" * 60)
    print("测试随机种子一致性")
    print("=" * 60)
    
    # 第一次生成
    set_seed(SEED)
    data1 = torch.randn(5)
    
    # 第二次生成（不重置种子）
    data2 = torch.randn(5)
    
    # 第三次生成（重置种子）
    set_seed(SEED)
    data3 = torch.randn(5)
    
    print(f"\n第一次生成: {data1.numpy()}")
    print(f"第二次生成: {data2.numpy()} (不重置种子)")
    print(f"第三次生成: {data3.numpy()} (重置种子)")
    
    print(f"\ndata1 == data2: {torch.allclose(data1, data2)}")
    print(f"data1 == data3: {torch.allclose(data1, data3)}")
    
    assert torch.allclose(data1, data3), "❌ 随机种子重置失败"
    assert not torch.allclose(data1, data2), "❌ 随机种子未生效"
    
    print("\n✅ 随机种子测试通过")

if __name__ == "__main__":
    test_seed_consistency()
