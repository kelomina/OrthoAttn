
import sys
sys.path.insert(0, r'e:\Project\python\DSRA')
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dsra.mhdsra2.improved_dsra_mha import MultiHeadDSRA2, MHDSRA2Config

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load(r'e:\Project\python\DSRA\models\hybrid_lm\best_model.pt', map_location=device, weights_only=False)

print('='*80)
print('fuse_gates 详细分析')
print('='*80)

for i in range(8):
    weight = ckpt['model_state_dict'][f'fuse_gates.{i}.weight']
    bias = ckpt['model_state_dict'][f'fuse_gates.{i}.bias']
    
    print(f'\nLayer {i}:')
    print(f'  bias = {bias.tolist()}')
    
    # 分割权重为 ST 部分和 MH 部分
    w_st = weight[:, :256]
    w_mh = weight[:, 256:]
    
    print(f'  w_st norm: {w_st.norm():.4f}')
    print(f'  w_mh norm: {w_mh.norm():.4f}')
    
    print(f'  w_st mean abs: {w_st.abs().mean():.4f}')
    print(f'  w_mh mean abs: {w_mh.abs().mean():.4f}')
    
    # 检查第一行（ST 门）和第二行（MH 门）
    print(f'  Gate 0 (ST) - w_st: {w_st[0].norm():.4f}, w_mh: {w_mh[0].norm():.4f}')
    print(f'  Gate 1 (MH) - w_st: {w_st[1].norm():.4f}, w_mh: {w_mh[1].norm():.4f}')

print()
print('='*80)
print('结论')
print('='*80)
print()
print('问题不只是输出幅度！fuse_gate 的权重矩阵本身可能也在向 ST 分支倾斜')
print('这说明门控崩溃是训练过程中形成的，修复需要从头训练')

