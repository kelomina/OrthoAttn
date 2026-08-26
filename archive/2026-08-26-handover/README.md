# DSRA 接手文档 — 2026-08-26

> 归档目录：`archive/2026-08-26-handover/`（76 个文件，含本会话全部 A/B JSON、checkpoint .pt、诊断日志）
> 分支：`main` @ `1a9ed57`（已推送至 `kelomina/OrthoAttn`）

## 一、原始任务

修复 **NIAH 长程插针准确率**。6 月历史在 `needle_in_haystack_test.py` 上阶段性训练（L=8192/60 epochs）全线失败，最佳 0–16.7%，诊断显示证据召回成功但权重上不去（hit_rate=1.0, weight 低）。

## 二、本会话实际执行路径

1. **MQAR 分支**：沿 `walkthrough.md` 验证 Path A（缩小 chunk 1024）→ 证伪（Query 分离至 42/53 仍 50%）；进而按 Query 拆分+串扰统计与三路强制路由（slot-only 100% @700 步，retrieval-only 坍缩），定位“检索支路垄断→绑定坍缩”。
2. **门控修复分支**：`retrieval_quality_gate_bias=-4` 在 MQAR-K2/L64 上解锁至 100%/100%（门控锁定均衡 29/34/38），并证明阈值在 (-4,-2]；单变量归因矩阵完成。
3. **RULER 权威基准落地**：实现 `src/dsra/domain/ruler_niah.py`（逐字对齐 NVIDIA/RULER niah.py：噪声句、针句模板、adj-noun 键、7 位数的逐数字位 token 化），`scripts/benchmark_ruler_niah.py`（gate 轨迹、require-cuda 取证、checkpoint 保存），`tests/test_ruler_niah_data_generation.py`（现 9 项）。
4. **性能事故修复**：py-spy 定位 `paged_exact_memory.py:352` 邻居扩展纯 Python 循环为 718% CPU 炉；置 `neighbor_span=0`、chunk 默认 1024、`torch.set_num_threads(8)`，GPU 4%→83%，步速 15s→4.8s。
5. **幻觉复查与训练循环 Bug**：两次复查后修复 `evaluate_ruler_niah_exact_match` 死函数及其背书测试、训练循环固定 seed 导致的数据冻结（`loss 0.0000` 假象）、`--eval-batches` help 中 `%` 未转义。
6. **RULER A/B 链**：S-NIAH-1（h192≈3.5K 上下文，3000 步）于修复后重启，当前 bias0 臂跑至 ~500/3000，后台低优先级运行中。

## 三、已验证结论（按证据强度）

- **门控垄断机制在 MQAR 与 RULER 上均复现**：基线检索门控会漂移至 >50% 并伴随坍缩，负偏置可锁定均衡。
- **单针 100% vs 多针 50% 的核心矛盾已解释为读出绑定失败而非容量**（值召回无损、xtalk 补满 100%）。
- **RULER 任务形式化是第一阻塞点**：7 位精确复制对 dim128/2L 从零模型不可学（Transformer 对照在 d4 变体满分而 7 位版 0%），历史 NIAH 失败首先是此层问题。
- **待验证**：`-4` 在 RULER-S-NIAH-1 上的跨尺度迁移（本次重启的 A/B 双臂即为判定实验）。

## 四、当前代码状态

- 已提交推送：
  - `b03eb7c` 门控熵正则（默认关闭，含梯度流测试）
  - `b4db270` RULER 生成器与基准脚本
  - `e88cd87` 评测器 off-by-one 修复
  - `1a9ed57` 逐训练步重播种修复
- 未提交生产改动：`archive/2026-08-26-handover/` 本身待推送；`docs/` 全目录按 `.gitignore` 本地化（案例库追加 10–13 在本地）
- **已排队 P2**：`_build_vocab` 改用 `_tokenize` 构造以消除垃圾 token `0.` —— 会改变词表，需在 A/B 结束后执行。

## 五、如何复跑

```bash
# 冒烟
python scripts/benchmark_ruler_niah.py --num-haystack 48 --epochs 60 --eval-interval 30 --device cuda:0 --output-json /tmp/smoke.json
# A/B 单臂（示例：bias=-4）
python scripts/benchmark_ruler_niah.py --num-haystack 192 --epochs 3000 --eval-interval 50 --retrieval-quality-gate-bias -4 --device cuda:0 --output-json reports/ruler_niah_s1_bias-4_gpu.json
```

## 六、接手必读风险

- **共享 GPU**：本机 RTX 4070 与 `asi-lab` 共用；训练务必用低优先级包装器 `run_bg_retry.ps1`（BelowNormal + 重试≤8），并尊重“只能用 GPU”纪律。
- **已杀进程**：本轮终止的后台链为 `ruler_ab_chain.ps1` 及其两个 python 子进程，已按 PID 精确清理；残留显存约 2.8GB 需重启回收。
- **P2 词表变更会使在跑 checkpoint 失效**，务必等当前双臂落盘后再动。

## 七、下一步建议（按优先级）

1. 等当前双臂落盘，读 `first_digit_acc`（纯检索信号）与 EM 对比，判定 `-4` 在权威基准是否成立；
2. 执行 P2；
3. 在 d4 变体（4 位数值）上补跑 Transformer 对照臂，确认 RULER 任务本身的可学性边界。
