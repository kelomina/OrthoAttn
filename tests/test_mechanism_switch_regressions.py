"""机制开关回归测试: write_drive_mode 与 page_score_mode.

中文说明:
- 调用方 / Called by: pytest
- 被测对象: `MHDSRA2Config.write_drive_mode`(覆盖感知门控开关) 与
  `PagedExactMemory.page_score_mode`(两级页面评分开关)
- 作用: 验证 (1) 非法值拒绝; (2) 开关真实接线(同输入下新模式产生不同输出,
  防"参数存在但不接线"的幻觉开关); (3) novelty_only 复现"同键覆写死锁"方向;
  (4) page_mean 复现"单 needle 被整页稀释漏检"方向; (5) 开关透传链路完整;
  (6) 默认值路径与显式默认值路径逐位一致。
"""

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dsra.dsra_model import MultiLayerMHDSRA2Model  # noqa: E402
from src.dsra.infrastructure.paged_memory_repository import (  # noqa: E402
    PagedMemoryRepository,
)
from src.dsra.mhdsra2.improved_dsra_mha import (  # noqa: E402
    MHDSRA2Config,
    MultiHeadDSRA2,
)
from src.dsra.mhdsra2.paged_exact_memory import PagedExactMemory  # noqa: E402


def _tiny_cfg(**overrides) -> MHDSRA2Config:
    base = dict(
        dim=8, heads=1, slots=4, read_topk=2, write_topk=2,
        local_window=4, use_local=False, use_retrieval=False,
        detach_state=False,
    )
    base.update(overrides)
    return MHDSRA2Config(**base)


# ---------------------------------------------------------------------------
# 配置校验 / Config validation
# ---------------------------------------------------------------------------
def test_write_drive_mode_rejects_invalid_value():
    with pytest.raises(ValueError, match="write_drive_mode"):
        _tiny_cfg(write_drive_mode="bogus")


def test_page_score_mode_rejects_invalid_value():
    with pytest.raises(ValueError, match="page_score_mode"):
        _tiny_cfg(page_score_mode="bogus")


def test_paged_memory_page_score_mode_rejects_invalid_value():
    with pytest.raises(ValueError, match="page_score_mode"):
        PagedExactMemory(page_size=4, page_score_mode="bogus")


# ---------------------------------------------------------------------------
# 默认值无漂移 / Default-value equivalence
# ---------------------------------------------------------------------------
def test_write_drive_default_equals_explicit_overwrite_aware():
    """默认构造与显式 overwrite_aware 在同输入下输出逐位一致."""
    torch.manual_seed(7)
    x = torch.randn(1, 8, 8)
    m_default = MultiHeadDSRA2(_tiny_cfg())
    m_explicit = MultiHeadDSRA2(_tiny_cfg(write_drive_mode="overwrite_aware"))
    m_default.load_state_dict(m_explicit.state_dict())
    s1 = m_default.init_state(1)
    s2 = m_explicit.init_state(1)
    y1, s1 = m_default(x, s1)
    y2, s2 = m_explicit(x, s2)
    torch.testing.assert_close(y1, y2)


# ---------------------------------------------------------------------------
# write_drive_mode 接线与行为 / Wiring & behavior
# ---------------------------------------------------------------------------
def test_novelty_only_changes_slot_updates():
    """开关真实接线: 同 seed 同输入下 novelty_only 产生不同的 slot 更新."""
    torch.manual_seed(11)
    x = torch.randn(1, 8, 8)
    m_aware = MultiHeadDSRA2(_tiny_cfg())
    m_novelty = MultiHeadDSRA2(_tiny_cfg(write_drive_mode="novelty_only"))
    m_novelty.load_state_dict(m_aware.state_dict())
    s1 = m_aware.init_state(1)
    s2 = m_novelty.init_state(1)
    _, s1 = m_aware(x, s1)
    _, s2 = m_novelty(x, s2)
    assert not torch.allclose(s1.slot_v, s2.slot_v), (
        "novelty_only did not change slot updates; switch may not be wired"
    )


def test_novelty_only_reproduces_overwrite_deadlock_direction():
    """novelty_only 下同键覆写变化量应小于 overwrite_aware(更新死锁方向).

    场景: 先以 key k 写入 v1, 再读该槽(产生 read_mass), 再以同 key k 写入 v2.
    overwrite_aware 依赖 read hint 强化同槽覆写; novelty_only 无该强化,
    且同 key novelty≈0, slot_v 写入量应更小.
    """
    torch.manual_seed(3)
    d = 8
    # 构造 chunk: 3 个 token, 全部同方向 key(同键), value 不同
    key_dir = torch.randn(d)
    key_dir = key_dir / key_dir.norm()
    k = key_dir.view(1, 1, 1, d).repeat(1, 1, 3, 1)
    v1 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).view(1, 1, 1, d).repeat(1, 1, 3, 1)
    v2 = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).view(1, 1, 1, d).repeat(1, 1, 3, 1)

    def run(mode):
        model = MultiHeadDSRA2(_tiny_cfg(write_drive_mode=mode))
        state = model.init_state(1)
        state = model._slot_write(k, v1, state, torch.zeros(1, model.cfg.heads, model.cfg.slots))
        # 模拟读回: read_mass 集中在第一个槽
        read_mass = torch.zeros(1, model.cfg.heads, model.cfg.slots)
        read_mass[..., 0] = 1.0
        state = model._slot_write(k, v2, state, read_mass)
        return state.slot_v

    slot_v_aware = run("overwrite_aware")
    slot_v_novelty = run("novelty_only")

    def alignment_to_v2(slot_v):
        # 取实际被写入的槽(范数最大者)与 v2 方向的对齐度
        norms = slot_v.reshape(-1, slot_v.shape[-1]).norm(dim=-1)
        written = slot_v.reshape(-1, slot_v.shape[-1])[int(norms.argmax())]
        return torch.nn.functional.cosine_similarity(
            written, v2[0, 0, 0], dim=0).item()

    # latest-wins 度量: 被写槽与新值 v2 的方向对齐度.
    # overwrite_aware 依赖 read hint 强化同槽覆写, 槽值应向 v2 移动;
    # novelty_only 下同键 novelty≈0 且无 read hint 强化, v2 写入弱,
    # 槽值仍主要停留在 v1 → 对 v2 对齐度更低(更新死锁方向).
    cos_aware = alignment_to_v2(slot_v_aware)
    cos_novelty = alignment_to_v2(slot_v_novelty)
    assert cos_novelty < cos_aware, (
        f"novelty_only alignment to new value ({cos_novelty:.4f}) should be below "
        f"overwrite_aware ({cos_aware:.4f}) in same-key overwrite scenario"
    )


def test_novelty_only_zeroes_overwrite_gate_diagnostic():
    """novelty_only 诊断字段 overwrite_gate_mean 应为 0(覆盖感知成分关闭)."""
    torch.manual_seed(5)
    x = torch.randn(1, 8, 8)
    model = MultiHeadDSRA2(_tiny_cfg(write_drive_mode="novelty_only"))
    state = model.init_state(1)
    model(x, state)
    stats = model.last_write_stats
    assert float(stats["overwrite_gate_mean"]) == 0.0


# ---------------------------------------------------------------------------
# page_score_mode 接线与行为 / Wiring & behavior
# ---------------------------------------------------------------------------
def _build_needle_page_memory(page_score_mode):
    """构造单 needle 页 + 中等相关干扰页的外部记忆.

    中文说明: page0 内 1 个与 query 完全对齐的 needle key + 15 个正交背景 key
    (页均值评分≈1/16≈0.06); page1 的 16 个 key 与 query 保持中等正相关
    (≈0.5, 页均值评分≈0.5)。因此纯 page_mean 初筛会选中 page1 而漏掉
    needle 所在的 page0; 两级评分下 page0 的 max_token=1.0 > page1 的 ≈0.5,
    仍选中 page0 并召回 needle token。
    """
    torch.manual_seed(13)
    d = 8
    memory = PagedExactMemory(page_size=16, dtype=torch.float32,
                              page_score_mode=page_score_mode)
    q_dir = torch.randn(d)
    q_dir = q_dir / q_dir.norm()
    needle_key = q_dir.clone()
    bg = torch.randn(15, d)
    bg = torch.nn.functional.normalize(bg, dim=-1)
    # 保证背景与 query 近正交
    bg = bg - (bg @ q_dir).unsqueeze(-1) * q_dir
    bg = torch.nn.functional.normalize(bg, dim=-1)
    page0_keys = torch.cat([needle_key.unsqueeze(0), bg], dim=0).unsqueeze(0)  # [H=1,T,d]
    # 干扰页: 每个 key 与 query 中等相关(cos≈0.5)
    distract = torch.randn(16, d)
    distract = distract - (distract @ q_dir).unsqueeze(-1) * q_dir
    distract = torch.nn.functional.normalize(distract, dim=-1)
    page1_keys = torch.nn.functional.normalize(
        0.5 * q_dir.unsqueeze(0) + 0.87 * distract, dim=-1
    ).unsqueeze(0)
    values0 = torch.zeros(1, 16, d)
    values0[0, 0] = 1.0  # needle value 可识别
    values1 = torch.zeros(1, 16, d)
    memory.append(page0_keys, values0)
    memory.append(page1_keys, values1)
    return memory, q_dir


def test_page_mean_mode_misses_single_needle_token():
    """两级评分召回 needle, 纯 page_mean 漏检(单 token 被整页均值稀释)."""
    mem_two, q_dir = _build_needle_page_memory("two_level")
    k, v, pos = mem_two.retrieve(q_dir.view(1, 1, 1, 8), top_pages=1, max_tokens=4)
    # two_level: 召回的 value 应包含 needle value(非零)
    assert v is not None and float(v.abs().sum()) > 1e-6

    mem_mean, q_dir2 = _build_needle_page_memory("page_mean")
    k2, v2, pos2 = mem_mean.retrieve(q_dir2.view(1, 1, 1, 8), top_pages=1, max_tokens=4)
    # page_mean: needle 特征被稀释, 召回的 token 值应为零或缺失
    assert v2 is None or float(v2.abs().sum()) < 1e-6, (
        "page_mean mode unexpectedly retrieved the diluted needle token"
    )


def test_page_mean_mode_retrieve_returns_well_formed_results():
    """page_mean 模式下 retrieve 正常返回(向量路径 smoke, 与 two_level 结果可区分)."""
    mem_mean, q_dir = _build_needle_page_memory("page_mean")
    k, v, pos = mem_mean.retrieve(q_dir.view(1, 1, 1, 8), top_pages=1, max_tokens=4)
    assert k is not None
    assert pos is not None
    assert k.shape[-1] == 8


# ---------------------------------------------------------------------------
# 透传链路 / Parameter threading
# ---------------------------------------------------------------------------
def test_repository_threads_page_score_mode():
    repo = PagedMemoryRepository(enabled=True, page_score_mode="page_mean")
    assert repo.memory.page_score_mode == "page_mean"
    repo_default = PagedMemoryRepository(enabled=True)
    assert repo_default.memory.page_score_mode == "two_level"


def test_model_config_threads_page_score_mode_to_repository():
    """MHDSRA2Config.page_score_mode 应经 _new_retrieval_repositories 传到仓储."""
    model = MultiLayerMHDSRA2Model(
        vocab_size=16, dim=8, num_layers=1, K=4, kr=2, chunk_size=4,
        use_retrieval=True,
        mhdsra2_config_override={"page_score_mode": "page_mean"},
    )
    repos = model._new_retrieval_repositories()
    assert repos[0].memory.page_score_mode == "page_mean"

    model_default = MultiLayerMHDSRA2Model(
        vocab_size=16, dim=8, num_layers=1, K=4, kr=2, chunk_size=4,
        use_retrieval=True,
    )
    repos_default = model_default._new_retrieval_repositories()
    assert repos_default[0].memory.page_score_mode == "two_level"
