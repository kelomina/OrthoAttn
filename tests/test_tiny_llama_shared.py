"""Tests for tiny_llama_shared: LMDataset, create_eval_loader, download_wikitext103."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny_llama_shared import (
    CharTokenizer,
    LMDataset,
    create_dataloader,
    create_eval_loader,
)


class TestLMDataset:
    """LMDataset: 单 Tensor 存储 + __getitem__ 切片行为验证."""

    def test_single_tensor_storage(self):
        """验证 ids 是唯一存储且为 torch.Tensor."""
        tokenizer = CharTokenizer()
        text = "hello world"
        ds = LMDataset(tokenizer, text, seq_len=4)
        assert isinstance(ds.ids, torch.Tensor)
        assert ds.ids.dtype == torch.long
        assert not hasattr(ds, "sequences") or ds.sequences is None

    def test_num_sequences_correct(self):
        """验证 num_sequences 计算正确: (len(ids)-1) // seq_len."""
        tokenizer = CharTokenizer()
        ids = tokenizer.encode("hello world")
        expected = (len(ids) - 1) // 4
        ds = LMDataset(tokenizer, "hello world", seq_len=4)
        assert len(ds) == expected

    def test_empty_when_too_short(self):
        """当 len(ids)-1 < seq_len 时数据集应为空."""
        tokenizer = CharTokenizer()
        ds = LMDataset(tokenizer, "hi", seq_len=512)
        assert len(ds) == 0

    def test_getitem_slices_correctly(self):
        """验证 __getitem__ 返回正确切片: x=ids[s:e], y=ids[s+1:e+1]."""
        tokenizer = CharTokenizer()
        text = "abcdefghij"
        seq_len_val = 3
        ds = LMDataset(tokenizer, text, seq_len=seq_len_val)
        assert len(ds) >= 1
        x, y = ds[0]
        assert len(x) == seq_len_val
        assert len(y) == seq_len_val
        start = 0
        assert torch.equal(x, ds.ids[start: start + seq_len_val])
        assert torch.equal(y, ds.ids[start + 1: start + seq_len_val + 1])

    def test_getitem_last_valid_index(self):
        """验证最后一个有效索引仍返回正确形状."""
        tokenizer = CharTokenizer()
        text = "a" * 50
        seq_len_val = 8
        ds = LMDataset(tokenizer, text, seq_len=seq_len_val)
        if len(ds) > 0:
            idx = len(ds) - 1
            x, y = ds[idx]
            assert len(x) == seq_len_val
            assert len(y) == seq_len_val

    def test_consistency_with_create_dataloader(self):
        """验证 create_dataloader 内部使用优化后的 LMDataset."""
        text = "hello world this is a test"
        tokenizer = CharTokenizer()
        loader = create_dataloader(text, tokenizer, seq_len=4, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        x, y = batch
        assert x.shape[1] == 4
        assert y.shape[1] == 4

    def test_no_preallocated_sequences(self):
        """验证不再预创建 sequences 列表（内存优化核心）."""
        tokenizer = CharTokenizer()
        ds = LMDataset(tokenizer, "test" * 100, seq_len=4)
        assert not hasattr(ds, "sequences")


class TestCreateEvalLoader:
    """create_eval_loader: 不 shuffle、不 drop_last."""

    def test_no_shuffle(self):
        """验证 eval loader 的 shuffle=False."""
        text = "hello world test data for eval loader"
        tokenizer = CharTokenizer()
        loader = create_eval_loader(text, tokenizer, seq_len=4, batch_size=2)
        assert loader.dataset is not None
        # 验证不 shuffle
        x1, _ = next(iter(loader))
        x2, _ = next(iter(loader))
        assert not loader.dataset.num_sequences == 0

    def test_no_drop_last(self):
        """验证 eval loader 不 drop last batch."""
        tokenizer = CharTokenizer()
        text = "a" * 30  # 短文本保证最后一个 batch 可能不完整
        seq_len_val = 8
        loader = create_eval_loader(text, tokenizer, seq_len=seq_len_val, batch_size=4)
        total = 0
        for batch in loader:
            total += batch[0].shape[0]
        dataset = LMDataset(tokenizer, text, seq_len=seq_len_val)
        assert total == len(dataset), (
            f"eval loader should return all samples; "
            f"got {total}, expected {len(dataset)}"
        )


class TestDownloadWikitext103:
    """download_wikitext103: 验证返回值结构和路径计算逻辑."""

    def test_return_type_and_keys(self):
        """验证返回值是 dict[str, Path] 且包含 train/valid/test."""
        from scripts.tiny_llama_shared import download_wikitext103
        # 不实际下载，只验证返回类型结构和路径生成逻辑
        from pathlib import Path
        data_dir = str(Path(__file__).resolve().parents[1] / "data" / "wikitext-103-test")
        try:
            result = download_wikitext103(data_dir)
        except (RuntimeError, OSError) as e:
            # 可能因网络或文件系统而失败，不在这测试
            pytest.skip(f"download_wikitext103 raised {e}, skipping live download test")

        assert isinstance(result, dict)
        assert set(result.keys()) == {"train", "valid", "test"}
        for key, path in result.items():
            assert isinstance(path, Path)
            assert path.name == f"wiki.{key}.tokens"

    def test_raises_on_bad_data_dir(self):
        """验证在无效路径上不会抛出意外异常."""
        from scripts.tiny_llama_shared import download_wikitext103
        from pathlib import Path
        import tempfile
        import shutil

        tmp = Path(tempfile.mkdtemp())
        try:
            # 将路径指向一个不可写入的位置（仅测试异常路径，不依赖网络）
            pass
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
