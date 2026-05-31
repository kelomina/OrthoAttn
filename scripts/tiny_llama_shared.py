"""Shared tokenizer, data loading and config for tiny LLaMA comparison."""
from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset, DataLoader


LM_VOCAB = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:\"'()-[]{}<>=+*/@#$%&\n\t"
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
SPECIAL_TOKENS = 4  # pad, bos, eos, unk

LMConfig = {
    "dim": 256,
    "heads": 4,
    "num_layers": 6,
    "vocab_size": len(LM_VOCAB) + SPECIAL_TOKENS + 1,
    "seq_len": 512,
    "batch_size": 8,
    "lr": 3e-4,
    "max_steps": 50000,
    "eval_interval": 1000,
    "warmup_steps": 1000,
    "device": "auto",
    "data_dir": str(Path(__file__).resolve().parents[1] / "data" / "wikitext-2"),
    "dataset": "wikitext2",  # "wikitext2" or "wikitext103"
}


class CharTokenizer:
    """Character-level tokenizer for LM comparison."""

    def __init__(self, vocab: str = LM_VOCAB):
        self.vocab = vocab
        self._char_to_id = {
            ch: i + SPECIAL_TOKENS for i, ch in enumerate(vocab)
        }
        self._id_to_char = {
            i + SPECIAL_TOKENS: ch for i, ch in enumerate(vocab)
        }
        self.vocab_size = len(vocab) + SPECIAL_TOKENS + 1

    def encode(self, text: str) -> list[int]:
        ids = [BOS_ID]
        for ch in text:
            ids.append(self._char_to_id.get(ch, UNK_ID))
        ids.append(EOS_ID)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        chars = []
        for i in ids:
            if i < SPECIAL_TOKENS:
                continue
            chars.append(self._id_to_char.get(i, "?"))
        return "".join(chars)


def download_wikitext2(data_dir: str) -> Path:
    """Download WikiText-2 if not present. Returns path to raw text file."""
    data_path = Path(data_dir)
    raw_path = data_path / "wiki.train.tokens"
    if raw_path.exists():
        return raw_path

    data_path.mkdir(parents=True, exist_ok=True)
    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
    zip_path = data_path / "wikitext-2-v1.zip"

    print(f"Downloading WikiText-2 from {url}...")
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_path)

    return data_path / "wikitext-2" / "wiki.train.tokens"


def load_wikitext2_splits(data_dir: str) -> dict[str, Path | None]:
    """Return WikiText-2 split paths, downloading the archive if needed.

    中文说明:
    - 调用方 / Called by: `scripts.tiny_llama_baseline`, `scripts.tiny_llama_mhdsra2`.
    - 调用对象 / Calls: `download_wikitext2`, `Path.exists`.
    - 作用 / Purpose: 为 tiny LLaMA 对比提供明确 train/validation split；若旧缓存缺少官方
      valid/test 文件，则由调用方对训练文本做固定尾部分割，避免把训练 batch loss 当成验证 PPL。
    - 参数 / Parameters: `data_dir` 是 WikiText-2 数据缓存目录。
    - 返回 / Returns: `{"train": Path, "valid": Path|None, "test": Path|None}`。
    - 错误处理 / Error handling: 若 train 文件不存在，抛出 `RuntimeError`。
    - 副作用 / Side effects: 首次运行可能下载并解压 WikiText-2。

    English documentation:
    Function name:
        load_wikitext2_splits
    Purpose:
        Resolve explicit train/validation/test files for WikiText-2.
    Called by:
        tiny LLaMA training entry points.
    Calls:
        `download_wikitext2`, `Path.exists`.
    Parameters:
        - data_dir: WikiText-2 cache directory.
    Returns:
        Dict with train path and optional valid/test paths.
    Error handling:
        Raises `RuntimeError` when the train split is missing.
    Side effects:
        May download and extract WikiText-2.
    English keywords:
        wikitext2, split, train, validation, perplexity
    """
    train_path = download_wikitext2(data_dir)
    split_dir = train_path.parent
    valid_path = split_dir / "wiki.valid.tokens"
    test_path = split_dir / "wiki.test.tokens"
    if not train_path.exists():
        raise RuntimeError(f"WikiText-2 train file missing after download: {train_path}")
    return {
        "train": split_dir / "wiki.train.tokens",
        "valid": valid_path if valid_path.exists() else None,
        "test": test_path if test_path.exists() else None,
    }


def split_train_validation_text(
    text: str,
    *,
    validation_chars: int = 200_000,
) -> tuple[str, str]:
    """Split one train-only text file into train and validation tails.

    中文说明:
    - 调用方 / Called by: tiny LLaMA scripts when WikiText-2 valid split is absent.
    - 调用对象 / Calls: string slicing.
    - 作用 / Purpose: 兼容旧 `data/wikitext-2/wiki.train.tokens` 缓存，同时保证 PPL 来自未训练尾部文本。
    - 参数 / Parameters: `text` 是完整训练文本；`validation_chars` 是尾部验证字符预算。
    - 返回 / Returns: `(train_text, validation_text)`。
    - 错误处理 / Error handling: 文本过短时仍至少保留一半用于训练、一半用于验证。
    - 副作用 / Side effects: 无。

    English documentation:
    Function name:
        split_train_validation_text
    Purpose:
        Build a deterministic validation tail when only a train file exists.
    Called by:
        tiny LLaMA scripts.
    Calls:
        String slicing.
    Parameters:
        - text: full training text.
        - validation_chars: validation tail budget.
    Returns:
        `(train_text, validation_text)`.
    Error handling:
        Short text is split in half to keep both sides non-empty.
    Side effects:
        None.
    English keywords:
        validation split, fallback, train tail, perplexity
    """
    if len(text) < 2:
        raise ValueError("Cannot split train/validation text with fewer than two characters.")
    split_size = min(max(1, int(validation_chars)), max(1, len(text) // 5))
    split_at = max(1, len(text) - split_size)
    return text[:split_at], text[split_at:]


def download_wikitext103(data_dir: str) -> dict[str, Path]:
    """Download WikiText-103 via HuggingFace datasets, cache as .tokens files.

    通过 HuggingFace datasets 库下载 WikiText-103，缓存为 .tokens 文件。
    S3 源已失效，改用 datasets 库加载后写出到本地文件。

    Returns {"train": Path, "valid": Path, "test": Path}.
    Raises RuntimeError if download fails.

    Returns:
        {"train": Path, "valid": Path, "test": Path}

    Raises:
        RuntimeError: 下载或写出文件失败时抛出
    """
    data_path = Path(data_dir)
    extract_path = data_path / "wikitext-103"
    splits_map = {
        "train": extract_path / "wiki.train.tokens",
        "valid": extract_path / "wiki.valid.tokens",
        "test": extract_path / "wiki.test.tokens",
    }

    # 如果三个切分文件都已存在，直接返回
    if all(p.exists() for p in splits_map.values()):
        return splits_map

    data_path.mkdir(parents=True, exist_ok=True)
    extract_path.mkdir(parents=True, exist_ok=True)

    print("Loading WikiText-103 via HuggingFace datasets...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    except Exception as e:
        raise RuntimeError(f"Failed to download WikiText-103 via datasets: {e}") from e

    hf_split_map = {"train": "train", "valid": "validation", "test": "test"}
    for key, path in splits_map.items():
        hf_split = hf_split_map[key]
        print(f"  Writing {key} split ({len(dataset[hf_split])} lines) to {path}...")
        try:
            text = "\n".join(dataset[hf_split]["text"])
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            raise RuntimeError(f"Failed to write {key} split to {path}: {e}") from e

    # 验证文件完整性
    missing = [name for name, p in splits_map.items() if not p.exists()]
    if missing:
        raise RuntimeError(
            f"WikiText-103 file writing incomplete: missing splits {missing}. "
            f"Expected files in {extract_path}"
        )

    return splits_map


def load_text(file_path: Path, max_chars: int | None = None) -> str:
    """Load text file, optionally truncated."""
    text = file_path.read_text(encoding="utf-8")
    if max_chars:
        text = text[:max_chars]
    return text


class LMDataset(Dataset):
    """Language modeling dataset with fixed-length sequences.

    内存高效的 LM 数据集实现。将所有 token ID 存储在单个连续 Tensor 中，
    __getitem__ 通过切片直接索引，避免预创建大量独立 Tensor。

    Memory-efficient LM dataset. Stores all token IDs in a single contiguous
    tensor and slices on-the-fly in __getitem__.
    """

    def __init__(self, tokenizer: CharTokenizer, text: str, seq_len: int):
        self.seq_len = seq_len
        # 避免创建 Python list（518M ints ≈ 14.5GB 峰值内存）
        # 使用 numpy 向量化转换字符到 token ID
        import numpy as np
        n_tokens = len(text) + 2
        # 构建 ASCII 查找表 (0-127)
        lookup = np.full(128, UNK_ID, dtype=np.int64)
        for ch, tid in tokenizer._char_to_id.items():
            c = ord(ch)
            if c < 128:
                lookup[c] = tid
        # 向量化转换：text → bytes → ids
        raw = text.encode("ascii", errors="replace")
        ids_array = lookup[np.frombuffer(raw, dtype=np.uint8)]
        # 组装为 torch Tensor（含 BOS/EOS）
        self.ids = torch.empty(n_tokens, dtype=torch.long)
        self.ids[0] = BOS_ID
        self.ids[1:-1] = torch.from_numpy(ids_array)
        self.ids[-1] = EOS_ID
        self.num_sequences = (n_tokens - 1) // seq_len

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        return (
            self.ids[start: start + self.seq_len],
            self.ids[start + 1: start + self.seq_len + 1],
        )


def create_dataloader(
    text: str,
    tokenizer: CharTokenizer,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = LMDataset(tokenizer, text, seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def create_eval_loader(
    text: str,
    tokenizer: CharTokenizer,
    seq_len: int,
    batch_size: int,
) -> DataLoader:
    """Create a non-shuffled DataLoader for evaluation (no drop_last).

    创建用于评测的 DataLoader，不 shuffle、不 drop last，保证所有序列都被评估。

    Args:
        text: 原始文本
        tokenizer: 字符级 tokenizer
        seq_len: 序列长度
        batch_size: 批次大小

    Returns:
        不 shuffle、不 drop_last 的 DataLoader
    """
    dataset = LMDataset(tokenizer, text, seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)
